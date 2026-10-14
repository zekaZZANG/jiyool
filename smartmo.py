
import io
import math
import os
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import fitz
import numpy as np
import streamlit as st
from PIL import Image


# =========================
# 기본 설정
# =========================

st.set_page_config(
    page_title="시험지 PDF 복원",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

cv2.setUseOptimized(True)

if "result_pdf" not in st.session_state:
    st.session_state["result_pdf"] = None

if "result_name" not in st.session_state:
    st.session_state["result_name"] = None

if "result_key" not in st.session_state:
    st.session_state["result_key"] = None


st.markdown(
    """
    <style>
    .block-container {
        padding-top: 0.8rem;
        padding-bottom: 0.8rem;
        max-width: 1450px;
    }
    div[data-testid="stVerticalBlock"] {
        gap: 0.45rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📄 시험지 PDF 복원")
st.caption(
    "기울기 보정 · 내용 중앙 정렬 · 뒷장 비침 감소 · "
    "배경 백색화 · 글자 선명도 개선"
)


# =========================
# PDF 렌더링
# =========================

def render_page(page, dpi=200):
    scale = dpi / 72.0

    pix = page.get_pixmap(
        matrix=fitz.Matrix(scale, scale),
        colorspace=fitz.csRGB,
        alpha=False,
    )

    image = np.frombuffer(
        pix.samples, dtype=np.uint8
    ).reshape(pix.height, pix.width, 3)

    return image.copy()


# =========================
# 빠른 기울기 보정
# =========================

def estimate_skew_angle(gray):
    h, w = gray.shape
    target_w = min(w, 900)

    if w > target_w:
        ratio = target_w / w
        small = cv2.resize(
            gray,
            (target_w, max(1, int(h * ratio))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        small = gray

    edges = cv2.Canny(small, 60, 160)

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 1800,
        threshold=max(35, small.shape[1] // 12),
        minLineLength=max(35, small.shape[1] // 7),
        maxLineGap=10,
    )

    if lines is None:
        return 0.0

    angles = []

    for line in lines[:, 0]:
        x1, y1, x2, y2 = map(int, line)

        angle = math.degrees(
            math.atan2(y2 - y1, x2 - x1)
        )

        length = math.hypot(x2 - x1, y2 - y1)

        if -10 <= angle <= 10:
            if length >= small.shape[1] * 0.08:
                angles.append(angle)

    if not angles:
        return 0.0

    return float(np.median(angles))


def rotate_bound(image, angle):
    h, w = image.shape[:2]
    center = (w / 2, h / 2)

    matrix = cv2.getRotationMatrix2D(
        center, angle, 1.0
    )

    cos_value = abs(matrix[0, 0])
    sin_value = abs(matrix[0, 1])

    new_w = int(h * sin_value + w * cos_value)
    new_h = int(h * cos_value + w * sin_value)

    matrix[0, 2] += new_w / 2 - center[0]
    matrix[1, 2] += new_h / 2 - center[1]

    return cv2.warpAffine(
        image,
        matrix,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


# =========================
# 내용 중앙 정렬
# =========================

def center_content(image):
    h, w = image.shape[:2]

    gray = cv2.cvtColor(
        image, cv2.COLOR_RGB2GRAY
    )

    margin_y = max(2, int(h * 0.015))
    margin_x = max(2, int(w * 0.015))

    inner = gray[
        margin_y:h - margin_y,
        margin_x:w - margin_x,
    ]

    if inner.size == 0:
        return image

    # 어두운 글자와 선을 검출한다.
    mask = cv2.threshold(
        inner,
        215,
        255,
        cv2.THRESH_BINARY_INV,
    )[1]

    points = cv2.findNonZero(mask)

    if points is None:
        return image

    x, y, bw, bh = cv2.boundingRect(points)

    # 페이지 전체가 내용으로 잡힌 경우 이동하지 않는다.
    if bw > inner.shape[1] * 0.97:
        return image

    if bh > inner.shape[0] * 0.97:
        return image

    x1 = x + margin_x
    y1 = y + margin_y
    x2 = x1 + bw
    y2 = y1 + bh

    content_cx = (x1 + x2) / 2
    content_cy = (y1 + y2) / 2

    dx = int(round(w / 2 - content_cx))
    dy = int(round(h / 2 - content_cy))

    if abs(dx) < w * 0.005 and abs(dy) < h * 0.005:
        return image

    matrix = np.float32([
        [1, 0, dx],
        [0, 1, dy],
    ])

    return cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


# =========================
# 자동 배경 백색화 및 뒷장 비침 감소
# =========================

def remove_bleed_and_whiten(image, strength=0.65):
    """
    종이의 불균일한 밝기를 보정하고,
    어두운 앞면 글자는 최대한 보존하면서
    옅은 회색 배경과 뒷장 비침을 줄인다.

    이 기능은 별도 체크박스 없이 항상 적용한다.
    """
    gray = cv2.cvtColor(
        image, cv2.COLOR_RGB2GRAY
    )

    h, w = gray.shape

    # 작은 이미지에서 배경 밝기를 추정해 계산량을 줄인다.
    small_w = max(1, w // 4)
    small_h = max(1, h // 4)

    small = cv2.resize(
        gray,
        (small_w, small_h),
        interpolation=cv2.INTER_AREA,
    )

    background_small = cv2.GaussianBlur(
        small,
        (0, 0),
        sigmaX=9,
        sigmaY=9,
    )

    background = cv2.resize(
        background_small,
        (w, h),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)

    background = np.maximum(background, 30)

    # 국소 배경 밝기에 맞춰 회색 종이를 보정한다.
    corrected = (
        gray.astype(np.float32)
        * (245.0 / background)
    )

    corrected = np.clip(
        corrected, 0, 255
    ).astype(np.uint8)

    # 밝은 영역을 흰색에 가깝게 매핑한다.
    # 진한 글자와 선은 비교적 그대로 유지한다.
    lut = np.arange(256, dtype=np.float32)

    start = 150.0
    end = 220.0

    middle = (
        start
        + (lut - start)
        * ((255.0 - start) / (end - start))
    )

    lut = np.where(
        lut <= start,
        lut,
        np.where(
            lut >= end,
            255.0,
            middle,
        ),
    )

    lut = np.clip(lut, 0, 255).astype(np.uint8)

    whitened = cv2.LUT(corrected, lut)

    # 강도에 따라 원본 대비 보정량을 조절한다.
    result = cv2.addWeighted(
        corrected,
        1.0 - strength,
        whitened,
        strength,
        0,
    )

    return cv2.cvtColor(
        result, cv2.COLOR_GRAY2RGB
    )


# =========================
# 이미지 복원
# =========================

def process_image(image, settings):
    result = image

    if settings["deskew"]:
        gray = cv2.cvtColor(
            result, cv2.COLOR_RGB2GRAY
        )

        angle = estimate_skew_angle(gray)

        if abs(angle) >= 0.15:
            result = rotate_bound(result, -angle)

    if settings["center"]:
        result = center_content(result)

    # 항상 자동 적용: 별도 체크박스 없음
    result = remove_bleed_and_whiten(
        result,
        settings["background_strength"],
    )

    if settings["denoise"]:
        gray = cv2.cvtColor(
            result, cv2.COLOR_RGB2GRAY
        )

        gray = cv2.fastNlMeansDenoising(
            gray,
            None,
            h=3,
            templateWindowSize=7,
            searchWindowSize=15,
        )

        result = cv2.cvtColor(
            gray, cv2.COLOR_GRAY2RGB
        )

    if settings["sharpen"]:
        blurred = cv2.GaussianBlur(
            result, (0, 0), 0.8
        )

        result = cv2.addWeighted(
            result,
            1.2,
            blurred,
            -0.2,
            0,
        )

    return result


# =========================
# 캐시된 미리보기
# =========================

@st.cache_data(show_spinner=False, max_entries=12)
def make_preview(pdf_bytes, page_index, settings_key):
    settings = dict(settings_key)

    with fitz.open(
        stream=pdf_bytes,
        filetype="pdf",
    ) as doc:
        image = render_page(
            doc[page_index],
            dpi=100,
        )

    return process_image(image, settings)


# =========================
# 병렬 페이지 처리
# =========================

def process_single_page(pdf_bytes, page_index, dpi, settings):
    try:
        with fitz.open(
            stream=pdf_bytes,
            filetype="pdf",
        ) as doc:
            page = doc[page_index]

            page_size = (
                page.rect.width,
                page.rect.height,
            )

            image = render_page(
                page,
                dpi=dpi,
            )

        image = process_image(image, settings)

        return {
            "index": page_index,
            "image": image,
            "size": page_size,
            "error": None,
        }

    except Exception as exc:
        return {
            "index": page_index,
            "image": None,
            "size": None,
            "error": str(exc),
        }


def get_max_workers(page_count):
    cpu_count = os.cpu_count() or 2
    return max(1, min(page_count, cpu_count, 3))


def images_to_pdf(results):
    output = fitz.open()

    try:
        for result in results:
            width, height = result["size"]

            page = output.new_page(
                width=width,
                height=height,
            )

            buffer = io.BytesIO()

            # 빠른 PNG 압축. 이미지 해상도는 유지한다.
            Image.fromarray(
                result["image"]
            ).save(
                buffer,
                format="PNG",
                compress_level=1,
            )

            page.insert_image(
                page.rect,
                stream=buffer.getvalue(),
            )

        return output.tobytes(
            garbage=1,
            deflate=False,
        )

    finally:
        output.close()


# =========================
# 업로드
# =========================

uploaded_files = st.file_uploader(
    "PDF 업로드",
    type=["pdf"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("PDF를 업로드해 줘.")
    st.stop()

file_names = [
    f"{i + 1}. {file.name}"
    for i, file in enumerate(uploaded_files)
]

selected_label = st.selectbox(
    "처리할 PDF",
    file_names,
)

selected_index = file_names.index(selected_label)
selected_file = uploaded_files[selected_index]
pdf_bytes = selected_file.getvalue()

# 파일이 바뀌면 이전 파일의 다운로드 결과를 무효화한다.
file_hash = hashlib.sha256(pdf_bytes).hexdigest()

try:
    source_doc = fitz.open(
        stream=pdf_bytes,
        filetype="pdf",
    )
except Exception as exc:
    st.error(f"PDF를 열 수 없어: {exc}")
    st.stop()

page_count = len(source_doc)

m1, m2 = st.columns(2)

m1.metric(
    "전체 페이지",
    f"{page_count}페이지",
)

m2.metric(
    "파일 크기",
    f"{len(pdf_bytes) / 1048576:.1f} MB",
)


# =========================
# 설정 및 미리보기
# =========================

control_col, preview_col = st.columns([0.9, 1.5])

with control_col:
    st.subheader("⚙️ 복원 설정")

    r1, r2 = st.columns(2)

    with r1:
        start_page = st.number_input(
            "시작 페이지",
            min_value=1,
            max_value=page_count,
            value=1,
            step=1,
        )

    with r2:
        end_page = st.number_input(
            "끝 페이지",
            min_value=1,
            max_value=page_count,
            value=page_count,
            step=1,
        )

    deskew = st.checkbox(
        "기울기 보정",
        value=True,
    )

    center = st.checkbox(
        "내용 중앙 정렬",
        value=True,
    )

    sharpen = st.checkbox(
        "글자 선명도 개선",
        value=True,
    )

    denoise = st.checkbox(
        "노이즈 제거 (느릴 수 있음)",
        value=False,
    )

    with st.expander("세부 설정"):
        background_strength = st.slider(
            "배경 백색화 및 비침 감소 강도",
            min_value=0.0,
            max_value=1.0,
            value=0.65,
            step=0.05,
        )

        dpi = st.select_slider(
            "최종 출력 해상도",
            options=[120, 150, 200, 250, 300],
            value=200,
        )

        st.caption(
            "비침 감소는 자동 적용돼. 강도를 높이면 "
            "옅은 글씨도 함께 희미해질 수 있어."
        )

    settings = {
        "deskew": deskew,
        "center": center,
        "sharpen": sharpen,
        "denoise": denoise,
        "background_strength": background_strength,
    }

    settings_key = tuple(sorted(settings.items()))

with preview_col:
    st.subheader("👁️ 미리보기")

    if start_page <= end_page:
        preview_page = st.number_input(
            "미리 볼 페이지",
            min_value=int(start_page),
            max_value=int(end_page),
            value=int(start_page),
            step=1,
        )

        left, right = st.columns(2)

        try:
            original_preview = render_page(
                source_doc[int(preview_page) - 1],
                dpi=100,
            )

            processed_preview = make_preview(
                pdf_bytes,
                int(preview_page) - 1,
                settings_key,
            )

            with left:
                st.caption("원본")
                st.image(
                    original_preview,
                    use_container_width=True,
                )

            with right:
                st.caption("복원 결과")
                st.image(
                    processed_preview,
                    use_container_width=True,
                )

        except Exception as exc:
            st.error(f"미리보기 오류: {exc}")

    else:
        st.warning(
            "끝 페이지는 시작 페이지보다 작을 수 없어."
        )


# =========================
# 다운로드 식별 키
# =========================

job_key = hashlib.sha256(
    repr((
        file_hash,
        int(start_page),
        int(end_page),
        int(dpi),
        settings_key,
    )).encode("utf-8")
).hexdigest()


# =========================
# 실행
# =========================

st.divider()

valid_range = int(start_page) <= int(end_page)

if st.button(
    "🚀 PDF 복원 시작",
    type="primary",
    use_container_width=True,
    disabled=not valid_range,
):
    selected_indices = list(
        range(
            int(start_page) - 1,
            int(end_page),
        )
    )

    total = len(selected_indices)
    results_by_index = {}
    failures = []

    progress = st.progress(0)
    status = st.empty()

    workers = get_max_workers(total)

    status.info(
        f"{total}페이지 처리 시작 · 동시 작업 {workers}개"
    )

    with ThreadPoolExecutor(
        max_workers=workers
    ) as executor:
        futures = {
            executor.submit(
                process_single_page,
                pdf_bytes,
                index,
                int(dpi),
                settings,
            ): index
            for index in selected_indices
        }

        for completed, future in enumerate(
            as_completed(futures),
            start=1,
        ):
            index = futures[future]

            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "index": index,
                    "image": None,
                    "size": None,
                    "error": str(exc),
                }

            if result["error"] is not None:
                failures.append(
                    (index + 1, result["error"])
                )

                # 처리 실패 시 원본 페이지 이미지로 대체한다.
                try:
                    with fitz.open(
                        stream=pdf_bytes,
                        filetype="pdf",
                    ) as fallback_doc:
                        page = fallback_doc[index]

                        size = (
                            page.rect.width,
                            page.rect.height,
                        )

                        image = render_page(
                            page,
                            dpi=int(dpi),
                        )

                    result.update({
                        "image": image,
                        "size": size,
                    })

                except Exception as fallback_exc:
                    failures.append(
                        (
                            index + 1,
                            f"원본 대체 실패: {fallback_exc}",
                        )
                    )

                    progress.progress(completed / total)
                    continue

            results_by_index[index] = result

            progress.progress(completed / total)
            status.write(
                f"페이지 처리 중: {completed}/{total}"
            )

    ordered_results = [
        results_by_index[index]
        for index in selected_indices
        if index in results_by_index
        and results_by_index[index]["image"] is not None
    ]

    if len(ordered_results) != total:
        st.error(
            "일부 페이지를 처리하지 못했어. "
            "오류 내용을 확인하거나 해상도를 낮춰 줘."
        )

    elif ordered_results:
        try:
            status.info("PDF 생성 중...")

            result_pdf = images_to_pdf(ordered_results)

            base_name = Path(
                selected_file.name
            ).stem

            output_name = (
                f"{base_name}_{start_page}-{end_page}_restored.pdf"
            )

            # 다운로드 버튼이 다시 실행되어도 결과를 유지한다.
            st.session_state["result_pdf"] = result_pdf
            st.session_state["result_name"] = output_name
            st.session_state["result_key"] = job_key
            st.session_state["result_failures"] = failures

            status.success(
                f"완료! {len(ordered_results)}페이지 처리"
            )

        except Exception as exc:
            st.error(f"PDF 생성에 실패했어: {exc}")


# =========================
# 다운로드 버튼
# 실행 버튼과 분리해 결과를 유지한다.
# =========================

if (
    st.session_state.get("result_pdf") is not None
    and st.session_state.get("result_key") == job_key
):
    st.success("복원된 PDF를 다운로드할 수 있어.")

    failures = st.session_state.get(
        "result_failures", []
    )

    if failures:
        with st.expander("일부 페이지의 처리 오류 확인"):
            for page_no, message in failures:
                st.write(f"- {page_no}페이지: {message}")

    st.download_button(
        label="📥 복원된 PDF 다운로드",
        data=st.session_state["result_pdf"],
        file_name=st.session_state["result_name"],
        mime="application/pdf",
        type="primary",
        use_container_width=True,
        key=f"download_{job_key}",
    )

elif st.session_state.get("result_pdf") is not None:
    st.caption(
        "현재 파일 또는 설정이 이전 결과와 달라졌어. "
        "새 PDF를 복원하면 다운로드 버튼이 표시돼."
    )

source_doc.close()
