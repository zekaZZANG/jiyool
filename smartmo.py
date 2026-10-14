import io
import math
import os
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
    page_title="Exam PDF Restorer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 1rem;
        max-width: 1500px;
    }
    div[data-testid="stVerticalBlock"] {
        gap: 0.65rem;
    }
    div[data-testid="stMetric"] {
        padding: 0.5rem;
        border: 1px solid rgba(128,128,128,0.2);
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📄 시험지 PDF 복원")
st.caption("스캔 시험지의 기울기 보정 · 배경 백색화 · 글자 선명도 개선 · 병렬 페이지 처리")


# =========================
# 이미지 처리 함수
# =========================

def render_page(page, dpi=250):
    """PDF 페이지를 이미지로 변환한다."""
    scale = dpi / 72.0
    pix = page.get_pixmap(
        matrix=fitz.Matrix(scale, scale),
        alpha=False,
    )

    arr = np.frombuffer(
        pix.samples, dtype=np.uint8
    ).reshape(pix.height, pix.width, pix.n)

    if pix.n == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
    elif pix.n == 1:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)

    return arr


def estimate_skew_angle(gray):
    """페이지의 작은 기울기를 추정한다."""
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 1800,
        threshold=max(50, min(gray.shape) // 8),
        minLineLength=max(40, gray.shape[1] // 8),
        maxLineGap=12,
    )

    if lines is None:
        return 0.0

    angles = []

    for line in lines[:, 0]:
        x1, y1, x2, y2 = map(int, line)
        angle = math.degrees(
            math.atan2(y2 - y1, x2 - x1)
        )

        if -15 <= angle <= 15:
            angles.append(angle)

    if not angles:
        return 0.0

    return float(np.median(angles))


def rotate_bound(image, angle):
    """이미지 회전 시 모서리가 잘리지 않도록 회전한다."""
    h, w = image.shape[:2]
    center = (w / 2, h / 2)

    matrix = cv2.getRotationMatrix2D(
        center, angle, 1.0
    )

    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])

    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    matrix[0, 2] += new_w / 2 - center[0]
    matrix[1, 2] += new_h / 2 - center[1]

    return cv2.warpAffine(
        image,
        matrix,
        (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def detect_and_warp_paper(image):
    """
    종이의 사각형 경계가 명확할 때 원근을 보정한다.
    종이 경계가 불분명하면 원본을 유지한다.
    """
    h, w = image.shape[:2]

    gray = cv2.cvtColor(
        image, cv2.COLOR_RGB2GRAY
    )

    blurred = cv2.GaussianBlur(
        gray, (5, 5), 0
    )

    edges = cv2.Canny(
        blurred, 30, 100
    )

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True,
    )[:15]

    for contour in contours:
        perimeter = cv2.arcLength(
            contour, True
        )

        approx = cv2.approxPolyDP(
            contour, 0.02 * perimeter, True
        )

        if (
            len(approx) != 4
            or cv2.contourArea(approx) < 0.35 * h * w
        ):
            continue

        pts = approx.reshape(
            4, 2
        ).astype(np.float32)

        sums = pts.sum(axis=1)
        diffs = np.diff(
            pts, axis=1
        ).reshape(-1)

        rect = np.array(
            [
                pts[np.argmin(sums)],
                pts[np.argmin(diffs)],
                pts[np.argmax(sums)],
                pts[np.argmax(diffs)],
            ],
            dtype=np.float32,
        )

        tl, tr, br, bl = rect

        width = int(
            max(
                np.linalg.norm(br - bl),
                np.linalg.norm(tr - tl),
            )
        )

        height = int(
            max(
                np.linalg.norm(tr - br),
                np.linalg.norm(tl - bl),
            )
        )

        if width < 100 or height < 100:
            continue

        destination = np.array(
            [
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1],
            ],
            dtype=np.float32,
        )

        transform = cv2.getPerspectiveTransform(
            rect, destination
        )

        return cv2.warpPerspective(
            image,
            transform,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )

    return image


def normalize_background(image, strength=0.65):
    """종이의 회색 배경과 불균일한 조명을 보정한다."""
    gray = cv2.cvtColor(
        image, cv2.COLOR_RGB2GRAY
    )

    kernel_size = max(
        31, (min(gray.shape[:2]) // 12) | 1
    )

    if kernel_size % 2 == 0:
        kernel_size += 1

    background = cv2.GaussianBlur(
        gray,
        (kernel_size, kernel_size),
        0,
    )

    background = np.maximum(
        background, 20
    ).astype(np.float32)

    corrected = (
        gray.astype(np.float32)
        * (245.0 / background)
    )

    corrected = np.clip(
        corrected, 0, 255
    )

    result = (
        gray.astype(np.float32) * (1 - strength)
        + corrected * strength
    )

    result = np.clip(
        result, 0, 255
    ).astype(np.uint8)

    return cv2.cvtColor(
        result, cv2.COLOR_GRAY2RGB
    )


def process_image(image, settings):
    """설정된 복원 기능만 적용한다."""
    result = image.copy()

    if settings["perspective"]:
        result = detect_and_warp_paper(result)

    if settings["deskew"]:
        gray = cv2.cvtColor(
            result, cv2.COLOR_RGB2GRAY
        )

        angle = estimate_skew_angle(gray)

        if abs(angle) >= 0.15:
            result = rotate_bound(
                result, -angle
            )

    if settings["background"]:
        result = normalize_background(
            result,
            settings["background_strength"],
        )

    if settings["denoise"]:
        result = cv2.fastNlMeansDenoisingColored(
            result,
            None,
            4,
            4,
            7,
            21,
        )

    if settings["sharpen"]:
        blurred = cv2.GaussianBlur(
            result, (0, 0), 1.1
        )

        result = cv2.addWeighted(
            result,
            1.35,
            blurred,
            -0.35,
            0,
        )

    if settings["contrast"]:
        lab = cv2.cvtColor(
            result, cv2.COLOR_RGB2LAB
        )

        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=1.5,
            tileGridSize=(8, 8),
        )

        l = clahe.apply(l)

        result = cv2.cvtColor(
            cv2.merge((l, a, b)),
            cv2.COLOR_LAB2RGB,
        )

    return result


# =========================
# 병렬 페이지 처리
# =========================

def process_single_page(
    pdf_bytes,
    page_index,
    dpi,
    settings,
):
    """
    각 작업이 별도의 PDF 문서를 열어 처리한다.
    PyMuPDF 문서 객체를 여러 스레드에서 공유하지 않는다.
    """
    doc = None

    try:
        doc = fitz.open(
            stream=pdf_bytes,
            filetype="pdf",
        )

        page = doc[page_index]

        page_size = (
            page.rect.width,
            page.rect.height,
        )

        image = render_page(
            page, dpi=dpi
        )

        image = process_image(
            image, settings
        )

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

    finally:
        if doc is not None:
            doc.close()


def images_to_pdf(results, original_doc):
    """처리 결과를 원래 페이지 크기에 맞춰 PDF로 만든다."""
    output = fitz.open()

    try:
        for result in results:
            page_index = result["index"]
            image = result["image"]

            original_page = original_doc[
                page_index
            ]

            page = output.new_page(
                width=original_page.rect.width,
                height=original_page.rect.height,
            )

            buffer = io.BytesIO()

            Image.fromarray(image).save(
                buffer,
                format="PNG",
                optimize=True,
            )

            page.insert_image(
                page.rect,
                stream=buffer.getvalue(),
            )

        return output.tobytes(
            deflate=True,
            garbage=4,
        )

    finally:
        output.close()


def get_max_workers(page_count):
    """페이지 수에 따라 병렬 작업 수를 제한한다."""
    cpu_count = os.cpu_count() or 2

    # 메모리 사용량이 큰 고해상도 이미지 처리를 고려해 제한한다.
    return max(
        1,
        min(page_count, cpu_count, 4),
    )


# =========================
# 메인 인터페이스
# =========================

uploaded_files = st.file_uploader(
    "PDF 업로드",
    type=["pdf"],
    accept_multiple_files=True,
    help="여러 파일을 한 번에 올릴 수 있습니다.",
)

if not uploaded_files:
    st.info(
        "PDF를 업로드하면 페이지 범위와 복원 옵션을 설정할 수 있습니다."
    )
    st.stop()


# 파일 선택
file_names = [
    f"{i + 1}. {file.name}"
    for i, file in enumerate(uploaded_files)
]

selected_label = st.selectbox(
    "처리할 PDF",
    file_names,
)

selected_index = file_names.index(
    selected_label
)

selected_file = uploaded_files[
    selected_index
]

pdf_bytes = selected_file.getvalue()

try:
    source_doc = fitz.open(
        stream=pdf_bytes,
        filetype="pdf",
    )
except Exception as exc:
    st.error(
        f"PDF를 열 수 없습니다: {exc}"
    )
    st.stop()


page_count = len(source_doc)

# 상단 정보
metric1, metric2, metric3 = st.columns(3)

metric1.metric(
    "전체 페이지",
    f"{page_count}페이지",
)

metric2.metric(
    "파일 크기",
    f"{len(pdf_bytes) / (1024 * 1024):.1f} MB",
)

metric3.metric(
    "병렬 작업 수",
    f"{get_max_workers(page_count)}개",
)


# 설정과 미리보기를 같은 화면에 배치
control_col, preview_col = st.columns(
    [0.9, 1.5],
    gap="medium",
)


with control_col:
    st.subheader("⚙️ 복원 설정")

    range_col1, range_col2 = st.columns(2)

    with range_col1:
        start_page = st.number_input(
            "시작 페이지",
            min_value=1,
            max_value=page_count,
            value=1,
            step=1,
        )

    with range_col2:
        end_page = st.number_input(
            "끝 페이지",
            min_value=1,
            max_value=page_count,
            value=page_count,
            step=1,
        )

    st.caption(
        f"처리 대상: {max(0, int(end_page) - int(start_page) + 1)}페이지"
    )

    st.markdown("**보정 기능**")

    deskew = st.checkbox(
        "기울기 보정",
        value=True,
    )

    perspective = st.checkbox(
        "종이 원근 보정",
        value=False,
        help="종이 경계가 뚜렷한 사진에서 유용합니다.",
    )

    background = st.checkbox(
        "종이 배경 백색화",
        value=True,
    )

    denoise = st.checkbox(
        "노이즈 제거",
        value=True,
    )

    sharpen = st.checkbox(
        "글자 선명도 개선",
        value=True,
    )

    contrast = st.checkbox(
        "명암 개선",
        value=False,
    )

    with st.expander
