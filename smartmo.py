import io
import math
import os
import tempfile
import zipfile
from pathlib import Path

import cv2
import fitz
import numpy as np
import pytesseract
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from pytesseract import Output


st.set_page_config(
    page_title="Exam PDF Restorer",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Exam PDF Restorer")
st.caption("스캔 시험지 PDF 복원 · OCR 기반 텍스트 재구성")
st.warning(
    "시험지 정확성 안내: 이미지 복원 모드는 원본 내용을 보존하는 데 초점을 둡니다. "
    "OCR 재구성 모드는 인식 오류가 생길 수 있으며 수식·도형·표를 완벽하게 재현하지 못합니다. "
    "중요한 시험지는 반드시 원본과 결과를 대조하세요."
)


# ---------- Image processing helpers ----------

def render_page(page, dpi=250):
    scale = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
    return arr


def estimate_skew_angle(gray):
    # Estimate small rotation from dark pixels; avoid extreme correction.
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 1800, threshold=max(50, min(gray.shape) // 8),
        minLineLength=max(40, gray.shape[1] // 8), maxLineGap=12
    )
    if lines is None:
        return 0.0
    angles = []
    for line in lines[:, 0]:
        x1, y1, x2, y2 = map(int, line)
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        if -15 <= angle <= 15:
            angles.append(angle)
    if not angles:
        return 0.0
    return float(np.median(angles))


def rotate_bound(image, angle):
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    matrix[0, 2] += (new_w / 2) - center[0]
    matrix[1, 2] += (new_h / 2) - center[1]
    return cv2.warpAffine(
        image, matrix, (new_w, new_h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )


def detect_and_warp_paper(image):
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Use a blurred edge map to find a large quadrilateral, if present.
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4 and cv2.contourArea(approx) > 0.35 * h * w:
            pts = approx.reshape(4, 2).astype(np.float32)
            # Order points: top-left, top-right, bottom-right, bottom-left.
            s = pts.sum(axis=1)
            d = np.diff(pts, axis=1).reshape(-1)
            rect = np.array([
                pts[np.argmin(s)], pts[np.argmin(d)],
                pts[np.argmax(s)], pts[np.argmax(d)]
            ], dtype=np.float32)
            tl, tr, br, bl = rect
            width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
            height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
            if width < 100 or height < 100:
                continue
            dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
            transform = cv2.getPerspectiveTransform(rect, dst)
            return cv2.warpPerspective(
                image, transform, (width, height), flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255)
            )
    return image


def normalize_background(image, strength=0.7):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Estimate slowly varying paper illumination. Keep dark text detail.
    kernel_size = max(31, (min(gray.shape[:2]) // 12) | 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    background = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    background = np.maximum(background, 20).astype(np.float32)
    corrected = gray.astype(np.float32) * (245.0 / background)
    corrected = np.clip(corrected, 0, 255)
    # Blend original luminance with correction to avoid overprocessing.
    out = gray.astype(np.float32) * (1 - strength) + corrected * strength
    out = np.clip(out, 0, 255).astype(np.uint8)
    # Preserve subtle anti-aliasing instead of hard thresholding.
    return cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)


def process_image(image, settings):
    result = image.copy()
    if settings["perspective"]:
        result = detect_and_warp_paper(result)
    if settings["deskew"]:
        gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
        angle = estimate_skew_angle(gray)
        if abs(angle) >= 0.15:
            result = rotate_bound(result, -angle)
    if settings["background"]:
        result = normalize_background(result, settings["background_strength"])
    if settings["denoise"]:
        result = cv2.fastNlMeansDenoisingColored(result, None, 4, 4, 7, 21)
    if settings["sharpen"]:
        blurred = cv2.GaussianBlur(result, (0, 0), 1.1)
        result = cv2.addWeighted(result, 1.35, blurred, -0.35, 0)
    if settings["contrast"]:
        lab = cv2.cvtColor(result, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        l = clahe.apply(l)
        result = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)
    return result


# ---------- OCR reconstruction helpers ----------

def find_korean_font():
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def ocr_reconstruct(image, lang="kor+eng", threshold=45):
    """
    Experimental text-first reconstruction:
    detects OCR word boxes and redraws recognized text on a white canvas.
    This does not faithfully reconstruct equations, diagrams, tables or page design.
    """
    pil = Image.fromarray(image).convert("RGB")
    data = pytesseract.image_to_data(pil, lang=lang, output_type=Output.DICT, config="--psm 3")
    font_path = find_korean_font()
    canvas = Image.new("RGB", pil.size, "white")
    draw = ImageDraw.Draw(canvas)
    low_confidence = 0
    drawn = 0

    n = len(data["text"])
    for i in range(n):
        text = (data["text"][i] or "").strip()
        try:
            conf = float(data["conf"][i])
        except (ValueError, TypeError):
            conf = -1
        if not text or conf < threshold:
            if text:
                low_confidence += 1
            continue
        x, y, w, h = [int(data[k][i]) for k in ("left", "top", "width", "height")]
        if w <= 0 or h <= 0:
            continue
        # Tesseract boxes can be word-level; fit a font to the box height.
        font_size = max(8, min(96, int(h * 0.82)))
        try:
            font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
        except OSError:
            font = ImageFont.load_default()
        # Render a word at approximate original location. OCR font metrics differ.
        draw.text((x, y), text, fill=(0, 0, 0), font=font, stroke_width=0)
        drawn += 1

    return np.array(canvas), {"drawn_words": drawn, "low_confidence_words": low_confidence}


def images_to_pdf(images, page_sizes=None):
    output = fitz.open()
    for idx, image in enumerate(images):
        h, w = image.shape[:2]
        if page_sizes and idx < len(page_sizes):
            page_w, page_h = page_sizes[idx]
        else:
            # Use pixel aspect ratio and a standard A4-like page.
            page_w, page_h = 595.0, 842.0
            ratio = w / max(h, 1)
            if ratio > page_w / page_h:
                page_h = page_w / ratio
        page = output.new_page(width=page_w, height=page_h)
        png = Image.fromarray(image).convert("RGB")
        buffer = io.BytesIO()
        png.save(buffer, format="PNG", optimize=True)
        page.insert_image(page.rect, stream=buffer.getvalue())
    data = output.tobytes(deflate=True, garbage=4)
    output.close()
    return data


def make_pdf_from_images(images, original_doc, original_page_indices):
    output = fitz.open()
    for image, page_idx in zip(images, original_page_indices):
        original_page = original_doc[page_idx]
        page = output.new_page(width=original_page.rect.width, height=original_page.rect.height)
        buffer = io.BytesIO()
        Image.fromarray(image).convert("RGB").save(buffer, format="PNG", optimize=True)
        page.insert_image(page.rect, stream=buffer.getvalue())
    result = output.tobytes(deflate=True, garbage=4)
    output.close()
    return result


def get_pdf_bytes(uploaded_file):
    return uploaded_file.getvalue()


# ---------- UI ----------

uploaded_files = st.file_uploader(
    "PDF 파일을 업로드하세요 (여러 파일 선택 가능)",
    type=["pdf"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("PDF를 업로드하면 파일별 페이지 범위와 복원 옵션을 설정할 수 있습니다.")
    st.markdown(
        "**권장 사용 순서**\n"
        "1. PDF 업로드 → 2. 처리할 파일 선택 → 3. 페이지 범위와 옵션 지정 → "
        "4. 미리보기 → 5. 결과 PDF 생성 및 다운로드"
    )
    st.stop()

file_names = [f.name for f in uploaded_files]
selected_name = st.selectbox("설정할 파일 선택", file_names)
selected_file = next(f for f in uploaded_files if f.name == selected_name)
pdf_bytes = get_pdf_bytes(selected_file)

try:
    source_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
except Exception as exc:
    st.error(f"PDF를 열 수 없습니다: {exc}")
    st.stop()

page_count = len(source_doc)
st.write(f"**{selected_name}** · 총 {page_count}페이지")

left, right = st.columns(2)
with left:
    start_page = st.number_input("시작 페이지", min_value=1, max_value=page_count, value=1, step=1)
with right:
    end_page = st.number_input("끝 페이지", min_value=1, max_value=page_count, value=page_count, step=1)

if end_page < start_page:
    st.error("끝 페이지는 시작 페이지보다 작을 수 없습니다.")
    st.stop()

st.subheader("출력 모드")
mode = st.radio(
    "처리 방식",
    ["이미지 복원", "OCR 텍스트 재구성 (실험적)"],
    horizontal=True,
    help="OCR 재구성은 인식된 텍스트를 흰 페이지에 다시 그립니다. 수식·그림·표의 정확한 재현을 보장하지 않습니다."
)

st.subheader("복원 기능")
c1, c2, c3 = st.columns(3)
with c1:
    deskew = st.checkbox("기울기 보정", value=True)
    perspective = st.checkbox("종이 원근/영역 보정", value=False)
with c2:
    background = st.checkbox("배경 백색화", value=True)
    denoise = st.checkbox("노이즈 제거", value=True)
with c3:
    sharpen = st.checkbox("글자 선명도 개선", value=True)
    contrast = st.checkbox("명암 개선", value=False)

background_strength = st.slider("배경 보정 강도", min_value=0.0, max_value=1.0, value=0.65, step=0.05)
dpi = st.select_slider("처리 해상도 (DPI)", options=[150, 200, 250, 300, 350], value=250)

if mode.startswith("OCR"):
    st.info(
        "실험적 OCR 모드: Tesseract가 인식한 텍스트만 재배치하므로 원본의 도형, 그래프, 수식 레이아웃이 "
        "사라지거나 인식 오류가 생길 수 있습니다. 결과를 반드시 확인하세요."
    )
    ocr_lang = st.selectbox("OCR 언어", ["kor+eng", "kor", "eng"], index=0)
    confidence_threshold = st.slider("최소 OCR 신뢰도", 0, 90, 35, 5)
else:
    ocr_lang = "kor+eng"
    confidence_threshold = 35

st.divider()
st.subheader("미리보기")
preview_page_num = st.number_input(
    "미리 볼 페이지", min_value=int(start_page), max_value=int(end_page),
    value=int(start_page), step=1
)

settings = {
    "deskew": deskew,
    "perspective": perspective,
    "background": background,
    "background_strength": background_strength,
    "denoise": denoise,
    "sharpen": sharpen,
    "contrast": contrast,
}

preview_col1, preview_col2 = st.columns(2)
try:
    preview_image = render_page(source_doc[int(preview_page_num) - 1], dpi=min(dpi, 200))
    preview_result = process_image(preview_image, settings)
    if mode.startswith("OCR"):
        preview_result, preview_stats = ocr_reconstruct(
            preview_result, lang=ocr_lang, threshold=confidence_threshold
        )
    with preview_col1:
        st.markdown("**원본**")
        st.image(preview_image, use_container_width=True)
    with preview_col2:
        st.markdown("**처리 결과 미리보기**")
        st.image(preview_result, use_container_width=True)
        if mode.startswith("OCR"):
            st.caption(
                f"인식 단어: {preview_stats['drawn_words']} · "
                f"저신뢰 단어 제외: {preview_stats['low_confidence_words']}"
            )
except Exception as exc:
    st.error(f"미리보기를 생성하지 못했습니다: {exc}")

st.divider()
st.subheader("PDF 생성")

if st.button("선택한 페이지 처리 및 PDF 생성", type="primary", use_container_width=True):
    selected_indices = list(range(int(start_page) - 1, int(end_page)))
    result_images = []
    result_page_sizes = []
    progress = st.progress(0)
    status = st.empty()
    failures = []

    for idx, page_idx in enumerate(selected_indices):
        status.write(f"페이지 처리 중: {page_idx + 1}/{end_page}")
        try:
            image = render_page(source_doc[page_idx], dpi=dpi)
            image = process_image(image, settings)
            if mode.startswith("OCR"):
                image, stats = ocr_reconstruct(
                    image, lang=ocr_lang, threshold=confidence_threshold
                )
            result_images.append(image)
            page = source_doc[page_idx]
            result_page_sizes.append((page.rect.width, page.rect.height))
        except Exception as exc:
            failures.append((page_idx + 1, str(exc)))
            # Fallback: keep a rendered original rather than silently dropping a page.
            result_images.append(render_page(source_doc[page_idx], dpi=dpi))
            page = source_doc[page_idx]
            result_page_sizes.append((page.rect.width, page.rect.height))
        progress.progress((idx + 1) / len(selected_indices))

    result_pdf = make_pdf_from_images(result_images, source_doc, selected_indices)
    status.success(f"완료: {len(result_images)}페이지 처리")
    if failures:
        st.warning("일부 페이지는 처리 중 오류가 발생하여 원본 렌더링으로 대체했습니다.")
        for page_no, message in failures:
            st.write(f"- {page_no}페이지: {message}")

    base_name = Path(selected_name).stem
    suffix = "restored" if mode == "이미지 복원" else "ocr_reconstructed_experimental"
    output_name = f"{base_name}_{start_page}-{end_page}_{suffix}.pdf"
    st.download_button(
        "📥 결과 PDF 다운로드",
        data=result_pdf,
        file_name=output_name,
        mime="application/pdf",
        use_container_width=True,
    )
    st.success("결과 PDF가 준비되었습니다. 저장한 뒤 원본과 페이지별로 비교해 주세요.")

source_doc.close()
