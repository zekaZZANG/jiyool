import streamlit as st
import cv2
import numpy as np
from PIL import Image
import fitz  # PyMuPDF
import os
import io
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# 페이지 설정
st.set_page_config(
    page_title="권지율의 스마트 모의고사 파일 수정 프로그램",
    page_icon="📚",
    layout="centered"
)

# [UI 디자인] 동글동글하고 산뜻한 하늘색 테마 커스텀 CSS 적용
st.markdown("""
    <style>
    .stButton>button {
        background-color: #0ea5e9;
        color: white;
        border-radius: 14px;
        font-weight: bold;
        height: 50px;
        width: 100%;
        border: none;
    }
    .stButton>button:hover {
        background-color: #0284c7;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

st.title("✨ 권지율의 스마트 모의고사 파일 수정 프로그램")
st.write("웹 브라우저에서 실행되어 **PC나 스마트폰 어디서든** 접속해 시험지 파일을 완벽하게 보정합니다!")

# 📱 모바일 브라우저에서 PDF 파일 비활성화 방지를 위해 type 제한 제거
uploaded_files = st.file_uploader(
    "📁 변환할 PDF 또는 이미지 파일들을 선택하거나 올려주세요", 
    accept_multiple_files=True
)

file_configs = []

if uploaded_files:
    st.markdown("---")
    st.markdown("### 📁 파일별 페이지 범위 설정")
    
    for idx, uploaded_file in enumerate(uploaded_files):
        file_bytes = uploaded_file.read()
        file_name = uploaded_file.name
        ext = os.path.splitext(file_name)[1].lower()

        if ext not in ['.pdf', '.jpg', '.jpeg', '.png', '.bmp']:
            st.warning(f"⚠️ '{file_name}'은(는) 지원하지 않는 형식입니다. (PDF, JPG, PNG만 가능)")
            continue

        max_pages = 1
        if ext == '.pdf':
            try:
                doc = fitz.open(stream=file_bytes, filetype="pdf")
                max_pages = len(doc)
                doc.close()
            except:
                max_pages = 1

        with st.container():
            st.write(f"📄 **{file_name}** (총 {max_pages}쪽)")
            col1, col2 = st.columns(2)
            with col1:
                start_page = st.number_input(
                    "시작 페이지", min_value=1, max_value=max_pages, value=1, key=f"start_{idx}_{file_name}"
                )
            with col2:
                end_page = st.number_input(
                    "끝 페이지", min_value=1, max_value=max_pages, value=max_pages, key=f"end_{idx}_{file_name}"
                )
            st.markdown("")

        file_configs.append({
            'bytes': file_bytes,
            'ext': ext,
            'max_pages': max_pages,
            'start': int(start_page),
            'end': int(end_page)
        })

# 옵션 체크박스 영역
st.markdown("---")
col_opt1, col_opt2 = st.columns(2)
with col_opt1:
    remove_bleed = st.checkbox("✨ 선명도 및 가독성 향상", value=True)
with col_opt2:
    straighten_top_line_opt = st.checkbox("📐 상단 가로선 기준 수평 맞춤", value=False)

def straighten_exam_by_top_line(img):
    """시험지 상단 영역의 가로선을 탐지하여 기울어진 각도를 계산하고 수평으로 맞춥니다."""
    try:
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        top_region = gray[0:int(h * 0.35), :]
        edges = cv2.Canny(top_region, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=int(w * 0.3), maxLineGap=20)
        
        angles = []
        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    if x2 - x1 != 0:
                        angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
                        if abs(angle) < 15:
                            angles.append(angle)
                            
        if angles:
            median_angle = np.median(angles)
            if abs(median_angle) > 0.2:
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    except Exception:
        pass
    return img

def crop_and_align_paper(img):
    """불필요한 바깥 여백을 자르고, 가로형 2페이지 스프레드일 때만 중앙 제본선 정렬 적용"""
    try:
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            if area > (h * w * 0.25):
                x, y, bw, bh = cv2.boundingRect(largest_contour)
                inset = 4
                x = max(0, x + inset)
                y = max(0, y + inset)
                bw = min(w - x, bw - (inset * 2))
                bh = min(h - y, bh - (inset * 2))
                
                if bw > w * 0.4 and bh > h * 0.4:
                    img = img[y:y+bh, x:x+bw]

        h2, w2 = img.shape[:2]
        if w2 > h2 * 1.1:
            gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mid_band = gray2[int(h2 * 0.2):int(h2 * 0.8), :]
            col_sums = np.sum(mid_band, axis=0)
            
            search_start = int(w2 * 0.4)
            search_end = int(w2 * 0.6)
            
            if search_start < search_end:
                trough_x = search_start + np.argmin(col_sums[search_start:search_end])
                target_x = w2 // 2
                shift = target_x - trough_x
                
                if abs(shift) > 5:
                    M = np.float32([[1, 0, shift], [0, 1, 0]])
                    img = cv2.warpAffine(img, M, (w2, h2), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    except Exception:
        pass
    return img

def safe_high_fidelity_enhance(img, enhance=True, straighten_top=False):
    """수식, 폰트, 기호를 100% 보존하면서 깔끔하게 렌더링하는 비파괴 고화질 보정 함수"""
    try:
        if straighten_top:
            img = straighten_exam_by_top_line(img)
            
        img = crop_and_align_paper(img)

        if enhance:
            # 글자를 뭉개는 강한 이진화 대신, 부드러운 CLAHE 명암비 조절과 양방향 필터로 가독성만 높임
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            smoothed = cv2.bilateralFilter(enhanced, d=3, sigmaColor=15, sigmaSpace=15)
            final_img = cv2.cvtColor(smoothed, cv2.COLOR_GRAY2BGR)
            return Image.fromarray(final_img)
        else:
            return Image.fromarray(img)
    except Exception:
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))

if file_configs:
    if st.button("🚀 수식 깨짐 없는 고화질 PDF 생성"):
        with st.spinner("수식과 폰트를 완벽하게 보존하며 PDF 생성 중입니다... 잠시만 기다려주세요!"):
            all_images = []
            
            for config in file_configs:
                file_bytes = config['bytes']
                ext = config['ext']
                max_pages = config['max_pages']
                s_val = config['start']
                e_val = config['end']

                s_idx = max(0, s_val - 1)
                e_idx = min(max_pages - 1, e_val - 1)
                
                if s_idx <= e_idx:
                    pages_to_process = list(range(s_idx, e_idx + 1))
                else:
                    pages_to_process = [s_idx]

                if ext == '.pdf':
                    doc = fitz.open(stream=file_bytes, filetype="pdf")
                    for page_index in pages_to_process:
                        page = doc[page_index]
                        # 300 DPI 초고해상도 렌더링 유지
                        pix = page.get_pixmap(dpi=300)
                        img_np = cv2.imdecode(np.frombuffer(pix.tobytes("png"), np.uint8), cv2.IMREAD_COLOR)
                        if img_np is not None:
                            all_images.append(img_np)
                    doc.close()
                elif ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    img_np = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
                    if img_np is not None:
                        all_images.append(img_np)

            if len(all_images) > 0:
                output_pdf_path = "권지율의_불꽃모의고사_완벽보존.pdf"
                a4_width, a4_height = A4
                c = canvas.Canvas(output_pdf_path, pagesize=A4)

                for img_np in all_images:
                    pil_img = safe_high_fidelity_enhance(img_np, remove_bleed, straighten_top_line_opt)
                    base_margin = 0
                    usable_width = a4_width - (base_margin * 2)
                    usable_height = a4_height - (base_margin * 2)
                    
                    img_w, img_h = pil_img.size
                    aspect = img_h / img_w
                    
                    draw_width = usable_width
                    draw_height = usable_width * aspect
                    
                    if draw_height > usable_height:
                        draw_height = usable_height
                        draw_width = usable_height / aspect

                    x = base_margin + (usable_width - draw_width) / 2
                    y = a4_height - base_margin - draw_height

                    c.drawImage(ImageReader(pil_img), x, y, width=draw_width, height=draw_height)
                    c.showPage()

                c.save()

                with open(output_pdf_path, "rb") as f:
                    pdf_data = f.read()

                st.success("✨ 수식 깨짐 없이 완벽하게 변환되었습니다!")
                
                st.download_button(
                    label="📥 완벽 보존된 모의고사 PDF 다운로드",
                    data=pdf_data,
                    file_name="권지율의_불꽃모의고사_완벽보존.pdf",
                    mime="application/pdf"
                )
            else:
                st.warning("처리할 수 있는 유효한 파일이나 페이지가 없습니다.")
