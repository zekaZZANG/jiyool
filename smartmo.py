import os
import tempfile
import threading
import concurrent.futures
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
import fitz  # PyMuPDF
from PIL import Image
import cv2  # OpenCV (스마트 크롭, 조건부 중앙 정렬, 비침 해결)
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# 모던 UI 테마 설정
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class ExamPdfApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("권지율의 스마트 모의고사 파일 수정 프로그램")
        self.geometry("600x720")
        self.resizable(False, False)

        self.file_items = []  # 파일 정보 및 위젯들을 관리할 리스트

        # --- [UI 디자인] 메인 컨테이너 프레임 ---
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.pack(fill="both", expand=True, padx=25, pady=20)

        # 상단 타이틀 영역
        self.title_label = ctk.CTkLabel(
            self.main_frame, 
            text="✨ 권지율의 스마트 모의고사 파일 수정 프로그램", 
            font=ctk.CTkFont(family="맑은 고딕", size=18, weight="bold")
        )
        self.title_label.pack(anchor="w", pady=(0, 2))

        self.desc_label = ctk.CTkLabel(
            self.main_frame, 
            text="표지와 내지 형태를 스마트하게 구분하여 쏠림 없이 완벽하게 정렬합니다.", 
            font=ctk.CTkFont(family="맑은 고딕", size=11),
            text_color="gray"
        )
        self.desc_label.pack(anchor="w", pady=(0, 12))

        # --- [UI 디자인] 파일 관리 카드 섹션 (동글동글한 모서리) ---
        self.file_card = ctk.CTkFrame(self.main_frame, corner_radius=18, fg_color=("gray92", "gray17"))
        self.file_card.pack(fill="x", pady=(0, 12))

        self.file_header_frame = ctk.CTkFrame(self.file_card, fg_color="transparent")
        self.file_header_frame.pack(fill="x", padx=15, pady=(12, 8))

        self.file_section_label = ctk.CTkLabel(
            self.file_header_frame, 
            text="📁 변환할 파일 및 페이지 범위 설정", 
            font=ctk.CTkFont(family="맑은 고딕", size=13, weight="bold")
        )
        self.file_section_label.pack(side="left")

        self.select_btn = ctk.CTkButton(
            self.file_header_frame, 
            text="+ 파일 추가", 
            command=self.select_files,
            font=ctk.CTkFont(family="맑은 고딕", size=12, weight="bold"),
            width=95, height=32,
            corner_radius=14,
            fg_color="#0284c7", hover_color="#0369a1"  # 하늘색 포인트
        )
        self.select_btn.pack(side="right")

        # 파일 목록이 들어갈 스크롤 프레임
        self.file_scroll_frame = ctk.CTkScrollableFrame(
            self.file_card, 
            width=530, 
            height=200, 
            corner_radius=14,
            fg_color=("white", "gray22")
        )
        self.file_scroll_frame.pack(padx=15, pady=(0, 15), fill="both", expand=True)

        # 초기 안내 텍스트 레이블
        self.empty_label = ctk.CTkLabel(
            self.file_scroll_frame, 
            text="추가된 파일이 없습니다. [+ 파일 추가] 버튼을 눌러주세요.", 
            font=ctk.CTkFont(family="맑은 고딕", size=11),
            text_color="gray"
        )
        self.empty_label.pack(pady=40)

        # --- [UI 디자인] 옵션 설정 카드 섹션 ---
        self.option_card = ctk.CTkFrame(self.main_frame, corner_radius=18, fg_color=("gray92", "gray17"))
        self.option_card.pack(fill="x", pady=(0, 12))

        self.option_inner = ctk.CTkFrame(self.option_card, fg_color="transparent")
        self.option_inner.pack(fill="x", padx=15, pady=14)

        self.bleed_var = ctk.BooleanVar(value=True)  # 기본 체크 상태
        self.bleed_checkbox = ctk.CTkCheckBox(
            self.option_inner, 
            text="✨ 뒷면 비침 해결 (뒷장 글씨를 하얗게 지우기)", 
            variable=self.bleed_var,
            font=ctk.CTkFont(family="맑은 고딕", size=12, weight="bold"),
            checkbox_width=22, checkbox_height=22,
            corner_radius=7,
            fg_color="#0284c7", hover_color="#0369a1",
            border_color="#0284c7"
        )
        self.bleed_checkbox.pack(anchor="w", pady=2)

        # --- [UI 디자인] 하단 진행 상황 및 실행 버튼 섹션 ---
        self.progress_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.progress_frame.pack(fill="x", pady=(0, 5))

        self.status_label = ctk.CTkLabel(
            self.progress_frame, 
            text="대기 중...", 
            font=ctk.CTkFont(family="맑은 고딕", size=11),
            text_color="gray"
        )
        self.status_label.pack(anchor="w", pady=(0, 4))

        self.progress_bar = ctk.CTkProgressBar(self.progress_frame, height=14, corner_radius=7, progress_color="#0284c7")
        self.progress_bar.pack(fill="x")
        self.progress_bar.set(0.0)

        self.convert_btn = ctk.CTkButton(
            self.main_frame, 
            text="🚀 초고속 병렬 PDF 생성하기", 
            command=self.start_conversion_thread,
            font=ctk.CTkFont(family="맑은 고딕", size=14, weight="bold"),
            fg_color="#0ea5e9", hover_color="#0284c7",  # 상큼한 하늘색
            height=48,
            corner_radius=16
        )
        self.convert_btn.pack(fill="x", pady=(5, 0))

    def select_files(self):
        file_paths = filedialog.askopenfilenames(
            title="변환할 PDF 또는 이미지 파일들을 선택하세요",
            filetypes=[
                ("모든 지원 파일", "*.pdf *.jpg *.jpeg *.png *.bmp"),
                ("PDF 파일", "*.pdf"),
                ("이미지 파일", "*.jpg *.jpeg *.png *.bmp")
            ]
        )
        
        if file_paths:
            if self.empty_label.winfo_exists():
                self.empty_label.pack_forget()

            for path in file_paths:
                if any(item['path'] == path for item in self.file_items):
                    continue
                
                ext = os.path.splitext(path)[1].lower()
                max_pages = 1
                if ext == '.pdf':
                    try:
                        doc = fitz.open(path)
                        max_pages = len(doc)
                        doc.close()
                    except:
                        max_pages = 1

                row_frame = ctk.CTkFrame(self.file_scroll_frame, fg_color=("gray95", "gray25"), corner_radius=10)
                row_frame.pack(fill="x", padx=5, pady=5)

                filename = os.path.basename(path)
                display_name = filename if len(filename) <= 20 else filename[:17] + "..."
                info_text = f"{display_name} (총 {max_pages}쪽)"
                
                name_label = ctk.CTkLabel(
                    row_frame, 
                    text=info_text, 
                    font=ctk.CTkFont(family="맑은 고딕", size=11, weight="bold"),
                    width=210,
                    anchor="w"
                )
                name_label.pack(side="left", padx=(10, 5), pady=8)

                del_btn = ctk.CTkButton(
                    row_frame, 
                    text="❌ 취소", 
                    command=lambda r=row_frame, p=path: self.remove_file(r, p),
                    font=ctk.CTkFont(family="맑은 고딕", size=10, weight="bold"),
                    width=55, height=28,
                    corner_radius=8,
                    fg_color="#f43f5e", hover_color="#e11d48"
                )
                del_btn.pack(side="right", padx=(5, 10))

                end_entry = ctk.CTkEntry(row_frame, width=45, height=28, corner_radius=6, font=ctk.CTkFont(family="맑은 고딕", size=11))
                end_entry.pack(side="right", padx=2)
                end_entry.insert(0, str(max_pages))

                tilde_label = ctk.CTkLabel(row_frame, text="~", font=ctk.CTkFont(family="맑은 고딕", size=11, weight="bold"))
                tilde_label.pack(side="right", padx=2)

                start_entry = ctk.CTkEntry(row_frame, width=45, height=28, corner_radius=6, font=ctk.CTkFont(family="맑은 고딕", size=11))
                start_entry.pack(side="right", padx=2)
                start_entry.insert(0, "1")

                range_label = ctk.CTkLabel(
                    row_frame, 
                    text="범위:", 
                    font=ctk.CTkFont(family="맑은 고딕", size=11),
                    text_color="gray"
                )
                range_label.pack(side="right", padx=(5, 2))

                self.file_items.append({
                    'path': path,
                    'max_pages': max_pages,
                    'row_frame': row_frame,
                    'start_entry': start_entry,
                    'end_entry': end_entry
                })

    def remove_file(self, row_frame, path):
        row_frame.destroy()
        self.file_items = [item for item in self.file_items if item['path'] != path]
        
        if not self.file_items:
            self.empty_label.pack(pady=40)

    def crop_and_align_paper(self, img):
        """불필요한 바깥 여백을 정밀 자르고, 가로형 2페이지 스프레드일 때만 중앙 제본선 정렬 적용"""
        try:
            h, w = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 1단계: 바깥쪽 불필요한 여백 및 그림자 크롭
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                if area > (h * w * 0.25):
                    x, y, bw, bh = cv2.boundingRect(largest_contour)
                    inset = 8
                    x = max(0, x + inset)
                    y = max(0, y + inset)
                    bw = min(w - x, bw - (inset * 2))
                    bh = min(h - y, bh - (inset * 2))
                    
                    if bw > w * 0.4 and bh > h * 0.4:
                        img = img[y:y+bh, x:x+bw]

            # 2단계 [스마트 판단]: 표지 같은 세로형 단일 페이지(w < h)는 쏠림 방지를 위해 제본선 정렬 생략!
            # 책을 펼친 가로형 2페이지 스프레드(w > h * 1.1)일 때만 정확히 정중앙으로 대칭 정렬 수행
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

    def smart_enhance_image_np(self, img, remove_bleed=True):
        """스마트 크롭 및 정렬 + 비침 해결(흰색 덮기) + 고화질 잉크 보정"""
        try:
            img = self.crop_and_align_paper(img)

            h, w = img.shape[:2]
            gray_crop = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if remove_bleed:
                kernel_bleed = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
                bg_bleed = cv2.morphologyEx(gray_crop, cv2.MORPH_CLOSE, kernel_bleed)
                diff = cv2.absdiff(bg_bleed, gray_crop)
                gray_crop[diff < 18] = 255

            kernel_bg = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
            bg = cv2.morphologyEx(gray_crop, cv2.MORPH_CLOSE, kernel_bg)
            normalized = cv2.divide(gray_crop, bg, scale=255)

            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            enhanced = clahe.apply(normalized)

            gamma = 0.70
            invGamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(256)]).astype("uint8")
            gamma_corrected = cv2.LUT(enhanced, table)

            final_gray = cv2.bilateralFilter(gamma_corrected, d=5, sigmaColor=30, sigmaSpace=30)
            final_img = cv2.cvtColor(final_gray, cv2.COLOR_GRAY2BGR)
            
            return Image.fromarray(final_img)
        except Exception:
            return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    def start_conversion_thread(self):
        if not self.file_items:
            messagebox.showwarning("경고", "변환할 파일을 먼저 추가해 주세요!")
            return

        save_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")],
            initialfile="권지율의_모의고사_완성.pdf"
        )
        
        if not save_path:
            return

        remove_bleed = self.bleed_var.get()
        threading.Thread(target=self.process_pdf_parallel, args=(save_path, remove_bleed), daemon=True).start()

    def process_pdf_parallel(self, save_path, remove_bleed):
        try:
            self.after(0, lambda: self.status_label.configure(text="페이지 추출 및 병렬 처리 준비 중..."))
            self.progress_bar.set(0.0)

            tasks = []
            task_id = 0

            for item in self.file_items:
                path = item['path']
                max_pages = item['max_pages']
                try:
                    s_val = int(item['start_entry'].get().strip())
                    e_val = int(item['end_entry'].get().strip())
                except:
                    s_val = 1
                    e_val = max_pages

                s_idx = max(0, s_val - 1)
                e_idx = min(max_pages - 1, e_val - 1)
                
                if s_idx <= e_idx:
                    pages_to_process = list(range(s_idx, e_idx + 1))
                else:
                    pages_to_process = [s_idx]

                ext = os.path.splitext(path)[1].lower()
                if ext == '.pdf':
                    doc = fitz.open(path)
                    for page_index in pages_to_process:
                        page = doc[page_index]
                        pix = page.get_pixmap(dpi=150)
                        img_np = cv2.imdecode(np.frombuffer(pix.tobytes("png"), np.uint8), cv2.IMREAD_COLOR)
                        tasks.append((task_id, img_np))
                        task_id += 1
                    doc.close()
                elif ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    img_np = cv2.imread(path)
                    if img_np is not None:
                        tasks.append((task_id, img_np))
                        task_id += 1

            total_tasks = len(tasks)
            if total_tasks == 0:
                self.after(0, lambda: messagebox.showwarning("경고", "유효한 페이지 범위가 없습니다."))
                return

            results = {}
            completed_count = 0

            self.after(0, lambda: self.status_label.configure(text=f"총 {total_tasks}개 페이지 스마트 정렬 및 비침 제거 중..."))

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_id = {executor.submit(self.smart_enhance_image_np, img_np, remove_bleed): tid for tid, img_np in tasks}
                
                for future in concurrent.futures.as_completed(future_to_id):
                    tid = future_to_id[future]
                    try:
                        pil_img = future.result()
                        results[tid] = pil_img
                    except Exception:
                        results[tid] = None
                    
                    completed_count += 1
                    progress_val = completed_count / total_tasks
                    self.after(0, lambda v=progress_val, c_num=completed_count, t_num=total_tasks: [
                        self.progress_bar.set(v),
                        self.status_label.configure(text=f"고속 처리 중... ({c_num}/{t_num} 페이지 완료)")
                    ])

            self.after(0, lambda: self.status_label.configure(text="최종 PDF 파일 병합 중..."))
            
            a4_width, a4_height = A4
            c = canvas.Canvas(save_path, pagesize=A4)

            for tid in range(total_tasks):
                pil_img = results.get(tid)
                if pil_img is not None:
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
            self.after(0, lambda: [
                self.progress_bar.set(1.0),
                self.status_label.configure(text="변환 완료!"),
                messagebox.showinfo("성공", "표지와 내지 모두 쏠림 없이 완벽하게 변환되었습니다!")
            ])
        except Exception as e:
            err_msg = str(e)
            self.after(0, lambda: messagebox.showerror("오류", f"오류가 발생했습니다:\n{err_msg}"))

if __name__ == "__main__":
    app = ExamPdfApp()
    app.mainloop()