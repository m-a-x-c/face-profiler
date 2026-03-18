import os
import sys
import random
import threading
import ctypes
import tkinter as tk
from pathlib import Path

from PIL import Image, ImageTk, ImageDraw

from face_profiler.constants import AGE_RANGES, FAIRFACE_RACE_LABELS
from face_profiler.detection import detect_faces, crop_face
from face_profiler.models import (
    load_mivolo, load_fairface, predict_age_gender, predict_race,
)
from face_profiler.rendering import (
    render_annotated_image, get_font, BG_COLOR,
)

DB_PATH = "images"

# Enable DPI awareness on Windows for sharp rendering on HiDPI displays
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


def _get_scale_factor():
    try:
        hdc = ctypes.windll.user32.GetDC(0)
        dpi = ctypes.windll.gdi32.GetDeviceCaps(hdc, 88)  # LOGPIXELSX
        ctypes.windll.user32.ReleaseDC(0, hdc)
        return dpi / 96.0
    except Exception:
        return 1.0


_SCALE = _get_scale_factor()
CANVAS_W = int(1200 * _SCALE)
CANVAS_H = int(700 * _SCALE)


class FaceAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Profiler")
        icon_path = str(Path(__file__).parent / "icon.ico")
        if os.path.exists(icon_path):
            self.root.iconbitmap(icon_path)
        s = _SCALE
        bar_h = int(80 * s)
        self.root.geometry(f"{CANVAS_W}x{CANVAS_H + bar_h * 2 + int(4 * s)}")
        self.root.configure(bg=BG_COLOR)
        self.root.resizable(True, True)
        self.root.minsize(int(800 * s), int(600 * s))
        self.root.state("zoomed")

        self.db_path = DB_PATH
        self.all_images = []
        self._refresh_image_list()

        self.models_loaded = False
        self.mivolo = None
        self.fairface = None

        # Scaled font sizes for widgets
        title_size = int(14 * s)
        status_size = int(10 * s)
        btn_size = int(12 * s)
        small_btn_size = int(10 * s)
        pad = int(20 * s)

        # Top bar
        top_frame = tk.Frame(root, bg="#181818", height=bar_h)
        top_frame.pack(fill=tk.X)
        top_frame.pack_propagate(False)

        tk.Label(
            top_frame, text="FACE PROFILER", font=("Segoe UI", title_size, "bold"),
            bg="#181818", fg="#ffffff", padx=pad,
        ).pack(side=tk.LEFT, pady=int(12 * s))

        self.status_var = tk.StringVar(value="Loading models...")
        tk.Label(
            top_frame, textvariable=self.status_var, font=("Segoe UI", status_size),
            bg="#181818", fg="#666666",
        ).pack(side=tk.RIGHT, padx=pad, pady=int(12 * s))

        # Folder label in top bar
        self.folder_var = tk.StringVar(value=f"Folder: {os.path.abspath(self.db_path)}")
        tk.Label(
            top_frame, textvariable=self.folder_var, font=("Segoe UI", status_size),
            bg="#181818", fg="#555555",
        ).pack(side=tk.RIGHT, padx=pad, pady=int(12 * s))

        # Accent line
        tk.Frame(root, bg="#0078d4", height=max(2, int(2 * s))).pack(fill=tk.X)

        # Bottom bar (pack first so it stays at bottom)
        bottom_frame = tk.Frame(root, bg="#181818", height=bar_h)
        bottom_frame.pack(fill=tk.X, side=tk.BOTTOM)
        bottom_frame.pack_propagate(False)

        # Scrollable canvas for annotated image
        self.scroll_frame = tk.Frame(root, bg=BG_COLOR)
        self.scroll_frame.pack(fill=tk.BOTH, expand=True)

        self.tk_canvas = tk.Canvas(self.scroll_frame, bg=BG_COLOR, highlightthickness=0)
        self.scrollbar = tk.Scrollbar(self.scroll_frame, orient=tk.VERTICAL, command=self.tk_canvas.yview)
        self.tk_canvas.configure(yscrollcommand=self.scrollbar.set)

        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tk_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas_label = tk.Label(self.tk_canvas, bg=BG_COLOR)
        self.canvas_window = self.tk_canvas.create_window((0, 0), window=self.canvas_label, anchor="n")

        # Bind mouse wheel for scrolling and resize for centering
        self.tk_canvas.bind_all("<MouseWheel>", lambda e: self.tk_canvas.yview_scroll(-1 * (e.delta // 120), "units"))
        self.canvas_label.bind("<Configure>", self._on_canvas_configure)
        self.tk_canvas.bind("<Configure>", self._on_tk_canvas_resize)

        # Bottom bar contents — left side
        self.filename_var = tk.StringVar()
        self.filename_label = tk.Label(
            bottom_frame, textvariable=self.filename_var, font=("Segoe UI", int(8 * s)),
            bg="#181818", fg="#444444", anchor="w",
        )
        self.filename_label.pack(side=tk.LEFT, padx=pad, pady=int(14 * s))

        # Bottom bar contents — right side (buttons right to left)
        self.analyze_btn = tk.Button(
            bottom_frame, text="LOADING...", font=("Segoe UI", btn_size, "bold"),
            bg="#0078d4", fg="white", relief=tk.FLAT,
            padx=int(28 * s), pady=int(8 * s),
            activebackground="#005a9e", activeforeground="white",
            cursor="hand2", command=self.analyze_random, state=tk.DISABLED,
            borderwidth=0,
        )
        self.analyze_btn.pack(side=tk.RIGHT, padx=(int(8 * s), pad), pady=int(10 * s))

        # Pick specific image button
        self.pick_btn = tk.Button(
            bottom_frame, text="PICK IMAGE", font=("Segoe UI", small_btn_size),
            bg="#444444", fg="white", relief=tk.FLAT,
            padx=int(16 * s), pady=int(6 * s),
            activebackground="#555555", activeforeground="white",
            cursor="hand2", command=self.pick_image, state=tk.DISABLED,
            borderwidth=0,
        )
        self.pick_btn.pack(side=tk.RIGHT, padx=int(4 * s), pady=int(10 * s))

        # Change folder button
        self.folder_btn = tk.Button(
            bottom_frame, text="CHANGE FOLDER", font=("Segoe UI", small_btn_size),
            bg="#444444", fg="white", relief=tk.FLAT,
            padx=int(16 * s), pady=int(6 * s),
            activebackground="#555555", activeforeground="white",
            cursor="hand2", command=self.change_folder,
            borderwidth=0,
        )
        self.folder_btn.pack(side=tk.RIGHT, padx=int(4 * s), pady=int(10 * s))

        # Show empty canvas
        self._show_placeholder()

        # Load models
        threading.Thread(target=self.load_models, daemon=True).start()

    def _show_placeholder(self):
        img = Image.new("RGB", (CANVAS_W, CANVAS_H), (15, 15, 15))
        draw = ImageDraw.Draw(img)
        font = get_font(int(36 * _SCALE))
        msg = "LOADING MODELS..."
        bbox = draw.textbbox((0, 0), msg, font=font)
        tw = bbox[2] - bbox[0]
        draw.text(((CANVAS_W - tw) // 2, CANVAS_H // 2 - int(20 * _SCALE)), msg, fill=(80, 80, 80), font=font)
        self.tk_image = ImageTk.PhotoImage(img)
        self.canvas_label.configure(image=self.tk_image)

    def _refresh_image_list(self):
        if os.path.isdir(self.db_path):
            self.all_images = [
                os.path.join(self.db_path, f)
                for f in os.listdir(self.db_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
            ]
        else:
            self.all_images = []

    def load_models(self):
        try:
            self.mivolo = load_mivolo(device=None)
            self.fairface = load_fairface(device=None)
            self.models_loaded = True
            self.root.after(0, self.status_var.set, "Ready")
            self.root.after(0, lambda: self.analyze_btn.configure(
                state=tk.NORMAL, text="RANDOM IMAGE",
            ))
            self.root.after(0, lambda: self.pick_btn.configure(state=tk.NORMAL))
            self.root.after(0, self._show_ready)
        except Exception as e:
            self.root.after(0, self.status_var.set, f"Error: {e}")
            print(f"Model loading error: {e}")

    def _show_ready(self):
        img = Image.new("RGB", (CANVAS_W, CANVAS_H), (15, 15, 15))
        draw = ImageDraw.Draw(img)
        font = get_font(int(36 * _SCALE))
        msg = "CLICK RANDOM IMAGE TO BEGIN"
        bbox = draw.textbbox((0, 0), msg, font=font)
        tw = bbox[2] - bbox[0]
        draw.text(((CANVAS_W - tw) // 2, CANVAS_H // 2 - int(20 * _SCALE)), msg, fill=(80, 80, 80), font=font)
        self.tk_image = ImageTk.PhotoImage(img)
        self.canvas_label.configure(image=self.tk_image)

    def _on_canvas_configure(self, event=None):
        self.tk_canvas.configure(scrollregion=self.tk_canvas.bbox("all"))

    def _on_tk_canvas_resize(self, event=None):
        canvas_w = self.tk_canvas.winfo_width()
        canvas_h = self.tk_canvas.winfo_height()
        x = canvas_w // 2
        if self.tk_image:
            img_h = self.tk_image.height()
            y = max(0, (canvas_h - img_h) // 2) if img_h <= canvas_h else 0
        else:
            y = 0
        self.tk_canvas.coords(self.canvas_window, x, y)
        self.tk_canvas.configure(scrollregion=self.tk_canvas.bbox("all"))

    def display_result(self, annotated_img):
        self.tk_image = ImageTk.PhotoImage(annotated_img)
        self.canvas_label.configure(image=self.tk_image)
        self.root.after(10, self._on_tk_canvas_resize)
        self.tk_canvas.yview_moveto(0)

    def change_folder(self):
        from tkinter import filedialog
        folder = filedialog.askdirectory(
            title="Select image folder",
            initialdir=self.db_path,
        )
        if folder:
            self.db_path = folder
            self._refresh_image_list()
            self.folder_var.set(f"Folder: {os.path.abspath(self.db_path)}")
            self.status_var.set(f"Folder changed — {len(self.all_images)} images found")

    def pick_image(self):
        from tkinter import filedialog
        filepath = filedialog.askopenfilename(
            title="Select an image",
            initialdir=self.db_path,
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("All files", "*.*"),
            ],
        )
        if filepath:
            self._analyze_image(filepath)

    def analyze_random(self):
        if not self.all_images:
            self.status_var.set("No images in folder")
            return
        img_path = random.choice(self.all_images)
        self._analyze_image(img_path)

    def _analyze_image(self, img_path):
        self.analyze_btn.configure(state=tk.DISABLED, text="ANALYZING...")
        self.pick_btn.configure(state=tk.DISABLED)
        self.status_var.set("Detecting faces...")
        name = os.path.basename(img_path)
        if len(name) > 30:
            name = name[:13] + "..." + name[-13:]
        self.filename_var.set(name)
        threading.Thread(target=self.run_analysis, args=(img_path,), daemon=True).start()

    def run_analysis(self, img_path):
        import cv2

        mivolo_model, mivolo_proc, mivolo_config, mivolo_device = self.mivolo
        ff_model, ff_transform, ff_device = self.fairface

        faces = detect_faces(img_path)

        if not faces:
            annotated = render_annotated_image(img_path, [], [], scale=_SCALE)
            self.root.after(0, self.display_result, annotated)
            self.root.after(0, self.status_var.set, "No faces detected")
            self.root.after(0, lambda: self.analyze_btn.configure(
                state=tk.NORMAL, text="RANDOM IMAGE",
            ))
            self.root.after(0, lambda: self.pick_btn.configure(state=tk.NORMAL))
            return

        img_cv = cv2.imread(img_path)
        all_results = []

        for i, face in enumerate(faces):
            self.root.after(0, self.status_var.set, f"Analyzing face {i+1}/{len(faces)}...")
            face_crop_cv = crop_face(img_cv, face["box"])
            face_crop_rgb = cv2.cvtColor(face_crop_cv, cv2.COLOR_BGR2RGB)
            face_crop_pil = Image.fromarray(face_crop_rgb)

            ag = predict_age_gender(mivolo_model, mivolo_proc, mivolo_config, mivolo_device, face_crop_rgb)
            race = predict_race(ff_model, ff_transform, ff_device, face_crop_pil)
            all_results.append({**ag, **race})

        annotated = render_annotated_image(img_path, faces, all_results, scale=_SCALE)
        self.root.after(0, self.display_result, annotated)

        count = len(all_results)
        self.root.after(0, self.status_var.set,
                        f"{count} person{'s' if count != 1 else ''} detected")
        self.root.after(0, lambda: self.analyze_btn.configure(
            state=tk.NORMAL, text="RANDOM IMAGE",
        ))
        self.root.after(0, lambda: self.pick_btn.configure(state=tk.NORMAL))


def main():
    root = tk.Tk()
    app = FaceAnalyzerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
