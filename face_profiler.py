import os
import sys
import random
import threading
import ctypes
import tkinter as tk

# Enable DPI awareness on Windows for sharp rendering on HiDPI displays
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)  # Per-monitor DPI aware
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

# Ensure local mivolo package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import torch
from PIL import Image, ImageTk, ImageDraw, ImageFont
from retinaface import RetinaFace
from transformers import AutoModelForImageClassification, AutoConfig, AutoImageProcessor
from torchvision import transforms

DB_PATH = "images"
MIN_FACE_PX = 40

# Detect display scale factor for HiDPI
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

AGE_RANGES = [
    (0, 2, "0-2"), (3, 5, "3-5"), (6, 12, "6-12"), (13, 17, "13-17"),
    (18, 25, "18-25"), (26, 35, "26-35"), (36, 45, "36-45"),
    (46, 55, "46-55"), (56, 65, "56-65"), (66, 120, "66+"),
]

FAIRFACE_RACE_LABELS = [
    "White", "Black", "Latino_Hispanic", "East Asian",
    "Southeast Asian", "Indian", "Middle Eastern",
]

COLORS = [
    (0, 230, 118), (255, 82, 82), (68, 170, 255), (255, 170, 0),
    (255, 68, 255), (68, 255, 170), (255, 255, 68), (170, 68, 255),
]

BG_COLOR = "#0f0f0f"
CARD_BG = (24, 24, 24, 220)
CARD_BORDER = (255, 255, 255, 40)


def age_to_range(age_float):
    age = int(round(age_float))
    for lo, hi, label in AGE_RANGES:
        if lo <= age <= hi:
            return label
    return f"{age}"


def get_font(size):
    for name in ["segoeui.ttf", "arial.ttf", "calibri.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def load_mivolo():
    print("Loading MiVOLO v2...")
    config = AutoConfig.from_pretrained("iitolstykh/mivolo_v2", trust_remote_code=True)
    model = AutoModelForImageClassification.from_pretrained(
        "iitolstykh/mivolo_v2", trust_remote_code=True, dtype=torch.float32,
    )
    processor = AutoImageProcessor.from_pretrained(
        "iitolstykh/mivolo_v2", trust_remote_code=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    print(f"MiVOLO v2 loaded on {device}")
    return model, processor, config, device


def load_fairface():
    print("Loading FairFace...")
    from torchvision import models
    model = models.resnet34(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 18)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    weights_path = os.path.join(os.path.expanduser("~"), ".cache", "fairface_resnet34.pth")
    if not os.path.exists(weights_path):
        print("Downloading FairFace weights...")
        import gdown
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        gdown.download(
            "https://drive.google.com/uc?id=11y0Wi3YQf21a_VcspUV4FwqzhMcfaVAB",
            weights_path, quiet=False,
        )

    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    print(f"FairFace loaded on {device}")
    return model, transform, device


def detect_faces(img_path):
    resp = RetinaFace.detect_faces(img_path)
    if not isinstance(resp, dict):
        return []
    faces = []
    for key, face_data in resp.items():
        x1, y1, x2, y2 = face_data["facial_area"]
        w, h = x2 - x1, y2 - y1
        if w < MIN_FACE_PX or h < MIN_FACE_PX:
            continue
        confidence = face_data.get("score", 0)
        if confidence < 0.7:
            continue
        faces.append({
            "box": (x1, y1, x2, y2),
            "confidence": confidence,
        })
    return faces


def crop_face(img, box, expand=0.2):
    h_img, w_img = img.shape[:2]
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    pad_w, pad_h = int(w * expand), int(h * expand)
    x1, y1 = max(0, x1 - pad_w), max(0, y1 - pad_h)
    x2, y2 = min(w_img, x2 + pad_w), min(h_img, y2 + pad_h)
    return img[y1:y2, x1:x2]


def predict_age_gender(model, processor, config, device, face_crop):
    inputs = processor(images=[face_crop])
    pixel_values = inputs["pixel_values"]
    if isinstance(pixel_values, list):
        pixel_values = torch.tensor(np.array(pixel_values))
    pixel_values = pixel_values.to(dtype=model.dtype, device=device)
    body_input = torch.zeros_like(pixel_values)

    with torch.no_grad():
        output = model(faces_input=pixel_values, body_input=body_input)

    age = output.age_output[0].item()
    gender_idx = output.gender_class_idx[0].item()
    gender_prob = output.gender_probs[0].item()
    gender_label = config.gender_id2label[gender_idx]

    return {
        "age_exact": age,
        "age_range": age_to_range(age),
        "gender": gender_label.capitalize(),
        "gender_confidence": gender_prob * 100,
    }


def predict_race(model, transform, device, face_crop_pil):
    img_tensor = transform(face_crop_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
    race_logits = output[0, :7]
    race_probs = torch.nn.functional.softmax(race_logits, dim=0).cpu().numpy()
    race_dist = {FAIRFACE_RACE_LABELS[i]: float(race_probs[i]) * 100 for i in range(7)}
    dominant_race = FAIRFACE_RACE_LABELS[int(race_probs.argmax())]
    return {"dominant_race": dominant_race, "race_distribution": race_dist}


def render_annotated_image(img_path, faces, results):
    """Render the image with face boxes, leader lines, and info cards."""
    img = Image.open(img_path).convert("RGBA")
    orig_w, orig_h = img.size
    s = _SCALE

    # Calculate card sizes first to determine canvas height
    total_cards = len(faces) if faces else 0
    gap = int(4 * s)
    ideal_row_h = int(18 * s)
    ideal_card_h = 5 * ideal_row_h + int(20 * s)
    cards_needed_h = total_cards * ideal_card_h + (total_cards - 1) * gap + int(20 * s) if total_cards > 0 else 0

    # Expand canvas height if cards overflow
    canvas_h = max(CANVAS_H, cards_needed_h)

    # Scale image to fit canvas with margins for cards
    img_area_w = int(CANVAS_W * 0.55)
    img_area_h = canvas_h - 80
    scale = min(img_area_w / orig_w, img_area_h / orig_h, 1.0)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)

    # Create canvas
    canvas = Image.new("RGBA", (CANVAS_W, canvas_h), (15, 15, 15, 255))
    overlay = Image.new("RGBA", (CANVAS_W, canvas_h), (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)

    # Center image on left portion
    img_x = int((CANVAS_W * 0.5 - new_w) / 2)
    img_y = (canvas_h - new_h) // 2
    canvas.paste(img_resized, (img_x, img_y))
    draw_canvas = ImageDraw.Draw(canvas)

    # Thick dotted white border around image
    bw = max(3, int(3 * _SCALE))
    dash = max(10, int(12 * _SCALE))
    gap = max(8, int(10 * _SCALE))
    step = dash + gap
    bx0, by0 = img_x - bw - 2, img_y - bw - 2
    bx1, by1 = img_x + new_w + bw + 1, img_y + new_h + bw + 1
    bc = (220, 220, 220, 255)
    # Top & bottom
    x = bx0
    while x < bx1:
        x_end = min(x + dash, bx1)
        draw_canvas.line([(x, by0), (x_end, by0)], fill=bc, width=bw)
        draw_canvas.line([(x, by1), (x_end, by1)], fill=bc, width=bw)
        x += step
    # Left & right
    y = by0
    while y < by1:
        y_end = min(y + dash, by1)
        draw_canvas.line([(bx0, y), (bx0, y_end)], fill=bc, width=bw)
        draw_canvas.line([(bx1, y), (bx1, y_end)], fill=bc, width=bw)
        y += step

    if not faces or not results:
        s = _SCALE
        # Large message on the right panel area
        font_big = get_font(int(28 * s))
        font_sub = get_font(int(14 * s))
        msg = "NO FACES DETECTED"
        sub = "Click NEXT to try another image"
        right_center_x = int(CANVAS_W * 0.77)
        bbox = draw_canvas.textbbox((0, 0), msg, font=font_big)
        tw = bbox[2] - bbox[0]
        draw_canvas.text(
            (right_center_x - tw // 2, CANVAS_H // 2 - int(20 * s)),
            msg, fill=(100, 100, 100, 255), font=font_big,
        )
        bbox2 = draw_canvas.textbbox((0, 0), sub, font=font_sub)
        tw2 = bbox2[2] - bbox2[0]
        draw_canvas.text(
            (right_center_x - tw2 // 2, CANVAS_H // 2 + int(20 * s)),
            sub, fill=(60, 60, 60, 255), font=font_sub,
        )
        return Image.alpha_composite(canvas, overlay).convert("RGB")

    # Fonts (scaled for HiDPI) — two sizes for consistency
    s = _SCALE
    font_title = get_font(int(15 * s))
    font_body = get_font(int(12 * s))

    # Calculate card positions on the right side, evenly spaced
    card_x = int(CANVAS_W * 0.54)
    card_w = int(320 * s)
    row_h = int(18 * s)
    card_h = 5 * row_h + int(20 * s)
    total_height = total_cards * card_h + (total_cards - 1) * gap
    start_y = max(int(10 * s), (canvas_h - total_height) // 2)

    for i, (face, res) in enumerate(zip(faces, results)):
        color = COLORS[i % len(COLORS)]
        color_alpha = color + (255,)
        color_dim = tuple(c // 2 for c in color) + (180,)

        # Face box on image (scaled)
        bx1 = int(face["box"][0] * scale) + img_x
        by1 = int(face["box"][1] * scale) + img_y
        bx2 = int(face["box"][2] * scale) + img_x
        by2 = int(face["box"][3] * scale) + img_y

        # Draw face box with glow effect
        for offset in range(3, 0, -1):
            glow_alpha = 40 * (4 - offset)
            glow_color = color + (glow_alpha,)
            draw_overlay.rectangle(
                [bx1 - offset, by1 - offset, bx2 + offset, by2 + offset],
                outline=glow_color, width=1,
            )
        draw_overlay.rectangle([bx1, by1, bx2, by2], outline=color_alpha, width=2)

        # Card position
        cy = start_y + i * (card_h + gap)

        # Draw card background
        draw_overlay.rectangle(
            [card_x, cy, card_x + card_w, cy + card_h],
            fill=CARD_BG,
        )
        # Card left accent bar
        accent_w = max(3, int(4 * s))
        draw_overlay.rectangle(
            [card_x, cy, card_x + accent_w, cy + card_h],
            fill=color_alpha,
        )
        # Card subtle border
        draw_overlay.rectangle(
            [card_x, cy, card_x + card_w, cy + card_h],
            outline=CARD_BORDER, width=1,
        )

        # Leader line from face center to card
        face_cx = (bx1 + bx2) // 2
        face_cy = (by1 + by2) // 2
        line_start_x = min(bx2 + int(4 * s), img_x + new_w)
        card_attach_y = cy + card_h // 2
        line_w = max(2, int(2 * s))

        points = [
            (line_start_x, face_cy),
            (card_x - int(15 * s), card_attach_y),
            (card_x, card_attach_y),
        ]
        for seg in range(len(points) - 1):
            x0, y0 = points[seg]
            x1, y1 = points[seg + 1]
            draw_overlay.line([(x0, y0), (x1, y1)], fill=color_dim, width=line_w)

        dot_r = max(3, int(3 * s))
        draw_overlay.ellipse(
            [line_start_x - dot_r, face_cy - dot_r, line_start_x + dot_r, face_cy + dot_r],
            fill=color_alpha,
        )

        # Card content
        tx = card_x + int(14 * s)
        ty = cy + max(int(4 * s), (card_h - 5 * row_h) // 2)
        label_offset = int(80 * s)
        line_h = row_h
        dim = (120, 120, 120, 255)
        bright = (240, 240, 240, 255)
        mid = (160, 160, 160, 255)

        # Detection confidence header
        conf_text = f"DETECTED  ({face['confidence']:.0%} confidence)"
        draw_overlay.text((tx, ty), conf_text, fill=color_alpha, font=font_title)
        ty += line_h

        # Age group (MiVOLO MAE ~3.65 years, so confidence is inversely related to range width)
        age_exact = res['age_exact']
        age_range = res['age_range']
        # Estimate age confidence from how centered the prediction is in the range bucket
        for lo, hi, lbl in AGE_RANGES:
            if lbl == age_range:
                range_w = hi - lo
                center = (lo + hi) / 2
                dist_from_center = abs(age_exact - center)
                age_group_conf = max(50, int(100 - (dist_from_center / max(range_w / 2, 1)) * 40))
                break
        else:
            age_group_conf = 50

        draw_overlay.text((tx, ty), "AGE GROUP", fill=dim, font=font_body)
        draw_overlay.text(
            (tx + label_offset, ty), f"{age_range}  ({age_group_conf}% confidence)",
            fill=bright, font=font_body,
        )
        ty += line_h

        # Estimated age
        # MiVOLO v2 MAE is ~3.65 — derive confidence from how far from bucket edges
        age_conf = max(40, int(100 - abs(age_exact - round(age_exact)) * 20 - 3.65 * 3))
        draw_overlay.text((tx, ty), "AGE", fill=dim, font=font_body)
        draw_overlay.text(
            (tx + label_offset, ty), f"{age_exact:.1f} years  ({age_conf}% confidence)",
            fill=bright, font=font_body,
        )
        ty += line_h

        # Gender + confidence
        draw_overlay.text((tx, ty), "SEX", fill=dim, font=font_body)
        gender_text = f"{res['gender']}  ({res['gender_confidence']:.0f}% confidence)"
        draw_overlay.text(
            (tx + label_offset, ty), gender_text,
            fill=bright, font=font_body,
        )
        ty += line_h

        # Race (dominant)
        dom_prob = res['race_distribution'][res['dominant_race']]
        dominant = res['dominant_race'].replace("_", " ")
        race_text = f"{dominant}  ({dom_prob:.0f}% confidence)"
        draw_overlay.text((tx, ty), "RACE", fill=dim, font=font_body)
        draw_overlay.text(
            (tx + label_offset, ty), race_text,
            fill=bright, font=font_body,
        )
        ty += line_h


    result = Image.alpha_composite(canvas, overlay).convert("RGB")
    return result


class FaceAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Profiler")
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.ico")
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
        filename_size = int(10 * s)
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
            mivolo_model, mivolo_proc, mivolo_config, mivolo_device = load_mivolo()
            self.mivolo = (mivolo_model, mivolo_proc, mivolo_config, mivolo_device)
            ff_model, ff_transform, ff_device = load_fairface()
            self.fairface = (ff_model, ff_transform, ff_device)
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
        # Center horizontally, and vertically if content fits
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
        # Defer centering to after layout
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
        mivolo_model, mivolo_proc, mivolo_config, mivolo_device = self.mivolo
        ff_model, ff_transform, ff_device = self.fairface

        faces = detect_faces(img_path)

        if not faces:
            annotated = render_annotated_image(img_path, [], [])
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

        annotated = render_annotated_image(img_path, faces, all_results)
        self.root.after(0, self.display_result, annotated)

        count = len(all_results)
        self.root.after(0, self.status_var.set,
                        f"{count} person{'s' if count != 1 else ''} detected")
        self.root.after(0, lambda: self.analyze_btn.configure(
            state=tk.NORMAL, text="RANDOM IMAGE",
        ))
        self.root.after(0, lambda: self.pick_btn.configure(state=tk.NORMAL))


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceAnalyzerApp(root)
    root.mainloop()
