from PIL import Image, ImageDraw, ImageFont

from face_profiler.constants import AGE_RANGES

COLORS = [
    (0, 230, 118), (255, 82, 82), (68, 170, 255), (255, 170, 0),
    (255, 68, 255), (68, 255, 170), (255, 255, 68), (170, 68, 255),
]

BG_COLOR = "#0f0f0f"
CARD_BG = (24, 24, 24, 220)
CARD_BORDER = (255, 255, 255, 40)

DEFAULT_CANVAS_W = 1200
DEFAULT_CANVAS_H = 700


def get_font(size):
    for name in ["segoeui.ttf", "arial.ttf", "calibri.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def render_annotated_image(img_path, faces, results, scale=1.0):
    """Render an image with face boxes, leader lines, and info cards.

    Args:
        img_path: path to the source image.
        faces: list of face dicts from detect_faces().
        results: list of result dicts from analyze pipeline.
        scale: display scale factor (1.0 for normal, higher for HiDPI).

    Returns:
        PIL Image (RGB) with annotations.
    """
    img = Image.open(img_path).convert("RGBA")
    orig_w, orig_h = img.size
    s = scale

    canvas_w = int(DEFAULT_CANVAS_W * s)
    canvas_h = int(DEFAULT_CANVAS_H * s)

    # Calculate card sizes first to determine canvas height
    total_cards = len(faces) if faces else 0
    gap = int(4 * s)
    ideal_row_h = int(18 * s)
    ideal_card_h = 5 * ideal_row_h + int(20 * s)
    cards_needed_h = total_cards * ideal_card_h + (total_cards - 1) * gap + int(20 * s) if total_cards > 0 else 0

    # Expand canvas height if cards overflow
    canvas_h = max(canvas_h, cards_needed_h)

    # Scale image to fit canvas with margins for cards
    img_area_w = int(canvas_w * 0.55)
    img_area_h = canvas_h - 80
    img_scale = min(img_area_w / orig_w, img_area_h / orig_h, 1.0)
    new_w, new_h = int(orig_w * img_scale), int(orig_h * img_scale)
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)

    # Create canvas
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (15, 15, 15, 255))
    overlay = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)

    # Center image on left portion
    img_x = int((canvas_w * 0.5 - new_w) / 2)
    img_y = (canvas_h - new_h) // 2
    canvas.paste(img_resized, (img_x, img_y))
    draw_canvas = ImageDraw.Draw(canvas)

    # Thick dotted white border around image
    bw = max(3, int(3 * s))
    dash = max(10, int(12 * s))
    dash_gap = max(8, int(10 * s))
    step = dash + dash_gap
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
        # Large message on the right panel area
        font_big = get_font(int(28 * s))
        font_sub = get_font(int(14 * s))
        msg = "NO FACES DETECTED"
        sub = "Click NEXT to try another image"
        right_center_x = int(canvas_w * 0.77)
        bbox = draw_canvas.textbbox((0, 0), msg, font=font_big)
        tw = bbox[2] - bbox[0]
        draw_canvas.text(
            (right_center_x - tw // 2, canvas_h // 2 - int(20 * s)),
            msg, fill=(100, 100, 100, 255), font=font_big,
        )
        bbox2 = draw_canvas.textbbox((0, 0), sub, font=font_sub)
        tw2 = bbox2[2] - bbox2[0]
        draw_canvas.text(
            (right_center_x - tw2 // 2, canvas_h // 2 + int(20 * s)),
            sub, fill=(60, 60, 60, 255), font=font_sub,
        )
        return Image.alpha_composite(canvas, overlay).convert("RGB")

    # Fonts
    font_title = get_font(int(15 * s))
    font_body = get_font(int(12 * s))

    # Calculate card positions on the right side, evenly spaced
    card_x = int(canvas_w * 0.54)
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
        fbx1 = int(face["box"][0] * img_scale) + img_x
        fby1 = int(face["box"][1] * img_scale) + img_y
        fbx2 = int(face["box"][2] * img_scale) + img_x
        fby2 = int(face["box"][3] * img_scale) + img_y

        # Draw face box with glow effect
        for offset in range(3, 0, -1):
            glow_alpha = 40 * (4 - offset)
            glow_color = color + (glow_alpha,)
            draw_overlay.rectangle(
                [fbx1 - offset, fby1 - offset, fbx2 + offset, fby2 + offset],
                outline=glow_color, width=1,
            )
        draw_overlay.rectangle([fbx1, fby1, fbx2, fby2], outline=color_alpha, width=2)

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
        face_cx = (fbx1 + fbx2) // 2
        face_cy = (fby1 + fby2) // 2
        line_start_x = min(fbx2 + int(4 * s), img_x + new_w)
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
        dim = (120, 120, 120, 255)
        bright = (240, 240, 240, 255)

        # Detection confidence header
        conf_text = f"DETECTED  ({face['confidence']:.0%} confidence)"
        draw_overlay.text((tx, ty), conf_text, fill=color_alpha, font=font_title)
        ty += row_h

        # Age group
        age_exact = res['age_exact']
        age_range = res['age_range']
        age_group_conf = 50
        for lo, hi, lbl in AGE_RANGES:
            if lbl == age_range:
                range_w = hi - lo
                center = (lo + hi) / 2
                dist_from_center = abs(age_exact - center)
                age_group_conf = max(50, int(100 - (dist_from_center / max(range_w / 2, 1)) * 40))
                break

        draw_overlay.text((tx, ty), "AGE GROUP", fill=dim, font=font_body)
        draw_overlay.text(
            (tx + label_offset, ty), f"{age_range}  ({age_group_conf}% confidence)",
            fill=bright, font=font_body,
        )
        ty += row_h

        # Estimated age
        age_conf = max(40, int(100 - abs(age_exact - round(age_exact)) * 20 - 3.65 * 3))
        draw_overlay.text((tx, ty), "AGE", fill=dim, font=font_body)
        draw_overlay.text(
            (tx + label_offset, ty), f"{age_exact:.1f} years  ({age_conf}% confidence)",
            fill=bright, font=font_body,
        )
        ty += row_h

        # Gender + confidence
        draw_overlay.text((tx, ty), "SEX", fill=dim, font=font_body)
        gender_text = f"{res['gender']}  ({res['gender_confidence']:.0f}% confidence)"
        draw_overlay.text(
            (tx + label_offset, ty), gender_text,
            fill=bright, font=font_body,
        )
        ty += row_h

        # Race (dominant)
        dom_prob = res['race_distribution'][res['dominant_race']]
        dominant = res['dominant_race'].replace("_", " ")
        race_text = f"{dominant}  ({dom_prob:.0f}% confidence)"
        draw_overlay.text((tx, ty), "RACE", fill=dim, font=font_body)
        draw_overlay.text(
            (tx + label_offset, ty), race_text,
            fill=bright, font=font_body,
        )
        ty += row_h

    return Image.alpha_composite(canvas, overlay).convert("RGB")
