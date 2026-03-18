from face_profiler.constants import MIN_FACE_PX


def detect_faces(image):
    """Detect faces in an image.

    Args:
        image: file path (str) or numpy array (BGR).

    Returns:
        List of dicts with 'box' (x1, y1, x2, y2) and 'confidence'.
    """
    from retinaface import RetinaFace

    resp = RetinaFace.detect_faces(image)
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
    """Crop a face region from a numpy array with padding.

    Args:
        img: numpy array (H, W, C).
        box: tuple (x1, y1, x2, y2).
        expand: fraction to expand the crop by.

    Returns:
        Cropped numpy array.
    """
    h_img, w_img = img.shape[:2]
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    pad_w, pad_h = int(w * expand), int(h * expand)
    x1, y1 = max(0, x1 - pad_w), max(0, y1 - pad_h)
    x2, y2 = min(w_img, x2 + pad_w), min(h_img, y2 + pad_h)
    return img[y1:y2, x1:x2]
