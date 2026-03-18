import os

from face_profiler.constants import FAIRFACE_RACE_LABELS, age_to_range


def _auto_device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_mivolo(device=None):
    """Load MiVOLO v2 model for age/gender estimation.

    Returns:
        Tuple of (model, processor, config, device_str).
    """
    import torch
    from transformers import AutoModelForImageClassification, AutoConfig, AutoImageProcessor

    if device is None:
        device = _auto_device()

    print("Loading MiVOLO v2...")
    config = AutoConfig.from_pretrained("iitolstykh/mivolo_v2", trust_remote_code=True)
    model = AutoModelForImageClassification.from_pretrained(
        "iitolstykh/mivolo_v2", trust_remote_code=True, dtype=torch.float32,
    )
    processor = AutoImageProcessor.from_pretrained(
        "iitolstykh/mivolo_v2", trust_remote_code=True,
    )
    model = model.to(device).eval()
    print(f"MiVOLO v2 loaded on {device}")
    return model, processor, config, device


def load_fairface(device=None):
    """Load FairFace ResNet34 model for race classification.

    Returns:
        Tuple of (model, transform, device_str).
    """
    import torch
    from torchvision import models, transforms

    if device is None:
        device = _auto_device()

    print("Loading FairFace...")
    model = models.resnet34(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 18)

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


def predict_age_gender(model, processor, config, device, face_crop):
    """Predict age and gender from a face crop.

    Args:
        face_crop: numpy array (RGB).

    Returns:
        Dict with age_exact, age_range, gender, gender_confidence.
    """
    import torch
    import numpy as np

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
    """Predict race from a face crop.

    Args:
        face_crop_pil: PIL Image (RGB).

    Returns:
        Dict with dominant_race and race_distribution.
    """
    import torch

    img_tensor = transform(face_crop_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
    race_logits = output[0, :7]
    race_probs = torch.nn.functional.softmax(race_logits, dim=0).cpu().numpy()
    race_dist = {FAIRFACE_RACE_LABELS[i]: float(race_probs[i]) * 100 for i in range(7)}
    dominant_race = FAIRFACE_RACE_LABELS[int(race_probs.argmax())]
    return {"dominant_race": dominant_race, "race_distribution": race_dist}
