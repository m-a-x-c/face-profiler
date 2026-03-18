import pathlib

from face_profiler.constants import age_to_range


class FaceProfiler:
    """High-level API for face detection and attribute analysis.

    Example::

        from face_profiler import FaceProfiler

        profiler = FaceProfiler()
        results = profiler.analyze("photo.jpg")
        for face in results:
            print(f"{face['gender']}, {face['age_range']}, {face['race']}")

        annotated = profiler.render("photo.jpg")
        annotated.save("output.jpg")
    """

    def __init__(self, device=None):
        """
        Args:
            device: ``"cuda"``, ``"cpu"``, or ``None`` (auto-detect).
                    Models are loaded lazily on first call to :meth:`analyze`.
        """
        self._device = device
        self._mivolo = None
        self._fairface = None

    def _ensure_models(self):
        if self._mivolo is None:
            from face_profiler.models import load_mivolo
            self._mivolo = load_mivolo(device=self._device)
        if self._fairface is None:
            from face_profiler.models import load_fairface
            self._fairface = load_fairface(device=self._device)

    def _normalize_input(self, image):
        """Return (img_path_or_array, cv_img) suitable for detection and cropping.

        Accepts str, pathlib.Path, PIL.Image, or numpy array.
        """
        import numpy as np

        if isinstance(image, (str, pathlib.Path)):
            import cv2
            path = str(image)
            cv_img = cv2.imread(path)
            if cv_img is None:
                raise FileNotFoundError(f"Could not read image: {path}")
            return path, cv_img

        # PIL Image
        try:
            from PIL import Image
            if isinstance(image, Image.Image):
                cv_img = np.array(image.convert("RGB"))[:, :, ::-1]  # RGB -> BGR
                return cv_img, cv_img
        except ImportError:
            pass

        # numpy array
        if isinstance(image, np.ndarray):
            return image, image

        raise TypeError(f"Unsupported image type: {type(image)}")

    def analyze(self, image):
        """Analyze all faces in an image.

        Args:
            image: file path (``str``/``Path``), ``PIL.Image``, or numpy array.

        Returns:
            List of dicts, one per detected face::

                {
                    "box": (x1, y1, x2, y2),
                    "confidence": 0.98,
                    "age": 24.3,
                    "age_range": "18-25",
                    "gender": "Female",
                    "gender_confidence": 97.2,
                    "race": "East Asian",
                    "race_distribution": {"White": 2.1, ...},
                }
        """
        import cv2
        from PIL import Image

        from face_profiler.detection import detect_faces, crop_face
        from face_profiler.models import predict_age_gender, predict_race

        self._ensure_models()
        mivolo_model, mivolo_proc, mivolo_config, mivolo_device = self._mivolo
        ff_model, ff_transform, ff_device = self._fairface

        detect_input, cv_img = self._normalize_input(image)
        faces = detect_faces(detect_input)

        results = []
        for face in faces:
            face_crop_cv = crop_face(cv_img, face["box"])
            face_crop_rgb = cv2.cvtColor(face_crop_cv, cv2.COLOR_BGR2RGB)
            face_crop_pil = Image.fromarray(face_crop_rgb)

            ag = predict_age_gender(mivolo_model, mivolo_proc, mivolo_config, mivolo_device, face_crop_rgb)
            race = predict_race(ff_model, ff_transform, ff_device, face_crop_pil)

            results.append({
                "box": face["box"],
                "confidence": face["confidence"],
                "age": ag["age_exact"],
                "age_range": ag["age_range"],
                "gender": ag["gender"],
                "gender_confidence": ag["gender_confidence"],
                "race": race["dominant_race"],
                "race_distribution": race["race_distribution"],
            })

        return results

    def render(self, image, results=None, scale=1.0):
        """Produce an annotated image with face boxes and info cards.

        Args:
            image: file path (``str``/``Path``), ``PIL.Image``, or numpy array.
            results: output from :meth:`analyze`. If ``None``, runs analysis first.
            scale: display scale factor (default 1.0).

        Returns:
            ``PIL.Image`` with annotations drawn.
        """
        import tempfile
        from PIL import Image

        from face_profiler.rendering import render_annotated_image

        # render_annotated_image needs a file path
        if isinstance(image, (str, pathlib.Path)):
            img_path = str(image)
        else:
            # Save to temp file
            if isinstance(image, Image.Image):
                pil_img = image
            else:
                import numpy as np
                pil_img = Image.fromarray(image[:, :, ::-1] if image.shape[2] == 3 else image)
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            pil_img.save(tmp.name)
            img_path = tmp.name

        if results is None:
            results = self.analyze(image)

        # Convert flat results back to faces/results format for render function
        faces = [{"box": r["box"], "confidence": r["confidence"]} for r in results]
        render_results = [
            {
                "age_exact": r["age"],
                "age_range": r["age_range"],
                "gender": r["gender"],
                "gender_confidence": r["gender_confidence"],
                "dominant_race": r["race"],
                "race_distribution": r["race_distribution"],
            }
            for r in results
        ]

        return render_annotated_image(img_path, faces, render_results, scale=scale)
