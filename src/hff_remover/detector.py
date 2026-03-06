"""Document layout detectors for headers, footers, and footnotes."""

from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod
import numpy as np


# =============================================================================
# Base Detector Interface
# =============================================================================

class BaseHFFDetector(ABC):
    """Abstract base class for HFF detectors."""

    @abstractmethod
    def detect(
        self,
        image: Union[str, Path, np.ndarray],
        image_size: int = 1024,
    ) -> List[Dict[str, Any]]:
        """
        Detect headers, footers, and footnotes in an image.

        Args:
            image: Path to image file or numpy array.
            image_size: Input size for the model.

        Returns:
            List of detection dictionaries with keys:
                - bbox: [x1, y1, x2, y2] bounding box coordinates
                - class_id: Class ID of the detection
                - class_name: Human-readable class name
                - confidence: Confidence score
        """
        pass

    @abstractmethod
    def detect_batch(
        self,
        images: List[Union[str, Path, np.ndarray]],
        image_size: int = 1024,
        batch_size: int = 8,
    ) -> List[List[Dict[str, Any]]]:
        """Detect HFF in multiple images."""
        pass


# =============================================================================
# DocLayout-YOLO Detector
# =============================================================================

# DocLayout-YOLO class mapping
# Reference: https://github.com/opendatalab/DocLayout-YOLO
DOCLAYOUT_YOLO_CLASS_NAMES = {
    0: "title",
    1: "plain_text",
    2: "abandon",
    3: "figure",
    4: "figure_caption",
    5: "table",
    6: "table_caption",
    7: "table_footnote",
    8: "isolate_formula",
    9: "formula_caption",
}

# Classes we want to detect (Header, Footer, Footnote, Text)
# In DocLayout-YOLO, "abandon" (class 2) represents headers/footers/page numbers
# "table_footnote" (class 7) represents footnotes in tables
# "plain_text" (class 1) represents text areas / body text
DOCLAYOUT_YOLO_HFF_CLASSES = {
    1: "plain_text",     # Text area / body text
    2: "abandon",        # Headers, footers, page numbers
    7: "table_footnote", # Table footnotes
}


class HFFDetector(BaseHFFDetector):
    """Detector for headers, footers, and footnotes using DocLayout-YOLO."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        confidence_threshold: float = 0.5,
        header_region_ratio: float = 0.33,
        footer_region_ratio: float = 0.67,
    ):
        """
        Initialize the HFF detector.

        Args:
            model_path: Path to the DocLayout-YOLO model weights.
                       If None, downloads from HuggingFace Hub.
            device: Device to run inference on ('cuda' or 'cpu').
            confidence_threshold: Minimum confidence score for detections.
        """
        from doclayout_yolo import YOLOv10
        from huggingface_hub import hf_hub_download

        self.device = device
        self.confidence_threshold = confidence_threshold
        self.header_region_ratio = header_region_ratio
        self.footer_region_ratio = footer_region_ratio

        if not (0.0 <= self.header_region_ratio <= 1.0) or not (0.0 <= self.footer_region_ratio <= 1.0):
            raise ValueError("header_region_ratio and footer_region_ratio must be within [0, 1]")
        if self.header_region_ratio >= self.footer_region_ratio:
            raise ValueError("header_region_ratio must be < footer_region_ratio")

        if model_path is None:
            # Download model from HuggingFace Hub
            model_path = hf_hub_download(
                repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
                filename="doclayout_yolo_docstructbench_imgsz1024.pt",
            )

        self.model = YOLOv10(model_path)
        self.model_path = model_path

    def _get_image_height(self, image: Union[str, Path, np.ndarray]) -> Optional[int]:
        """Best-effort extraction of image height for positional labeling."""
        if isinstance(image, np.ndarray):
            return int(image.shape[0])
        try:
            import cv2
            img_arr = cv2.imread(str(image))
            if img_arr is None:
                return None
            return int(img_arr.shape[0])
        except Exception:
            return None

    def _doclayout_to_hff_detection(
        self,
        class_id: int,
        bbox: List[float],
        confidence: float,
        image_height: Optional[int],
    ) -> Optional[Dict[str, Any]]:
        """Convert DocLayout-YOLO box into an HFF detection dict, or None to ignore."""
        # Keep plain_text / text area (class 1)
        if class_id == 1:
            return {
                "bbox": bbox,
                "class_id": class_id,
                "class_name": "text",
                "confidence": confidence,
            }

        # Keep table_footnote (class 7)
        if class_id == 7:
            return {
                "bbox": bbox,
                "class_id": class_id,
                "class_name": "table_footnote",
                "confidence": confidence,
            }

        # Handle abandon (class 2) by position: header/footer/middle-ignore
        if class_id == 2:
            if image_height is None:
                class_name: Optional[str] = "abandon"
            else:
                class_name = self._classify_abandon_by_position(bbox, image_height)
            if class_name is None:
                return None
            return {
                "bbox": bbox,
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
            }

        # Ignore all other classes
        return None

    def _get_result_image_height(self, result: Any, src: Union[str, Path, np.ndarray]) -> Optional[int]:
        """Best-effort extraction of image height for a single prediction result."""
        try:
            if hasattr(result, "orig_shape") and result.orig_shape is not None:
                return int(result.orig_shape[0])
        except Exception:
            pass
        return self._get_image_height(src)

    def _extract_hff_detections_from_boxes(
        self,
        boxes: Any,
        image_height: Optional[int],
    ) -> List[Dict[str, Any]]:
        """Extract HFF detections from a DocLayout-YOLO boxes object."""
        if boxes is None:
            return []

        detections: List[Dict[str, Any]] = []
        for j in range(len(boxes)):
            class_id = int(boxes.cls[j].item())
            bbox = boxes.xyxy[j].cpu().numpy().tolist()
            confidence = float(boxes.conf[j].item())

            det = self._doclayout_to_hff_detection(
                class_id=class_id,
                bbox=bbox,
                confidence=confidence,
                image_height=image_height,
            )
            if det is not None:
                detections.append(det)

        return detections

    def _classify_abandon_by_position(
        self,
        bbox_xyxy: List[float],
        image_height: int,
    ) -> Optional[str]:
        """
        For DocLayout-YOLO class_id=2 ("abandon"), decide whether it's a header/footer.

        - If bbox vertical center is in top `header_region_ratio` of page -> "header"
        - If bbox vertical center is in bottom `footer_region_ratio` of page -> "footer"
        - Otherwise -> None (ignore)
        """
        if image_height <= 0:
            return None

        y1 = float(bbox_xyxy[1])
        y2 = float(bbox_xyxy[3])
        y_center = (y1 + y2) / 2.0
        y_norm = y_center / float(image_height)

        if y_norm <= self.header_region_ratio:
            return "header"
        if y_norm >= self.footer_region_ratio:
            return "footer"
        return None

    def detect(
        self,
        image: Union[str, Path, np.ndarray],
        image_size: int = 1024,
    ) -> List[Dict[str, Any]]:
        """
        Detect headers, footers, and footnotes in an image.

        Args:
            image: Path to image file or numpy array (BGR format).
            image_size: Input size for the model.

        Returns:
            List of detection dictionaries with keys:
                - bbox: [x1, y1, x2, y2] bounding box coordinates
                - class_id: Class ID of the detection
                - class_name: Human-readable class name
                - confidence: Confidence score
        """
        # Run inference
        results = self.model.predict(
            image,
            imgsz=image_size,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
        )

        # Determine image height for positional header/footer labeling
        image_height = self._get_image_height(image)

        detections: List[Dict[str, Any]] = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                class_id = int(boxes.cls[i].item())

                bbox = boxes.xyxy[i].cpu().numpy().tolist()
                confidence = boxes.conf[i].item()

                det = self._doclayout_to_hff_detection(
                    class_id=class_id,
                    bbox=bbox,
                    confidence=float(confidence),
                    image_height=image_height,
                )
                if det is not None:
                    detections.append(det)

        return detections

    def detect_batch(
        self,
        images: List[Union[str, Path, np.ndarray]],
        image_size: int = 1024,
        batch_size: int = 8,
    ) -> List[List[Dict[str, Any]]]:
        """
        Detect headers, footers, and footnotes in multiple images.

        Args:
            images: List of image paths or numpy arrays.
            image_size: Input size for the model.
            batch_size: Number of images to process at once.

        Returns:
            List of detection lists, one per input image.
        """
        all_detections = []

        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]

            # Process batch
            results = self.model.predict(
                batch,
                imgsz=image_size,
                conf=self.confidence_threshold,
                device=self.device,
                verbose=False,
            )

            for idx, result in enumerate(results):
                src = batch[idx]
                image_height = self._get_result_image_height(result, src)
                all_detections.append(
                    self._extract_hff_detections_from_boxes(result.boxes, image_height)
                )

        return all_detections

    def get_all_detections(
        self,
        image: Union[str, Path, np.ndarray],
        image_size: int = 1024,
    ) -> List[Dict[str, Any]]:
        """
        Get all document layout detections (not just HFF).

        Useful for debugging or visualization.

        Args:
            image: Path to image file or numpy array.
            image_size: Input size for the model.

        Returns:
            List of all detection dictionaries.
        """
        results = self.model.predict(
            image,
            imgsz=image_size,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
        )

        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                class_id = int(boxes.cls[i].item())
                bbox = boxes.xyxy[i].cpu().numpy().tolist()
                confidence = boxes.conf[i].item()

                detections.append({
                    "bbox": bbox,
                    "class_id": class_id,
                    "class_name": DOCLAYOUT_YOLO_CLASS_NAMES.get(class_id, f"unknown_{class_id}"),
                    "confidence": confidence,
                })

        return detections


# =============================================================================
# PP-DocLayout-L Detector (PaddlePaddle)
# =============================================================================

# PP-DocLayout class mapping for HFF + text
# Reference: https://huggingface.co/PaddlePaddle/PP-DocLayout-L
PP_DOCLAYOUT_HFF_CLASSES = {
    "header",
    "footer",
    "footnote",
    "footnotes",       # Alternative naming
    "page_number",
    "page-header",
    "page-footer",
    "text",            # Text area / body text
    "plain text",      # Alternative naming
    "plain_text",      # Alternative naming
    "paragraph",       # Alternative naming
}


class PPDocLayoutDetector(BaseHFFDetector):
    """Detector for headers, footers, and footnotes using PP-DocLayout (PaddlePaddle)."""

    def __init__(
        self,
        model_name: str = "PP-DocLayout-L",
        confidence_threshold: float = 0.5,
        use_gpu: bool = False,
    ):
        """
        Initialize the PP-DocLayout detector.

        Args:
            model_name: Model name (not used, kept for API compatibility).
            confidence_threshold: Minimum confidence score for detections.
            use_gpu: Whether to use GPU for inference.
        """
        from paddleocr import PPStructure

        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.use_gpu = use_gpu

        # Initialize PPStructure for layout analysis
        self.model = PPStructure(
            layout=True,
            table=False,
            ocr=False,
            show_log=False,
            use_gpu=use_gpu,
            enable_mkldnn=False,  # Disable MKLDNN to avoid bugs
        )

    def detect(
        self,
        image: Union[str, Path, np.ndarray],
        image_size: int = 1024,
    ) -> List[Dict[str, Any]]:
        """
        Detect headers, footers, and footnotes in an image.

        Args:
            image: Path to image file or numpy array.
            image_size: Input size (not used for PP-DocLayout, kept for API compatibility).

        Returns:
            List of detection dictionaries with keys:
                - bbox: [x1, y1, x2, y2] bounding box coordinates
                - class_id: Class ID (mapped to string label)
                - class_name: Human-readable class name
                - confidence: Confidence score
        """
        import cv2

        # Load image if path provided
        if isinstance(image, (str, Path)):
            img_array = cv2.imread(str(image))
        else:
            img_array = image

        # Run inference - PPStructure takes numpy array directly
        results = self.model(img_array)

        detections = []

        # PPStructure returns list of dicts with 'type' and 'bbox' keys
        for item in results:
            if not isinstance(item, dict):
                continue

            label = item.get('type', '').lower()
            # PPStructure doesn't always provide confidence, default to 1.0
            score = item.get('score', 1.0)
            bbox = item.get('bbox', [])

            # Filter by confidence
            if score < self.confidence_threshold:
                continue

            # Only keep HFF classes
            if label not in PP_DOCLAYOUT_HFF_CLASSES:
                continue

            # Normalize label name
            normalized_label = self._normalize_label(label)

            # Convert bbox to [x1, y1, x2, y2] format
            if len(bbox) == 4:
                bbox = list(map(float, bbox))
            elif len(bbox) == 8:
                # Polygon format: get bounding box
                xs = bbox[0::2]
                ys = bbox[1::2]
                bbox = [min(xs), min(ys), max(xs), max(ys)]
            else:
                continue

            detections.append({
                "bbox": bbox,
                "class_id": label,
                "class_name": normalized_label,
                "confidence": float(score),
            })

        return detections

    def detect_batch(
        self,
        images: List[Union[str, Path, np.ndarray]],
        image_size: int = 1024,
        batch_size: int = 8,
    ) -> List[List[Dict[str, Any]]]:
        """
        Detect headers, footers, and footnotes in multiple images.

        Args:
            images: List of image paths or numpy arrays.
            image_size: Input size (not used for PP-DocLayout).
            batch_size: Number of images to process at once.

        Returns:
            List of detection lists, one per input image.
        """
        all_detections = []

        for image in images:
            detections = self.detect(image, image_size)
            all_detections.append(detections)

        return all_detections

    def _normalize_label(self, label: str) -> str:
        """Normalize label names to standard format."""
        label = label.lower().replace("-", "_").replace(" ", "_")

        if label in ("header", "page_header"):
            return "header"
        elif label in ("footer", "page_footer", "page_number"):
            return "footer"
        elif label in ("footnote", "footnotes"):
            return "footnote"
        elif label in ("text", "plain_text", "paragraph"):
            return "text"

        return label

    def get_all_detections(
        self,
        image: Union[str, Path, np.ndarray],
        image_size: int = 1024,
    ) -> List[Dict[str, Any]]:
        """
        Get all document layout detections (not just HFF).

        Args:
            image: Path to image file or numpy array.
            image_size: Input size.

        Returns:
            List of all detection dictionaries.
        """
        _ = image_size  # kept for API compatibility
        import cv2

        # Load image if path provided
        if isinstance(image, (str, Path)):
            img_array = cv2.imread(str(image))
        else:
            img_array = image

        # Run inference
        results = self.model(img_array)

        detections = []

        for item in results:
            if not isinstance(item, dict):
                continue

            label = item.get('type', '')
            score = item.get('score', 1.0)
            bbox = item.get('bbox', [])

            if score < self.confidence_threshold:
                continue

            # Convert bbox to [x1, y1, x2, y2] format
            if len(bbox) == 4:
                bbox = list(map(float, bbox))
            elif len(bbox) == 8:
                xs = bbox[0::2]
                ys = bbox[1::2]
                bbox = [min(xs), min(ys), max(xs), max(ys)]
            else:
                continue

            detections.append({
                "bbox": bbox,
                "class_id": label,
                "class_name": label,
                "confidence": float(score),
            })

        return detections


# Surya layout detector (our labels: 0=text, 1=footer, 2=header, 3=footnote)
SURYA_HFF_CLASS_IDS = {
    0: "text",
    1: "footer",
    2: "header",
    3: "footnote",
}
SURYA_LABEL_TO_OUR_CLASS = {
    "text": (0, "text"),
    "caption": (0, "text"),
    "list-item": (0, "text"),
    "formula": (0, "text"),
    "text-inline-math": (0, "text"),
    "handwriting": (0, "text"),
    "table-of-contents": (0, "text"),
    "form": (0, "text"),
    "table": (0, "text"),
    "picture": (0, "text"),
    "figure": (0, "text"),
    "section-header": (2, "header"),
    "page-header": (2, "header"),
    "page-footer": (1, "footer"),
    "footnote": (3, "footnote"),
}


def _surya_label_to_our_class(surya_label: str, default_to_text: bool = False) -> Optional[tuple]:
    key = surya_label.strip().lower().replace(" ", "-") if surya_label else ""
    out = SURYA_LABEL_TO_OUR_CLASS.get(key)
    if out is not None:
        return out
    return (0, "text") if default_to_text else None


class SuryaLayoutDetector(BaseHFFDetector):
    def __init__(
        self,
        confidence_threshold: float = 0.5,
        layout_checkpoint: Optional[str] = None,
    ):
        from surya.layout import LayoutPredictor
        from surya.foundation import FoundationPredictor
        from surya.settings import settings

        self.confidence_threshold = confidence_threshold
        checkpoint = layout_checkpoint or getattr(settings, "LAYOUT_MODEL_CHECKPOINT", None)
        foundation = FoundationPredictor() if checkpoint is None else FoundationPredictor(checkpoint=checkpoint)
        self.layout_predictor = LayoutPredictor(foundation)

    def _load_image(self, image: Union[str, Path, np.ndarray]) -> Any:
        from PIL import Image

        if isinstance(image, (str, Path)):
            return Image.open(str(image)).convert("RGB")
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                return Image.fromarray(image).convert("RGB")
            if image.shape[2] == 3:
                return Image.fromarray(image[:, :, ::-1])
            return Image.fromarray(image).convert("RGB")
        return image

    def _layout_result_to_detections(
        self,
        layout_pred: Any,
        apply_confidence_filter: bool = True,
        filter_to_hff_only: bool = True,
    ) -> List[Dict[str, Any]]:
        detections: List[Dict[str, Any]] = []
        if hasattr(layout_pred, "bboxes"):
            bboxes_data = layout_pred.bboxes or []
        elif isinstance(layout_pred, dict):
            bboxes_data = layout_pred.get("bboxes") or layout_pred.get("boxes") or []
        else:
            bboxes_data = []

        for item in bboxes_data:
            raw_label = (item.get("label") or item.get("type") or "") if isinstance(item, dict) else getattr(item, "label", "")
            bbox = item.get("bbox") if isinstance(item, dict) else getattr(item, "bbox", None)
            top_k = item.get("top_k") if isinstance(item, dict) else getattr(item, "top_k", None) or {}
            conf = 1.0
            if isinstance(item, dict) and "confidence" in item:
                conf = float(item["confidence"])
            elif hasattr(item, "confidence") and item.confidence is not None:
                conf = float(item.confidence)
            elif isinstance(top_k, dict) and top_k:
                key = str(raw_label).strip().lower().replace(" ", "-") if raw_label else ""
                conf = float(top_k.get(raw_label) or top_k.get(key) or list(top_k.values())[0])

            if apply_confidence_filter and conf < self.confidence_threshold:
                continue

            raw_label = str(raw_label or "")
            mapped = _surya_label_to_our_class(raw_label, default_to_text=not filter_to_hff_only)
            if mapped is None:
                continue
            our_class_id, our_class_name = mapped

            if not bbox or len(bbox) < 4:
                continue
            bbox = list(bbox)
            if len(bbox) == 4:
                bbox = list(map(float, bbox))
            elif len(bbox) == 8:
                xs, ys = bbox[0::2], bbox[1::2]
                bbox = [min(xs), min(ys), max(xs), max(ys)]
            else:
                continue

            detections.append({
                "bbox": bbox,
                "class_id": our_class_id,
                "class_name": our_class_name,
                "confidence": conf,
            })

        return detections

    def detect(
        self,
        image: Union[str, Path, np.ndarray],
        image_size: int = 1024,
    ) -> List[Dict[str, Any]]:
        pil_image = self._load_image(image)
        layout_predictions = self.layout_predictor([pil_image])
        if not layout_predictions:
            return []
        return self._layout_result_to_detections(
            layout_predictions[0],
            apply_confidence_filter=True,
            filter_to_hff_only=True,
        )

    def detect_batch(
        self,
        images: List[Union[str, Path, np.ndarray]],
        image_size: int = 1024,
        batch_size: int = 8,
    ) -> List[List[Dict[str, Any]]]:
        pil_list = [self._load_image(im) for im in images]
        layout_predictions = self.layout_predictor(pil_list)

        all_detections: List[List[Dict[str, Any]]] = []
        for pred in layout_predictions:
            all_detections.append(
                self._layout_result_to_detections(
                    pred,
                    apply_confidence_filter=True,
                    filter_to_hff_only=True,
                )
            )
        return all_detections

    def get_all_detections(
        self,
        image: Union[str, Path, np.ndarray],
        image_size: int = 1024,
    ) -> List[Dict[str, Any]]:
        _ = image_size
        pil_image = self._load_image(image)
        layout_predictions = self.layout_predictor([pil_image])
        if not layout_predictions:
            return []
        return self._layout_result_to_detections(
            layout_predictions[0],
            apply_confidence_filter=True,
            filter_to_hff_only=False,
        )

    def save_to_yolo(
        self,
        image: Union[str, Path, np.ndarray],
        image_rel_path: Union[str, Path],
        inference_dir: Union[str, Path] = "inference_data",
        detections: Optional[List[Dict[str, Any]]] = None,
        merge_same_class: bool = True,
    ) -> tuple:
        """
        Save detections in YOLO format using processor.YOLOInferenceDatasetWriter.
        Same output as processor: inference_dir/images/, labels/*.txt, data.yaml.
        Returns (image_out_path, label_out_path).
        """
        from hff_remover.processor import YOLOInferenceDatasetWriter
        from hff_remover.utils import load_image

        if detections is None:
            detections = self.detect(image)
        if isinstance(image, (str, Path)):
            image = load_image(image)
        if merge_same_class:
            by_class: Dict[int, List[Dict[str, Any]]] = {}
            for d in detections:
                cid = d.get("class_id")
                if cid is None:
                    continue
                by_class.setdefault(cid, []).append(d)
            merged = []
            for cid, group in by_class.items():
                boxes = [b.get("bbox") for b in group if b.get("bbox") and len(b.get("bbox", [])) == 4]
                if not boxes:
                    continue
                x1 = min(b[0] for b in boxes)
                y1 = min(b[1] for b in boxes)
                x2 = max(b[2] for b in boxes)
                y2 = max(b[3] for b in boxes)
                best = max(group, key=lambda b: b.get("confidence", 0))
                merged.append({
                    "bbox": [x1, y1, x2, y2],
                    "class_id": cid,
                    "class_name": best.get("class_name", ""),
                    "confidence": best.get("confidence", 0),
                })
            detections = merged
        class_name_to_id = {name: idx for idx, name in SURYA_HFF_CLASS_IDS.items()}
        writer = YOLOInferenceDatasetWriter(
            base_dir=inference_dir,
            class_name_to_id=class_name_to_id,
        )
        return writer.write_sample(
            image=image,
            detections=detections,
            image_rel_path=image_rel_path,
        )


# =============================================================================
# Ensemble Detector
# =============================================================================

class EnsembleDetector(BaseHFFDetector):
    """Ensemble detector that combines results from multiple detectors."""

    def __init__(
        self,
        detectors: List[BaseHFFDetector],
        merge_strategy: str = "union",
        iou_threshold: float = 0.5,
    ):
        """
        Initialize ensemble detector.

        Args:
            detectors: List of detector instances to use.
            merge_strategy: How to combine results ('union', 'intersection', 'cascade').
            iou_threshold: IoU threshold for merging overlapping boxes.
        """
        self.detectors = detectors
        self.merge_strategy = merge_strategy
        self.iou_threshold = iou_threshold

    def detect(
        self,
        image: Union[str, Path, np.ndarray],
        image_size: int = 1024,
    ) -> List[Dict[str, Any]]:
        """
        Detect using all detectors and merge results.

        Args:
            image: Path to image file or numpy array.
            image_size: Input size for the model.

        Returns:
            Merged list of detections.
        """
        all_detections = []

        if self.merge_strategy == "cascade":
            # Use first detector, only use second if first returns nothing or fails
            for detector in self.detectors:
                try:
                    detections = detector.detect(image, image_size)
                    if detections:
                        return detections
                except Exception:
                    # This detector failed, try the next one
                    continue
            return []

        # Collect all detections (robust: continue if one detector fails)
        for detector in self.detectors:
            try:
                detections = detector.detect(image, image_size)
                all_detections.extend(detections)
            except Exception:
                # This detector failed, continue with others
                continue

        if self.merge_strategy == "union":
            # Merge overlapping boxes using NMS
            return self._non_max_suppression(all_detections)
        if self.merge_strategy == "intersection":
            # Only keep boxes detected by all detectors
            # (simplified: treat as union for now)
            return self._non_max_suppression(all_detections)

        return all_detections

    def detect_batch(
        self,
        images: List[Union[str, Path, np.ndarray]],
        image_size: int = 1024,
        batch_size: int = 8,
    ) -> List[List[Dict[str, Any]]]:
        """Detect in multiple images."""
        return [self.detect(img, image_size) for img in images]

    def _non_max_suppression(
        self,
        detections: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Apply non-maximum suppression to merge overlapping boxes."""
        if not detections:
            return []

        # Sort by confidence
        sorted_dets = sorted(detections, key=lambda x: x["confidence"], reverse=True)
        keep = []

        while sorted_dets:
            best = sorted_dets.pop(0)
            keep.append(best)

            # Remove boxes with high IoU with the best box
            sorted_dets = [
                det for det in sorted_dets
                if self._compute_iou(best["bbox"], det["bbox"]) < self.iou_threshold
            ]

        return keep

    def _compute_iou(self, box1: List[float], box2: List[float]) -> float:
        """Compute IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)

        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union_area = box1_area + box2_area - inter_area

        if union_area <= 0:
            return 0.0

        return inter_area / union_area


# =============================================================================
# Convenience aliases (backward compatibility)
# =============================================================================

# Keep old names for backward compatibility
CLASS_NAMES = DOCLAYOUT_YOLO_CLASS_NAMES
HFF_CLASSES = DOCLAYOUT_YOLO_HFF_CLASSES
