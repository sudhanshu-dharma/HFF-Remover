"""Tests for the detector module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from hff_remover.detector import (
    HFFDetector,
    HFF_CLASSES,
    CLASS_NAMES,
    SuryaLayoutDetector,
    SURYA_HFF_CLASS_IDS,
    _surya_label_to_our_class,
)


class TestHFFClasses:
    """Tests for class definitions."""

    def test_hff_classes_defined(self):
        """Test that HFF classes are properly defined."""
        assert 2 in HFF_CLASSES  # abandon (headers/footers)
        assert 7 in HFF_CLASSES  # table_footnote

    def test_class_names_complete(self):
        """Test that all class names are defined."""
        for class_id in range(10):
            assert class_id in CLASS_NAMES


class TestHFFDetector:
    """Tests for HFFDetector class."""

    @patch("hff_remover.detector.hf_hub_download")
    @patch("hff_remover.detector.YOLOv10")
    def test_init_downloads_model(self, mock_yolo, mock_download):
        """Test that model is downloaded if path not provided."""
        mock_download.return_value = "/path/to/model.pt"

        detector = HFFDetector()

        mock_download.assert_called_once()
        mock_yolo.assert_called_once_with("/path/to/model.pt")

    @patch("hff_remover.detector.YOLOv10")
    def test_init_with_custom_path(self, mock_yolo):
        """Test initialization with custom model path."""
        detector = HFFDetector(model_path="/custom/model.pt")

        mock_yolo.assert_called_once_with("/custom/model.pt")
        assert detector.model_path == "/custom/model.pt"

    @patch("hff_remover.detector.hf_hub_download")
    @patch("hff_remover.detector.YOLOv10")
    def test_detect_filters_hff_classes(self, mock_yolo, mock_download):
        """Test that detect only returns HFF classes."""
        mock_download.return_value = "/path/to/model.pt"

        # Create mock detection results
        mock_boxes = MagicMock()
        mock_boxes.cls = MagicMock()
        mock_boxes.cls.__len__ = lambda self: 3
        mock_boxes.cls.__getitem__ = lambda self, i: MagicMock(
            item=lambda: [0, 2, 1][i]  # title, abandon, plain_text
        )
        mock_boxes.xyxy = MagicMock()
        mock_boxes.xyxy.__getitem__ = lambda self, i: MagicMock(
            cpu=lambda: MagicMock(numpy=lambda: MagicMock(tolist=lambda: [10, 10, 50, 50]))
        )
        mock_boxes.conf = MagicMock()
        mock_boxes.conf.__getitem__ = lambda self, i: MagicMock(item=lambda: 0.9)

        mock_result = MagicMock()
        mock_result.boxes = mock_boxes

        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_result]
        mock_yolo.return_value = mock_model

        detector = HFFDetector()
        detections = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))

        # Should only return the header/footer/table_footnote classes
        assert len(detections) == 1
        assert detections[0]["class_id"] == 2
        assert detections[0]["class_name"] == "header"

    @patch("hff_remover.detector.hf_hub_download")
    @patch("hff_remover.detector.YOLOv10")
    def test_detect_ignores_middle_abandon(self, mock_yolo, mock_download):
        """Test that abandon detections in the middle of the page are ignored."""
        mock_download.return_value = "/path/to/model.pt"

        # bbox with y-center at 50% of page height -> should be ignored
        mock_boxes = MagicMock()
        mock_boxes.__len__ = lambda self: 1
        mock_boxes.cls = [MagicMock(item=lambda: 2)]
        mock_boxes.xyxy = [
            MagicMock(cpu=lambda: MagicMock(numpy=lambda: MagicMock(tolist=lambda: [10, 40, 50, 60])))
        ]
        mock_boxes.conf = [MagicMock(item=lambda: 0.9)]

        mock_result = MagicMock()
        mock_result.boxes = mock_boxes

        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_result]
        mock_yolo.return_value = mock_model

        detector = HFFDetector()
        detections = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))
        assert detections == []

    @patch("hff_remover.detector.hf_hub_download")
    @patch("hff_remover.detector.YOLOv10")
    def test_get_all_detections_returns_all(self, mock_yolo, mock_download):
        """Test that get_all_detections returns all classes."""
        mock_download.return_value = "/path/to/model.pt"

        # Create mock detection results with multiple classes
        mock_boxes = MagicMock()
        mock_boxes.__len__ = lambda self: 2
        mock_boxes.cls = [MagicMock(item=lambda: 0), MagicMock(item=lambda: 2)]
        mock_boxes.xyxy = [
            MagicMock(cpu=lambda: MagicMock(numpy=lambda: MagicMock(tolist=lambda: [10, 10, 50, 50]))),
            MagicMock(cpu=lambda: MagicMock(numpy=lambda: MagicMock(tolist=lambda: [60, 60, 90, 90]))),
        ]
        mock_boxes.conf = [MagicMock(item=lambda: 0.9), MagicMock(item=lambda: 0.8)]

        mock_result = MagicMock()
        mock_result.boxes = mock_boxes

        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_result]
        mock_yolo.return_value = mock_model

        detector = HFFDetector()

        # Mock the iteration over boxes
        with patch.object(detector, 'get_all_detections') as mock_get_all:
            mock_get_all.return_value = [
                {"class_id": 0, "class_name": "title", "bbox": [10, 10, 50, 50], "confidence": 0.9},
                {"class_id": 2, "class_name": "abandon", "bbox": [60, 60, 90, 90], "confidence": 0.8},
            ]

            detections = detector.get_all_detections(np.zeros((100, 100, 3)))

            assert len(detections) == 2

    @patch("hff_remover.detector.hf_hub_download")
    @patch("hff_remover.detector.YOLOv10")
    def test_detect_batch(self, mock_yolo, mock_download):
        """Test batch detection."""
        mock_download.return_value = "/path/to/model.pt"

        # Create mock results for batch
        mock_boxes = MagicMock()
        mock_boxes.__len__ = lambda self: 1
        mock_boxes.cls = [MagicMock(item=lambda: 2)]
        mock_boxes.xyxy = [
            MagicMock(cpu=lambda: MagicMock(numpy=lambda: MagicMock(tolist=lambda: [10, 10, 50, 50])))
        ]
        mock_boxes.conf = [MagicMock(item=lambda: 0.9)]

        mock_result = MagicMock()
        mock_result.boxes = mock_boxes

        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_result, mock_result]
        mock_yolo.return_value = mock_model

        detector = HFFDetector()
        images = [np.zeros((100, 100, 3)), np.zeros((100, 100, 3))]

        results = detector.detect_batch(images, batch_size=2)

        assert len(results) == 2

    @patch("hff_remover.detector.hf_hub_download")
    @patch("hff_remover.detector.YOLOv10")
    def test_detect_empty_result(self, mock_yolo, mock_download):
        """Test detection with no results."""
        mock_download.return_value = "/path/to/model.pt"

        mock_result = MagicMock()
        mock_result.boxes = None

        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_result]
        mock_yolo.return_value = mock_model

        detector = HFFDetector()
        detections = detector.detect(np.zeros((100, 100, 3)))

        assert len(detections) == 0

    @patch("hff_remover.detector.hf_hub_download")
    @patch("hff_remover.detector.YOLOv10")
    def test_confidence_threshold(self, mock_yolo, mock_download):
        """Test that confidence threshold is passed to model."""
        mock_download.return_value = "/path/to/model.pt"
        mock_model = MagicMock()
        mock_model.predict.return_value = [MagicMock(boxes=None)]
        mock_yolo.return_value = mock_model

        detector = HFFDetector(confidence_threshold=0.7)
        detector.detect(np.zeros((100, 100, 3)))

        # Check that predict was called with the confidence threshold
        call_kwargs = mock_model.predict.call_args[1]
        assert call_kwargs["conf"] == 0.7


# =============================================================================
# Surya Layout Detector tests (our labels only; no YOLO/save_results in class)
# =============================================================================

class TestSuryaHFFClassMapping:
    """Tests for Surya HFF class constants and mapping (no surya import needed)."""

    def test_surya_hff_class_ids(self):
        """Our class_id mapping: 0=text, 1=footer, 2=header, 3=footnote."""
        assert SURYA_HFF_CLASS_IDS[0] == "text"
        assert SURYA_HFF_CLASS_IDS[1] == "footer"
        assert SURYA_HFF_CLASS_IDS[2] == "header"
        assert SURYA_HFF_CLASS_IDS[3] == "footnote"
        assert len(SURYA_HFF_CLASS_IDS) == 4

    def test_surya_label_to_our_class_mapping(self):
        """Normalized Surya labels map to our (class_id, class_name)."""
        assert _surya_label_to_our_class("Text") == (0, "text")
        assert _surya_label_to_our_class("Page-footer") == (1, "footer")
        assert _surya_label_to_our_class("Page-header") == (2, "header")
        assert _surya_label_to_our_class("Footnote") == (3, "footnote")
        assert _surya_label_to_our_class("Section-header") == (2, "header")

    def test_surya_label_unknown_returns_none_when_not_default(self):
        """Unknown Surya labels return None when default_to_text=False."""
        assert _surya_label_to_our_class("Table") is None
        assert _surya_label_to_our_class("Picture") is None

    def test_surya_label_unknown_maps_to_text_when_default(self):
        """Unknown Surya labels map to (0, 'text') when default_to_text=True."""
        assert _surya_label_to_our_class("Table", default_to_text=True) == (0, "text")


class TestSuryaLayoutDetector:
    """Tests for SuryaLayoutDetector: our class_id/class_name only, no raw Surya labels."""

    @pytest.fixture(autouse=True)
    def skip_if_no_surya(self):
        pytest.importorskip("surya")

    @patch("surya.layout.LayoutPredictor")
    @patch("surya.foundation.FoundationPredictor")
    def test_detect_returns_only_our_classes(self, mock_foundation, mock_layout_cls):
        """detect() returns only our HFF/text classes with our class_id and class_name."""
        mock_layout = MagicMock()
        mock_layout.return_value = [
            {
                "bboxes": [
                    {"bbox": [10, 10, 100, 30], "label": "Page-header", "top_k": {"Page-header": 0.9}},
                    {"bbox": [10, 500, 100, 530], "label": "Page-footer", "top_k": {"Page-footer": 0.85}},
                    {"bbox": [10, 100, 400, 400], "label": "Text", "top_k": {"Text": 0.95}},
                ]
            }
        ]
        mock_layout_cls.return_value = mock_layout
        mock_foundation.return_value = MagicMock()

        detector = SuryaLayoutDetector(confidence_threshold=0.3)
        detections = detector.detect(np.zeros((600, 500, 3), dtype=np.uint8))

        assert len(detections) == 3
        class_ids = {d["class_id"] for d in detections}
        class_names = {d["class_name"] for d in detections}
        assert class_ids <= {0, 1, 2, 3}
        assert class_names <= {"text", "footer", "header", "footnote"}
        for d in detections:
            assert "bbox" in d and len(d["bbox"]) == 4
            assert "confidence" in d
            assert isinstance(d["class_id"], int)
            assert d["class_name"] in ("text", "footer", "header", "footnote")

    @patch("surya.layout.LayoutPredictor")
    @patch("surya.foundation.FoundationPredictor")
    def test_detect_filters_by_confidence(self, mock_foundation, mock_layout_cls):
        """detect() filters out detections below confidence threshold."""
        mock_layout = MagicMock()
        mock_layout.return_value = [
            {
                "bboxes": [
                    {"bbox": [10, 10, 100, 30], "label": "Page-header", "top_k": {"Page-header": 0.9}},
                    {"bbox": [10, 500, 100, 530], "label": "Footnote", "top_k": {"Footnote": 0.2}},
                ]
            }
        ]
        mock_layout_cls.return_value = mock_layout
        mock_foundation.return_value = MagicMock()

        detector = SuryaLayoutDetector(confidence_threshold=0.5)
        detections = detector.detect(np.zeros((600, 500, 3), dtype=np.uint8))

        assert len(detections) == 1
        assert detections[0]["class_name"] == "header"

    @patch("surya.layout.LayoutPredictor")
    @patch("surya.foundation.FoundationPredictor")
    def test_detect_no_raw_surya_labels(self, mock_foundation, mock_layout_cls):
        """Output never contains raw Surya label names (e.g. Page-header, Text)."""
        mock_layout = MagicMock()
        mock_layout.return_value = [
            {
                "bboxes": [
                    {"bbox": [10, 10, 100, 30], "label": "Page-header", "top_k": {"Page-header": 0.9}},
                ]
            }
        ]
        mock_layout_cls.return_value = mock_layout
        mock_foundation.return_value = MagicMock()

        detector = SuryaLayoutDetector(confidence_threshold=0.3)
        detections = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))

        raw_names = {"Page-header", "Page-footer", "Footnote", "Text", "Section-header", "Caption"}
        for d in detections:
            assert d["class_name"] not in raw_names
            assert d["class_name"] in ("text", "footer", "header", "footnote")

    @patch("surya.layout.LayoutPredictor")
    @patch("surya.foundation.FoundationPredictor")
    def test_detect_batch_same_format(self, mock_foundation, mock_layout_cls):
        """detect_batch() returns list of lists with same dict format (our class_id/class_name)."""
        mock_layout = MagicMock()
        mock_layout.return_value = [
            {"bboxes": [{"bbox": [0, 0, 50, 20], "label": "Text", "top_k": {"Text": 0.8}}]},
            {"bboxes": [{"bbox": [0, 0, 50, 20], "label": "Footnote", "top_k": {"Footnote": 0.7}}]},
        ]
        mock_layout_cls.return_value = mock_layout
        mock_foundation.return_value = MagicMock()

        detector = SuryaLayoutDetector(confidence_threshold=0.3)
        results = detector.detect_batch(
            [np.zeros((100, 100, 3), dtype=np.uint8), np.zeros((100, 100, 3), dtype=np.uint8)]
        )

        assert len(results) == 2
        assert len(results[0]) == 1 and results[0][0]["class_id"] == 0 and results[0][0]["class_name"] == "text"
        assert len(results[1]) == 1 and results[1][0]["class_id"] == 3 and results[1][0]["class_name"] == "footnote"

    @patch("surya.layout.LayoutPredictor")
    @patch("surya.foundation.FoundationPredictor")
    def test_get_all_detections_same_format(self, mock_foundation, mock_layout_cls):
        """get_all_detections() returns our class_id and class_name only."""
        mock_layout = MagicMock()
        mock_layout.return_value = [
            {
                "bboxes": [
                    {"bbox": [0, 0, 50, 20], "label": "Text", "top_k": {"Text": 0.8}},
                    {"bbox": [0, 80, 50, 100], "label": "Table", "top_k": {"Table": 0.9}},
                ]
            }
        ]
        mock_layout_cls.return_value = mock_layout
        mock_foundation.return_value = MagicMock()

        detector = SuryaLayoutDetector(confidence_threshold=0.3)
        detections = detector.get_all_detections(np.zeros((100, 100, 3), dtype=np.uint8))

        # Table maps to (0, "text") when filter_to_hff_only=False (default_to_text=True)
        assert len(detections) == 2
        for d in detections:
            assert d["class_id"] in (0, 1, 2, 3)
            assert d["class_name"] in ("text", "footer", "header", "footnote")

    @patch("surya.layout.LayoutPredictor")
    @patch("surya.foundation.FoundationPredictor")
    def test_detect_empty_bboxes(self, mock_foundation, mock_layout_cls):
        """detect() returns empty list when layout returns no bboxes."""
        mock_layout = MagicMock()
        mock_layout.return_value = [{"bboxes": []}]
        mock_layout_cls.return_value = mock_layout
        mock_foundation.return_value = MagicMock()

        detector = SuryaLayoutDetector()
        detections = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))
        assert detections == []
