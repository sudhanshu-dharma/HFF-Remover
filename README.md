# HFF Remover

<h1 align="center">
  <br>
  <a href="https://openpecha.org"><img src="https://avatars.githubusercontent.com/u/82142807?s=400&u=19e108a15566f3a1449bafb03b8dd706a72aebcd&v=4" alt="OpenPecha" width="150"></a>
  <br>
</h1>

Remove **H**eaders, **F**ooters, and **F**ootnotes from scanned book images using DocLayout-YOLO.

## Overview

HFF Remover is a Python package that uses the [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO) model to automatically detect and mask headers, footers, and footnotes in scanned book images. It's designed to process large batches of images (200K+) efficiently using GPU acceleration.

## Features

- **Automatic Detection**: Uses DocLayout-YOLO to detect document layout elements
- **Batch Processing**: Process thousands of images with GPU acceleration
- **Resumable**: Checkpoint support for interruption recovery
- **Mixed Formats**: Supports JPEG, PNG, TIFF, BMP, and WebP
- **Configurable**: Adjustable confidence thresholds, padding, and output quality
- **CLI & API**: Both command-line and Python API interfaces

## Installation

```bash
# Clone the repository
git clone https://github.com/OpenPecha/HFF-Remover.git
cd HFF-Remover

# Install in development mode
pip install -e ".[dev]"
```

### Requirements

- Python >= 3.8
- CUDA-capable GPU (recommended for large batches)
- PyTorch >= 2.0.0

## Quick Start

### Command Line

```bash
# Process all images in a directory
hff-remover process /path/to/images --output /path/to/output

# Process with GPU and custom settings
hff-remover process /path/to/images --output /path/to/output \
    --device cuda \
    --batch-size 32 \
    --confidence 0.5 \
    --padding 5

# Resume interrupted processing
hff-remover process /path/to/images --output /path/to/output --resume

# Process a single image
hff-remover single input.jpg output.jpg

# Just detect without masking (view coordinates)
hff-remover detect input.jpg
```

### Python API

```python
from hff_remover import HFFDetector, HFFProcessor, BatchProcessor

# Initialize components
detector = HFFDetector(device="cuda", confidence_threshold=0.5)
processor = HFFProcessor(padding=5)

# Process a single image
from hff_remover.utils import load_image, save_image

image = load_image("input.jpg")
detections = detector.detect(image)
result = processor.mask_regions(image, detections)
save_image(result, "output.jpg")

# Batch processing
batch_processor = BatchProcessor(
    detector=detector,
    processor=processor,
    batch_size=32,
)

stats = batch_processor.process_directory(
    input_dir="/path/to/images",
    output_dir="/path/to/output",
    resume=True,
)

print(f"Processed {stats.processed_images} images")
print(f"Speed: {stats.images_per_second:.1f} images/sec")
```

## CLI Reference

### `hff-remover process`

Process all images in a directory.

| Option | Default | Description |
|--------|---------|-------------|
| `--output, -o` | Required | Output directory |
| `--device` | `cuda` | Device for inference (`cuda` or `cpu`) |
| `--batch-size` | `8` | Images per batch |
| `--confidence` | `0.5` | Minimum detection confidence |
| `--padding` | `0` | Extra pixels around detected regions |
| `--image-size` | `1024` | Model input size |
| `--quality` | `95` | Output image quality (0-100) |
| `--resume` | `false` | Resume from checkpoint |
| `--no-recursive` | `false` | Don't search subdirectories |

### `hff-remover single`

Process a single image.

```bash
hff-remover single input.jpg output.jpg --device cuda
```

### `hff-remover detect`

Detect HFF regions without masking.

```bash
hff-remover detect input.jpg --output detections.json
```

## Detection Classes

DocLayout-YOLO detects the following document elements. HFF Remover targets:

| Class | Name | Description |
|-------|------|-------------|
| 2 | abandon | Headers, footers, page numbers |
| 7 | table_footnote | Footnotes in tables |

## Performance

On a modern GPU (NVIDIA V100/T4):

- **Speed**: 50-100 images/second
- **200K images**: ~30-60 minutes

For best performance:
- Use SSD storage for faster I/O
- Increase batch size based on GPU memory
- Use multiple I/O workers (`--io-workers`)

## Cloud Deployment

For processing large batches on cloud GPUs:

```bash
# AWS/GCP with NVIDIA GPU
hff-remover process /data/images --output /data/output \
    --device cuda \
    --batch-size 64 \
    --io-workers 8 \
    --checkpoint-interval 1000
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=hff_remover
```
# Surya HFF detector

- **What:** A detector that uses the Surya layout model to find headers, footers, footnotes, and text in document images. Output uses our 4 classes only: **0=text, 1=footer, 2=header, 3=footnote** (no raw Surya labels).

- **Where:** `SuryaLayoutDetector` in `src/hff_remover/detector.py`. Same interface as other detectors: `detect()`, `detect_batch()`, `get_all_detections()`.

- **Saving:** YOLO output is written via the existing **processor** class (`YOLOInferenceDatasetWriter`). The Surya class has `save_to_yolo()` which calls that writer (optional same-class merge before saving).

- **Run:** `python test.py <image_path> [inference_dir]` — detects and saves to `inference_data/` (images, labels/*.txt, data.yaml). Use `--merge` to merge same-class boxes into one per class.

- **Dependencies:** `surya-ocr`, `transformers>=4.30.2,<5` (in pyproject.toml).

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO) for the document layout model
- [OpenPecha](https://openpecha.org) for supporting this project
