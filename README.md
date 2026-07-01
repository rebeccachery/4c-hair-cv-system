# 4C Hair CV System

A computer vision pipeline for analyzing Type 4 (coily/kinky) hair from photos. The system segments hair in an image, extracts texture and moisture proxies, estimates curl geometry (4A/4B/4C-like), and generates a prioritized care recommendation plan.

This project is designed for experimentation and prototyping. It can run on local images or capture frames from a [Viam](https://www.viam.com/) robot camera.

## Pipeline overview

```
Image input
    │
    ▼
Hair segmentation (SegFormer)
    │
    ├──► Feature extraction (frizz, shine, definition, scalp visibility)
    │
    └──► Curl analysis (FFT + orientation geometry)
              │
              ▼
        Condition classification
              │
              ▼
        Care recommendations (urgent / short-term / long-term)
```

### Stages

| Stage | Module | Description |
| --- | --- | --- |
| 1. Segmentation | `modules/segmentation.py` | Uses Hugging Face `nvidia/segformer-b0-finetuned-ade-512-512` to isolate the hair region |
| 2. Feature extraction | `modules/features.py` | Computes edge density, frizz score, curl variance, definition score, shine proxy, and scalp visibility |
| 3. Curl analysis | `modules/curl.py` | Estimates curl frequency, curvature, and orientation variance to classify 4A/4B/4C-like patterns |
| 4. Recommendations | `modules/recommendations.py` | Maps detected state to actionable hair-care suggestions |

The full pipeline is orchestrated in `modules/pipeline.py` via `run_hair_pipeline()`.

## Project structure

```
4c-hair-cv-system/
├── 4c_beauty_assistant.py              # Main CLI entry point
├── 01_4c_hair_vision_pipeline_segmentation.py
├── 02_4c_hair_feature_extraction_layer.py
├── 03_4c_hair_state_recommendation_engine_v1.py
├── modules/
│   ├── pipeline.py                     # End-to-end pipeline
│   ├── segmentation.py                 # Hair mask generation
│   ├── features.py                     # Visual feature extraction
│   ├── curl.py                         # Curl geometry + type estimate
│   ├── recommendations.py              # Classification + care plan
│   ├── image_utils.py                  # Image/mask conversion helpers
│   └── viam_adapter.py                 # Viam camera integration
├── sample.jpg                          # Example input image
└── requirements.txt
```

The numbered `01_`–`03_` scripts are standalone demos for each pipeline layer. Use `4c_beauty_assistant.py` to run the full system.

## Requirements

- Python 3.10+
- ~2 GB disk space for model weights (downloaded on first run)
- Optional: CUDA GPU for faster segmentation

## Installation

```bash
git clone <repository-url>
cd 4c-hair-cv-system

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

On first run, the SegFormer model is downloaded from Hugging Face automatically.

## Usage

### Full pipeline (recommended)

Analyze the bundled sample image and print JSON output:

```bash
python 4c_beauty_assistant.py
```

Analyze a custom image:

```bash
python 4c_beauty_assistant.py --image path/to/photo.jpg
```

Print a human-readable care report:

```bash
python 4c_beauty_assistant.py --image path/to/photo.jpg --report
```

### Viam robot camera

Capture a frame from a Viam machine camera and run the pipeline:

```bash
python 4c_beauty_assistant.py \
  --robot-address "<machine-address>" \
  --api-key "<api-key>" \
  --api-key-id "<api-key-id>" \
  --camera-name "<camera-component-name>" \
  --report
```

All four Viam flags are required when using robot mode. The CV modules themselves do not require `viam-sdk`; it is only imported when Viam mode is used.

### Individual pipeline stages

Run each layer independently for development or debugging:

```bash
# Stage 1: segmentation only
python 01_4c_hair_vision_pipeline_segmentation.py

# Stage 2: segmentation + feature extraction
python 02_4c_hair_feature_extraction_layer.py

# Stage 3: recommendation engine (uses sample feature data)
python 03_4c_hair_state_recommendation_engine_v1.py
```

Stages 1 and 2 expect `sample.jpg` in the project root by default.

## Output

The full pipeline returns a dictionary with these keys:

| Key | Contents |
| --- | --- |
| `features` | Raw numeric metrics (`edge_density`, `frizz_score`, `curl_variance`, etc.) |
| `curl` | Curl embedding, type estimate (e.g. `4C-like`), and confidence |
| `state` | Classified moisture state, detangling sensitivity, and maintenance priority |
| `recommendations` | Flat list of care actions |
| `plan` | Recommendations grouped into `urgent`, `short_term`, and `long_term` |
| `report` | Formatted text summary (same as `--report` output) |

Example:

```bash
python 4c_beauty_assistant.py --image sample.jpg --report
```

## Programmatic use

```python
from PIL import Image
from modules.pipeline import run_hair_pipeline

image = Image.open("sample.jpg").convert("RGB")
result = run_hair_pipeline(image)

print(result["curl"]["curl_type_estimate"])
print(result["report"])
```

## Dependencies

| Package | Purpose |
| --- | --- |
| `torch` | SegFormer inference (GPU optional) |
| `transformers` | Hugging Face image segmentation pipeline |
| `opencv-python` | Image processing and HSV analysis |
| `scipy` | Circular variance and FFT for curl geometry |
| `scikit-image` | Canny edge detection |
| `pillow` | Image I/O |
| `numpy` | Array operations |
| `viam-sdk` | Optional Viam robot camera integration |

See `requirements.txt` for pinned versions.

## Limitations

- **Prototype, not clinical advice.** Recommendations are heuristic and based on visual proxies, not a trained dermatological model.
- **Lighting and angle matter.** Shine, frizz, and scalp visibility scores are sensitive to photo conditions.
- **Curl typing is approximate.** The 4A/4B/4C estimate uses geometry heuristics, not a validated curl-pattern classifier.
- **Segmentation quality varies.** The model looks for a `hair` label in SegFormer output; poor framing or occlusions may produce empty masks.
- **First run is slow.** Model download and initialization add startup time.

## License

See repository license file for details.
