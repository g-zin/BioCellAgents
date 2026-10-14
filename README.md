# BioCellAgents

> **AI-Powered Computational Pathology & Multi-Agent Biomedical Analysis**

BioCellAgents is a deep learning platform for PanNuke histopathology analysis, combining DINOv2-based cell segmentation/classification with CrewAI multi-agent orchestration for automated pathology reporting.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        BioCellAgents Platform                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐   ┌──────────────────┐   ┌───────────────────┐  │
│  │  Data Layer  │   │   Model Layer    │   │  Inference Layer  │  │
│  │              │   │                  │   │                   │  │
│  │ • PanNuke    │   │ • DINOv2 Cell    │   │ • Two-Stage       │  │
│  │   Datasets   │   │   Segmenter      │   │   Cell Analyzer   │  │
│  │ • Transforms │   │ • DINOv2 Cell    │   │ • Evaluator       │  │
│  │ • Preprocess │   │   Classifier     │   │ • Hyperparam      │  │
│  │ • Validation │   │ • Organ          │   │   Tuner           │  │
│  │              │   │   Classifier     │   │ • Metrics         │  │
│  └──────────────┘   │ • BaseTrainer    │   └───────────────────┘  │
│                     │ • Losses         │                           │
│                     └──────────────────┘                           │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    Multi-Agent Orchestration                  │  │
│  │  CrewAI: PathologyCrew → Parser → Classifier → Graph → Report│  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    Infrastructure & Utilities                 │  │
│  │  • Structured Logging (JSON + Console)                       │  │
│  │  • Metrics Tracking (JSON/CSV/TensorBoard)                   │  │
│  │  • Docker Deployment (multi-stage, GPU-ready)                │  │
│  │  • CI/CD Pipeline (lint → test → docker-build)               │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📦 Installation

### Prerequisites

- Python ≥ 3.10
- CUDA 11.8+ (optional, for GPU training)
- Docker & Docker Compose (for deployment)
- Groq API key (for multi-agent LLM pipeline)

### Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/<your-org>/BioCellAgents.git
cd BioCellAgents/biocell-agent

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Install in development mode with CLI entry points
pip install -e .

# 5. Configure environment
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# 6. Download data & checkpoints (see below)

# 7. Run the pipeline
python main.py --fold_dir data/raw/fold1 --patch_index 0

# 8. Launch the web dashboard
streamlit run app.py
```

### Data Setup

The pipeline expects PanNuke data in the following structure:

```
data/
├── raw/
│   ├── fold1/
│   │   ├── images/       # H&E patch images (PNG)
│   │   └── masks/        # Nucleus segmentation masks (PNG)
│   ├── fold2/
│   └── fold3/
├── processed/
│   └── metadata.csv      # Preprocessing metadata
└── cell_patches/         # Extracted cell crops (generated)
```

Download the [PanNuke dataset](https://pannuke.grand-challenge.org) and organize it into the structure above.

### Model Checkpoints

Trained checkpoints are expected at:

```
checkpoints/
├── cell_vits14/
│   └── best_model.pt
├── organ_vits14/
│   ├── best_model.pt
│   └── calibration.json
└── segmentation_vits14/
    └── best_segmentation_model.pt
```

You can train your own models (see [Training](#training)) or obtain pre-trained checkpoints from the project maintainers.

### Docker

```bash
# Build the image
docker build -t biocell-agent .

# Run the Streamlit app
docker run -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  biocell-agent

# Or use Docker Compose (all services)
docker compose up -d

# With GPU support
docker compose --profile training up -d biocell-trainer

# With Ollama LLM agent
docker compose --profile agent up -d ollama
```

---

## 🚀 Usage

### Training

```bash
# Cell Segmentation (DINOv2)
python -m src.models.train \
  --metadata_path data/processed/metadata.csv \
  --train_fold 1,2 \
  --val_fold 3 \
  --epochs 25 \
  --batch_size 16 \
  --model_size vits14

# Cell Classification
python -m src.models.train_classification \
  --task cell \
  --epochs 30 \
  --batch_size 32

# Organ Classification
python -m src.models.train_classification \
  --task organ \
  --epochs 20 \
  --batch_size 16
```

### Evaluation

```bash
# Evaluate cell classification
python -m src.models.evaluate \
  --task cell \
  --model_path checkpoints/cell_vits14/best_model.pt \
  --val_fold 3

# Evaluate segmentation
python -m src.models.evaluate \
  --task segmentation \
  --model_path checkpoints/segmentation_vits14/best_model.pt

# Evaluate organ classification
python -m src.models.evaluate \
  --task organ \
  --model_path checkpoints/organ_vits14/best_model.pt
```

### Multi-Agent Pathology Pipeline

```bash
# Run full CrewAI multi-agent analysis
python main.py --fold_dir data/raw/fold1 --patch_index 0
```

### Web App (Streamlit Dashboard)

```bash
# Launch the interactive web dashboard
streamlit run app.py
```

The web app provides:
- **Image upload** — drag & drop H&E-stained tissue images (PNG, JPG, TIFF, BMP)
- **Sample image** — generate a synthetic demo image for quick testing
- **Two-stage analysis** — Cellpose segmentation + DINOv2 classification in one click
- **Visualizations** — original image, segmentation mask, annotated cells side-by-side
- **Results table** — per-cell ID, class, confidence, bounding box, centroid
- **Distribution charts** — interactive pie & bar charts (Plotly)
- **Validity gate** — statistical assessment of cell count reliability
- **Export** — download results as JSON or CSV

> **Note:** The web app requires trained model checkpoints at
> `checkpoints/cell_vits14/best_model.pt`. Without checkpoints, the
> classifier runs with random weights (useful for UI testing only).

### Hyperparameter Tuning

```python
from src.inference import HyperparameterTuner

tuner = HyperparameterTuner(
    param_grid={
        "learning_rate": [1e-5, 1e-4, 5e-4],
        "batch_size": [8, 16, 32],
        "weight_decay": [0.0, 1e-4],
    },
    search_strategy="random",
    max_trials=20,
)
best = tuner.run(train_fn, metric="val_dice", maximize=True)
tuner.save_results("tuning_results.json")

---

## 📊 Metrics & Logging

### Metrics Tracking

```python
from src.utils import MetricsTracker

tracker = MetricsTracker(output_dir="logs/metrics")
tracker.log("train_loss", 0.45, epoch=1)
tracker.log("val_dice", 0.72, epoch=1)
tracker.log_event("checkpoint_saved", {"path": "checkpoints/best.pt"})

best = tracker.get_best("val_dice", maximize=True)
summary = tracker.get_summary()
tracker.save()
```

### Structured Logging

```python
from src.utils import setup_logger, TrainingLogger, InferenceLogger, get_gpu_memory_info

# Basic logger
logger = setup_logger("my_module", log_file="logs/my_module.log")
logger.info("Training started", extra={"epoch": 1, "lr": 1e-4})

# Training logger (structured JSON)
train_logger = TrainingLogger(output_dir="logs")
train_logger.log_epoch(epoch=1, train_loss=0.45, val_dice=0.72, lr=1e-4)

# Inference logger
infer_logger = InferenceLogger(output_dir="logs")
infer_logger.log_inference(image_id="patch_001", prediction="Lymphocyte", confidence=0.94)

# GPU memory monitoring
print(get_gpu_memory_info())
```

---

## 📊 Segmentation Model Comparison (PanNuke, fold 3)

> **Canonical metric:** *per-batch mean Dice* — for each batch, sum the per-class
> Dice over the classes present, divide by the number of classes, then average over
> batches. This is the metric used by `evaluate.py`, the trainer's
> `validate_epoch`, and `src/inference/evaluator.py`, so all reported numbers are
> directly comparable. A *global/pooled Dice* is also shown for reference; it is
> more optimistic because it pools pixels across the whole set before averaging.

| Model | Per-batch mean Dice | Global Dice |
|-------|:---:|:---:|
| DINOv2 ViT-S/14 + decoder | 0.3419 | 0.5585 |
| **DeepLabV3+ (ResNet-50)** | **0.3630** | **0.6427** |

DeepLabV3+ is a consistent but **modest** improvement over the DINOv2 baseline on
both metrics. The much larger gap seen in raw training logs was an artifact of
comparing the trainer's global Dice against `evaluate.py`'s per-batch Dice; the
codebase now uses a single canonical metric throughout.

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=src --cov-report=xml

# Run specific test file
pytest tests/test_losses.py -v
pytest tests/test_metrics_tracker.py -v
pytest tests/test_transforms.py -v
pytest tests/test_metrics.py -v
```

**Current test coverage: 48 tests across 4 test suites**

| Suite | Tests | Coverage |
|-------|-------|----------|
| `test_losses.py` | 9 | DiceLoss, CombinedSegmentationLoss |
| `test_metrics.py` | 8 | compute_metrics, save_results |
| `test_metrics_tracker.py` | 13 | MetricsTracker, LoggingConfig |
| `test_transforms.py` | 18 | clean_organ_label, crop_nucleus, extract_cell_crops, extract_organ_patches |

---

## 🐳 Docker Services

| Service | Port | Description |
|---------|------|-------------|
| `biocell-app` | 8501 | Streamlit web dashboard |
| `biocell-trainer` | — | Batch training jobs (profile: `training`) |
| `biocell-ollama` | 11434 | Ollama LLM for agent analysis (profile: `agent`) |

```bash
# Start app only
docker compose up -d biocell-app

# Start app + trainer
docker compose --profile training up -d

# Start all services including LLM
docker compose --profile training --profile agent up -d
```

---

## 📁 Project Structure

```
biocell-agent/
├── src/
│   ├── data/           # Datasets, transforms, preprocessing
│   ├── models/         # Model architectures, losses, trainers
│   ├── inference/      # Evaluation, metrics, hyperparameter tuning
│   ├── tools/          # CrewAI tools (classifier, graph analytics, parser)
│   ├── utils/          # Logging, metrics tracking
│   └── crew.py         # Multi-agent orchestration
├── tests/              # Unit tests (48 tests)
├── scripts/            # Benchmark & diagnostic scripts
├── config/             # Agent & task configurations
├── checkpoints/        # Model checkpoints (gitignored)
├── data/               # PanNuke dataset (gitignored)
├── .github/workflows/  # CI/CD pipeline
├── Dockerfile          # Multi-stage Docker build
├── docker-compose.yml  # Service orchestration
├── main.py             # Multi-agent pipeline entry point
├── app.py              # Streamlit dashboard
├── pyproject.toml      # Project configuration
├── requirements.txt    # Dependencies
├── LICENSE             # MIT License
└── .env.example        # Environment variable template
```

---

## 📋 CLI Entry Points

| Command | Description |
|---------|-------------|
| `biocell-train` | Train DINOv2 cell segmentation model |
| `biocell-eval` | Evaluate trained models |
| `biocell-infer` | Run inference pipeline |

---

## 📄 License

[MIT License](LICENSE)

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`pytest tests/ -v`)
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

---

## 📚 References

- **PanNuke Challenge**: [pannuke.grand-challenge.org](https://pannuke.grand-challenge.org)
- **DINOv2**: [arxiv.org/abs/2304.07193](https://arxiv.org/abs/2304.07193)
- **CrewAI**: [docs.crewai.com](https://docs.crewai.com)
- **PyTorch**: [pytorch.org](https://pytorch.org)


