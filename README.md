# BioCellAgents

> AI-Powered Computational Pathology & Multi-Agent Biomedical Analysis

Deep learning platform for PanNuke histopathology: DINOv2 cell segmentation/classification + CrewAI multi-agent orchestration for automated pathology reporting.

---

## Quick Start

```bash
# 1. Clone & setup
git clone https://github.com/g-zin/BioCellAgents-v1.git
cd BioCellAgents-v1/biocell-agent
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -e .

# 2. Configure
cp .env.example .env   # add your GROQ_API_KEY

# 3. Prepare data (see Data Setup below)

# 4. Run
python main.py --fold_dir data/raw/fold1 --patch_index 0   # multi-agent pipeline
streamlit run app.py                                        # web dashboard
```

**Prerequisites:** Python >= 3.10, CUDA 11.8+ (optional), Groq API key (for agent pipeline).

---

## Data Setup

```
data/
├── raw/fold1/          # PanNuke fold 1 (images + masks)
├── raw/fold2/          # PanNuke fold 2
├── processed/          # Preprocessed patches
└── splits/             # Train/val/test splits
```

Download from the [PanNuke Challenge](https://pannuke.grand-challenge.org) and place in `data/raw/`.

---

## Usage

| Task | Command |
|------|---------|
| Train cell segmenter | `python -m src.models.train --fold_dir data/raw/fold1` |
| Train classifier | `python -m src.models.train_classification --fold_dir data/raw/fold1` |
| Evaluate | `python -m src.models.evaluate --fold_dir data/raw/fold1` |
| Inference pipeline | `python -m src.models.inference_pipeline --fold_dir data/raw/fold1` |
| Multi-agent analysis | `python main.py --fold_dir data/raw/fold1 --patch_index 0` |
| Web dashboard | `streamlit run app.py` |
| Tests | `pytest tests/ -v` |

---

## Docker

```bash
docker compose up -d biocell-app              # dashboard on :8501
docker compose --profile training up -d       # + batch trainer
docker compose --profile training --profile agent up -d  # + Ollama LLM
```

---

## Project Structure

```
biocell-agent/
├── src/
│   ├── models/         # DINOv2 segmenter, classifier, losses, trainers
│   ├── inference/      # Evaluation, metrics, hyperparameter tuning
│   ├── tools/          # CrewAI tools (classifier, graph, parser)
│   ├── dashboard/      # Streamlit UI components
│   ├── utils/          # Logging, metrics tracking
│   └── crew.py         # Multi-agent orchestration
├── tests/              # 57 unit tests
├── scripts/            # Benchmark & diagnostic scripts
├── config/             # Agent & task YAML configs
├── main.py             # Pipeline entry point
├── app.py              # Streamlit dashboard
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

---

## License

[MIT](LICENSE)

## Contributing

Fork → branch → test (`pytest tests/ -v`) → PR.
