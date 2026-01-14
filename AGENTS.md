# Repository Guidelines

## Project Structure & Module Organization
- `api/` contains the FastAPI entrypoint (`api/main.py`).
- `collectors/` fetches draw data per lottery (e.g., `collectors/usa_powerball.py`).
- `models_ai/` houses ML models and training code; `models_ai/src/` for implementations and `models_ai/trained/` for weights.
- `models_stat/` includes statistical analysis models (e.g., `physics_bias.py`).
- `scripts/` provides CLI utilities for data maintenance, training, and ONNX conversion.
- `data/` stores lottery-specific datasets (JSON/SQLite).
- `web/`, `web-static/`, and `web-vue/` provide UI variants (static HTML, static assets/ONNX, and Quasar/Vue app).
- `config/` holds lottery and training configuration JSON files.

## Build, Test, and Development Commands
- `python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt` to set up Python dependencies.
- `uvicorn api.main:app --reload` or `./run.sh` to run the API locally.
- `python models_ai/src/transformer/train.py --lottery korea_645 --epochs 30` to train a Transformer model.
- `python models_ai/src/lstm/train.py --lottery korea_645 --epochs 30` to train an LSTM model.
- `python scripts/convert_to_onnx.py` to export models for web usage.
- `cd web-vue && npm install && quasar dev` to run the Vue/Quasar frontend.

## Coding Style & Naming Conventions
- Python: 4-space indentation; prefer descriptive module names like `lotto_transformer.py`.
- JavaScript/Vue: follow existing Quasar/Vue patterns in `web-vue/src/` and keep component names in PascalCase.
- Config/data files: use snake_case IDs like `korea_645` to match `config/lotteries.json` and `data/` folders.

## Testing Guidelines
- No formal test suite is present. Validate changes by:
  - Running the API locally and calling endpoints used by `web/` or `web-vue/`.
  - Executing training or generation scripts for the affected model.
  - Checking model artifacts under `models_ai/trained/` or `web-static/models/` when relevant.

## Commit & Pull Request Guidelines
- Commit messages in history use short prefixes like `fix:`, `Docs:`, or `UI:`. Follow that pattern and keep messages concise.
- PRs should describe scope, list manual checks (commands run), and include screenshots for UI changes. Link related issues when applicable.

## Configuration & Data Notes
- Lottery definitions live in `config/lotteries.json`; update this first when adding a new lottery.
- Large data or model files should stay in their existing folders and avoid renaming unless necessary.
