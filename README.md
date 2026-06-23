# Neural Machine Translation (English → Arabic) with Transformers

This project fine-tunes a pretrained Transformer (MarianMT) using a parallel dataset:
**Translation Dataset.csv** with columns: `English`, `Arabic`.

## Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
