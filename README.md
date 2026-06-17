---
title: Sentiment Analysis
emoji: 🎭
colorFrom: purple
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: true
license: mit
short_description: Multi-class sentiment classification using RoBERTa
---

# Sentiment Analysis API

Multi-class sentiment classifier (Positive / Neutral / Negative) using `cardiffnlp/twitter-roberta-base-sentiment-latest` — a RoBERTa model fine-tuned on 124M tweets.

## Features

- Single text analysis with per-class confidence scores
- Batch inference (up to 50 texts at once)
- GPU-accelerated when available, CPU fallback
- Sub-100ms latency on GPU

## Model

[`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)

## Local Setup

```bash
pip install -r requirements.txt
python app.py
```

## Author

[Nagendra Chowdary](https://github.com/Nagendra7299)
