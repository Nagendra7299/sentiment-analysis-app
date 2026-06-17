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
short_description: Multi-class sentiment classification using RoBERTa with batch inference
---

# 🎭 Sentiment Analysis App

> **In plain English:** This app reads a piece of text and tells you whether the feeling behind it is **Positive**, **Neutral**, or **Negative** — like a mood detector for words.

---

## What Does It Do?

Have you ever read a product review and wondered "is this person happy or angry?" This app answers that automatically.

You type in any text — a tweet, a review, a comment — and the app instantly tells you:
- 😊 **Positive** — the writer is happy, satisfied, or excited
- 😐 **Neutral** — the writer is indifferent or just stating facts
- 😠 **Negative** — the writer is unhappy, frustrated, or critical

It also shows a **confidence score** — how sure the AI is about its answer (e.g. "92% Positive").

---

## Two Ways to Use It

### 1. Single Text
Paste one sentence or paragraph → click **Analyze Sentiment** → get instant result.

### 2. Batch Analysis
Paste multiple texts (one per line) → click **Analyze All** → get results for all of them at once. Useful for analyzing many reviews or comments quickly.

---

## Real-World Use Cases

| Who | How they use it |
|-----|----------------|
| Business owners | Understand if customers are happy with their product |
| Marketers | Gauge public reaction to a campaign or announcement |
| Researchers | Analyze thousands of social media posts at once |
| Students | Learn how AI understands human language |

---

## How Does the AI Work? (Simple Version)

Under the hood, this app uses a type of AI called a **language model** — specifically one called **RoBERTa**, developed by Meta AI.

Think of it like this:

> Imagine someone who has read **124 million tweets** and learned to recognize patterns in how people write when they're happy vs. upset. That's essentially what this model has done — it studied massive amounts of real text and learned to detect emotional tone automatically.

The model doesn't just guess — it calculates a probability for each emotion and picks the most likely one.

---

## Technical Details (For Developers)

- **Model:** [`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)
- **Architecture:** RoBERTa-base fine-tuned on ~124M tweets
- **Classes:** Positive · Neutral · Negative
- **Max input length:** 512 tokens (~380 words)
- **Framework:** Gradio + HuggingFace Transformers
- **Hardware:** CPU (free tier) — GPU supported via `device=0`

### Run Locally

```bash
# Clone the repo
git clone https://github.com/Nagendra7299/sentiment-analysis-app.git
cd sentiment-analysis-app

# Install dependencies
pip install -r requirements.txt

# Launch app
python app.py
```

App opens at `http://localhost:7860`

### Project Structure

```
sentiment-analysis-app/
├── app.py            # Gradio UI + inference logic
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

---

## Live Demo

Try it instantly — no setup needed:
👉 **[huggingface.co/spaces/Nagendra729/Sentiment-analysis](https://huggingface.co/spaces/Nagendra729/Sentiment-analysis)**

---

## Author

Built by **Nagendra Chowdary** — AI/ML Engineer

- GitHub: [github.com/Nagendra7299](https://github.com/Nagendra7299)
- Portfolio: [portfolio-one-chi-cvtu2ez2sd.vercel.app](https://portfolio-one-chi-cvtu2ez2sd.vercel.app)
