import gradio as gr
from transformers import pipeline
import torch

MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"

sentiment_pipeline = pipeline(
    "text-classification",
    model=MODEL_ID,
    top_k=None,
    device=0 if torch.cuda.is_available() else -1,
)

LABEL_MAP = {"negative": "Negative", "neutral": "Neutral", "positive": "Positive"}
EMOJI_MAP = {"Negative": "🔴", "Neutral": "🟡", "Positive": "🟢"}


def analyze_single(text: str):
    if not text.strip():
        return {}, ""
    results = sentiment_pipeline(text[:512])[0]
    scores = {LABEL_MAP.get(r["label"].lower(), r["label"]): round(r["score"], 4) for r in results}
    top = max(scores, key=scores.get)
    summary = f"{EMOJI_MAP[top]} **{top}** — {scores[top]*100:.1f}% confidence"
    return scores, summary


def analyze_batch(texts: str):
    lines = [l.strip() for l in texts.strip().split("\n") if l.strip()]
    if not lines:
        return "No input provided."
    results = sentiment_pipeline([l[:512] for l in lines])
    output = []
    for line, result in zip(lines, results):
        scores = {LABEL_MAP.get(r["label"].lower(), r["label"]): round(r["score"], 4) for r in result}
        top = max(scores, key=scores.get)
        preview = line[:80] + ("..." if len(line) > 80 else "")
        output.append(f"{EMOJI_MAP[top]} **{top}** ({scores[top]*100:.1f}%) — {preview}")
    return "\n\n".join(output)


with gr.Blocks(theme=gr.themes.Soft(), title="Sentiment Analysis") as demo:
    gr.Markdown("""
# 🎭 Sentiment Analysis
**Multi-class sentiment classification · RoBERTa fine-tuned on 124M tweets**

Detects Positive, Neutral, and Negative sentiment with per-class confidence scores.
Supports single text and batch inference.
    """)

    with gr.Tabs():
        with gr.Tab("Single Text"):
            with gr.Row():
                with gr.Column(scale=1):
                    text_input = gr.Textbox(
                        label="Input Text",
                        placeholder="Enter text to analyze...",
                        lines=5,
                    )
                    analyze_btn = gr.Button("Analyze Sentiment", variant="primary")
                with gr.Column(scale=1):
                    summary_output = gr.Markdown(label="Top Prediction")
                    scores_output = gr.Label(label="Confidence Scores", num_top_classes=3)

            analyze_btn.click(fn=analyze_single, inputs=text_input, outputs=[scores_output, summary_output])
            text_input.submit(fn=analyze_single, inputs=text_input, outputs=[scores_output, summary_output])

            gr.Examples(
                examples=[
                    ["The new product exceeded all my expectations. Absolutely love it!"],
                    ["The service was okay, nothing special but nothing terrible either."],
                    ["Terrible experience. Waited 2 hours and the food was cold. Never again."],
                    ["Package arrived. Haven't opened it yet."],
                    ["I can't believe how bad this is. Completely broken on arrival."],
                ],
                inputs=text_input,
                label="Try an Example",
            )

        with gr.Tab("Batch Analysis"):
            gr.Markdown("Enter multiple texts, **one per line**. Up to 50 at a time.")
            batch_input = gr.Textbox(
                label="Input Texts",
                placeholder="I love this product!\nThe quality is mediocre.\nAbsolute disaster, do not buy.",
                lines=10,
            )
            batch_btn = gr.Button("Analyze All", variant="primary")
            batch_output = gr.Markdown(label="Results")
            batch_btn.click(fn=analyze_batch, inputs=batch_input, outputs=batch_output)

    gr.Markdown("""
---
Built by [Nagendra Chowdary](https://github.com/Nagendra7299) ·
Model: [`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)
    """)


if __name__ == "__main__":
    demo.launch()
