from __future__ import annotations

import argparse

import gradio as gr
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def main(checkpoint: str) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(device)
    model.eval()

    @torch.inference_mode()
    def translate(text: str, num_beams: int = 4, max_length: int = 128) -> str:
        text = (text or "").strip()
        if not text:
            return ""
        enc = tokenizer([text], return_tensors="pt", padding=True, truncation=True).to(device)
        out = model.generate(**enc, num_beams=num_beams, max_length=max_length)
        return tokenizer.decode(out[0], skip_special_tokens=True)

    demo = gr.Interface(
        fn=translate,
        inputs=[
            gr.Textbox(lines=4, label="English input"),
            gr.Slider(1, 8, value=4, step=1, label="Beam size"),
            gr.Slider(16, 256, value=128, step=8, label="Max output length"),
        ],
        outputs=gr.Textbox(lines=4, label="Arabic translation"),
        title="English → Arabic Translator (Fine-tuned MarianMT)",
        description="Loads your fine-tuned model from the checkpoint folder and runs generation.",
        flagging_mode="never",
    )

    demo.launch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Model folder, e.g. outputs/en-ar")
    args = parser.parse_args()
    main(args.checkpoint)
