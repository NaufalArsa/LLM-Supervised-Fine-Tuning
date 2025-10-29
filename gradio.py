import gradio as gr
from transformers import pipeline

generator = pipeline("text-generation", model="dtp-fine-tuning/phi4-alpacaid")

def chat(prompt):
    response = generator(prompt, max_new_tokens=150)[0]["generated_text"]
    return response

gr.Interface(fn=chat, inputs="text", outputs="text").launch()