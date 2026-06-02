import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "wangchao-nlp/qwen2.5-0.5b-zh-en-lora"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

def translate(text):
    if not text.strip():
        return "Please enter Chinese text."

    messages = [
        {"role": "system", "content": "你是一个专业的翻译助手。只输出英文翻译，不添加任何额外内容。"},
        {"role": "user", "content": f"翻译成英文：{text}"},
    ]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "assistant" in response:
        response = response.split("assistant")[-1].strip()

    for sep in ['. ', '.\n', '!"', ".'", '.']:
        idx = response.find(sep)
        if 3 < idx < 200:
            response = response[:idx+1]
            break

    return response

demo = gr.Interface(
    fn=translate,
    inputs=gr.Textbox(lines=5, placeholder="输入中文，按回车翻译", label="Chinese"),
    outputs=gr.Textbox(lines=5, label="English Translation"),
    title="Chinese-English Translation - Qwen2.5 + LoRA",
    description="LoRA (rank=16) fine-tuned Qwen2.5-0.5B-Instruct on OPUS-100 | BLEU: 15.20 (+1.55 over baseline)",
    examples=[
        ["人工智能正在改变世界。"],
        ["今天天气真好。"],
        ["这本书很有趣。"],
        ["深度学习需要大量计算资源。"],
    ],
)

demo.launch()
