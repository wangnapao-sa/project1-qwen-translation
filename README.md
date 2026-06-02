# QLoRA Fine-tuned Qwen2.5 for Chinese-English Translation

LoRA (rank=16) fine-tuned [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) on OPUS-100 en-zh for Chinese-to-English translation.

**Live Demo**: Run `python app.py` locally (GPU required), or launch on [Kaggle](https://www.kaggle.com/) with a free T4 GPU. See screenshot below.

**Model**: [wangchao-nlp/qwen2.5-0.5b-zh-en-lora](https://huggingface.co/wangchao-nlp/qwen2.5-0.5b-zh-en-lora)

## Results

| Model | BLEU |
|---|---|
| Qwen2.5-0.5B (zero-shot) | 13.65 |
| Qwen2.5-0.5B + LoRA (ours) | **15.20** (+1.55) |

BLEU evaluated on 200 test samples from OPUS-100 en-zh using SACREBLEU.

## Training Details

- **Dataset**: [Helsinki-NLP/opus-100](https://huggingface.co/datasets/Helsinki-NLP/opus-100) en-zh, 20K filtered training pairs
- **Method**: LoRA (rank=16, alpha=32, dropout=0.05) on all attention + FFN linear layers
- **Base Model**: Qwen2.5-0.5B-Instruct (float16)
- **Training**: 1 epoch, lr=2e-4, cosine schedule, warmup=3%, effective batch size=32
- **GPU**: NVIDIA T4 (16GB), ~10 minutes
- **Trainable params**: 8,798,208 / 323,917,696 (2.72%)

## Data Filtering

OPUS-100 contains noisy subtitle data. Applied filters:
- Sentence length: zh < 80 chars, en < 120 chars
- Removed dialogue markers (`--`, `...`)
- Kept only single-sentence samples (one period-ending sentence)

Without filtering, the fine-tuned model generated multi-sentence continuations.

## Key Findings

- LoRA fine-tuning improved BLEU by **+1.55** over the zero-shot baseline
- Data quality matters more than training epochs — 3 epochs caused overfitting (BLEU dropped)
- First-sentence truncation post-processing was sufficient for the OpenSubtitles noise

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "wangchao-nlp/qwen2.5-0.5b-zh-en-lora",
    torch_dtype=torch.float16,
    trust_remote_code=True,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("wangchao-nlp/qwen2.5-0.5b-zh-en-lora")
tokenizer.pad_token = tokenizer.eos_token

messages = [
    {"role": "system", "content": "你是一个专业的翻译助手。只输出英文翻译，不添加任何额外内容。"},
    {"role": "user", "content": "翻译成英文：人工智能正在改变世界。"},
]
formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.3, pad_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# Output: AI is changing the world.
```

## Files

- `train.ipynb` — Kaggle training notebook
- `requirements.txt` — Python dependencies
- `app.py` — Gradio demo (also on [HF Spaces](https://huggingface.co/spaces/wangchao-nlp/qwen-zh-en-translation))
