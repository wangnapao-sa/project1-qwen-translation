"""
============================================================
项目一：基于 LoRA 的 Qwen2.5 中英翻译微调
============================================================
在 Kaggle Notebook 中按 Cell 顺序运行（T4 GPU, Internet On）
"""

# ============================================================
# Cell 1: 安装依赖
# ============================================================
!pip install -q transformers datasets peft accelerate sacrebleu gradio huggingface_hub

# ============================================================
# Cell 2: 加载模型 + LoRA 配置
# ============================================================
import torch, os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    trust_remote_code=True,
).to("cuda")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.enable_input_require_grads()

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"可训练参数: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

# ============================================================
# Cell 3: 加载数据 + 三道过滤
# ============================================================
from datasets import load_dataset

dataset = load_dataset("Helsinki-NLP/opus-100", "en-zh")

# 过滤 1: 长度限制
dataset = dataset.filter(
    lambda x: len(x['translation']['zh']) < 80 and len(x['translation']['en']) < 120
)

# 过滤 2+3: 去对话标记 + 单句筛选
def is_clean(example):
    en = example['translation']['en'].strip()
    zh = example['translation']['zh'].strip()
    # 去对话标记
    if '--' in en or '...' in en or '--' in zh:
        return False
    # 单句筛选: 英文必须收尾于标点且不超过 1 个句号
    if not en.endswith(('.', '!', '?', '"', "'")):
        return False
    if en.rstrip('"').rstrip("'").count('.') > 1:
        return False
    return True

dataset = dataset.filter(is_clean)
print(f"过滤后训练集: {len(dataset['train'])} 条")

# ============================================================
# Cell 4: 格式化为 Chat Template
# ============================================================
SYSTEM_PROMPT = "你是一个专业的翻译助手。你只输出英文翻译，不添加任何解释、评论或额外内容。"

def format_translation(example):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"翻译成英文：{example['translation']['zh']}"},
        {"role": "assistant", "content": example['translation']['en']},
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}

train_raw = dataset['train'].select(range(20000)).map(format_translation)

def tokenize_fn(examples):
    return tokenizer(examples["text"], truncation=True, max_length=200, padding=False)

train_data = train_raw.map(tokenize_fn, batched=True, remove_columns=train_raw.column_names)
print(f"训练样本数: {len(train_data)}")

lengths = [len(x['input_ids']) for x in train_data]
print(f"平均长度: {sum(lengths)/len(lengths):.0f} tokens")

# ============================================================
# Cell 5: 训练（单卡 T4, ~10 分钟）
# ============================================================
from transformers import DataCollatorForLanguageModeling, TrainingArguments, Trainer

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="/kaggle/working/qwen-translation-lora",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    optim="adamw_torch",
    save_strategy="epoch",
    save_total_limit=1,
    logging_steps=50,
    report_to="none",
    fp16=True,
    dataloader_num_workers=2,
    remove_unused_columns=True,
)

trainer = Trainer(model=model, args=training_args, train_dataset=train_data, data_collator=data_collator)
print("开始训练…")
trainer.train()
trainer.save_model("/kaggle/working/qwen-translation-lora-final")
tokenizer.save_pretrained("/kaggle/working/qwen-translation-lora-final")
print("训练完成")

# ============================================================
# Cell 6: 加载微调模型 + 测试翻译
# ============================================================
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    torch_dtype=torch.float16,
    trust_remote_code=True,
).to("cuda")

model_ft = PeftModel.from_pretrained(base_model, "/kaggle/working/qwen-translation-lora-final")
model_ft = model_ft.merge_and_unload()
tokenizer_ft = AutoTokenizer.from_pretrained("/kaggle/working/qwen-translation-lora-final", trust_remote_code=True)
tokenizer_ft.pad_token = tokenizer_ft.eos_token

def translate(text):
    messages = [
        {"role": "system", "content": "你是一个专业的翻译助手。你只输出英文翻译，不添加任何解释、评论或额外内容。"},
        {"role": "user", "content": f"翻译成英文：{text}"},
    ]
    formatted = tokenizer_ft.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer_ft(formatted, return_tensors="pt").to(model_ft.device)
    with torch.no_grad():
        outputs = model_ft.generate(
            **inputs, max_new_tokens=100, do_sample=True,
            temperature=0.3, top_p=0.9,
            pad_token_id=tokenizer_ft.pad_token_id,
            eos_token_id=tokenizer_ft.eos_token_id,
        )
    response = tokenizer_ft.decode(outputs[0], skip_special_tokens=True)
    if "assistant" in response:
        response = response.split("assistant")[-1].strip()
    # 只取第一句
    for sep in ['. ', '.\n', '!\n', '?\n', '.']:
        idx = response.find(sep)
        if 3 < idx < 200:
            response = response[:idx+1]
            break
    return response

for zh in ["人工智能正在改变世界。", "今天天气真好。", "这本书很有趣。"]:
    print(f"中文: {zh}")
    print(f"翻译: {translate(zh)}")
    print("---")

# ============================================================
# Cell 7: 基线 BLEU 评测（原始模型，不做任何修改）
# ============================================================
print("评测基线模型…")

model_base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    torch_dtype=torch.float16,
    trust_remote_code=True,
).to("cuda")
tokenizer_base = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
tokenizer_base.pad_token = tokenizer_base.eos_token

def translate_base(text):
    messages = [
        {"role": "system", "content": "你是一个专业的翻译助手。你只输出英文翻译，不添加任何解释、评论或额外内容。"},
        {"role": "user", "content": f"翻译成英文：{text}"},
    ]
    formatted = tokenizer_base.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer_base(formatted, return_tensors="pt").to(model_base.device)
    with torch.no_grad():
        outputs = model_base.generate(
            **inputs, max_new_tokens=100, do_sample=True,
            temperature=0.3, top_p=0.9,
            pad_token_id=tokenizer_base.pad_token_id,
            eos_token_id=tokenizer_base.eos_token_id,
        )
    response = tokenizer_base.decode(outputs[0], skip_special_tokens=True)
    if "assistant" in response:
        response = response.split("assistant")[-1].strip()
    for sep in ['. ', '.\n', '!\n', '?\n', '.']:
        idx = response.find(sep)
        if 3 < idx < 200:
            response = response[:idx+1]
            break
    return response

from datasets import load_dataset
from sacrebleu.metrics import BLEU

dataset = load_dataset("Helsinki-NLP/opus-100", "en-zh")
test_data_200 = dataset['test'].select(range(200))

preds_base, refs_base = [], []
for i, sample in enumerate(test_data_200):
    refs_base.append(sample['translation']['en'])
    preds_base.append(translate_base(sample['translation']['zh']))
    if (i+1) % 50 == 0:
        print(f"  {i+1}/200")

bleu_base = BLEU().corpus_score(preds_base, [refs_base])
print(f"\n基线 BLEU: {bleu_base.score:.2f}")

# ============================================================
# Cell 8: 微调模型 BLEU 评测
# ============================================================
print("评测微调模型…")

preds_ft, refs_ft = [], []
for i, sample in enumerate(test_data_200):
    refs_ft.append(sample['translation']['en'])
    preds_ft.append(translate(sample['translation']['zh']))
    if (i+1) % 50 == 0:
        print(f"  {i+1}/200")

bleu_ft = BLEU().corpus_score(preds_ft, [refs_ft])
print(f"\n微调后 BLEU: {bleu_ft.score:.2f}")
print(f"BLEU 提升: +{bleu_ft.score - bleu_base.score:.2f}")

# ============================================================
# Cell 9: Gradio 演示
# ============================================================
import gradio as gr

demo = gr.Interface(
    fn=translate,
    inputs=gr.Textbox(lines=5, placeholder="输入中文，按回车翻译", label="中文"),
    outputs=gr.Textbox(lines=5, label="英文翻译"),
    title="中英翻译 - Qwen2.5-0.5B + LoRA",
    description=f"LoRA (rank=16) 微调 | BLEU: {bleu_ft.score:.2f} (+{bleu_ft.score - bleu_base.score:.2f})",
    examples=[
        ["人工智能正在改变世界。"],
        ["今天天气真好。"],
        ["这本书很有趣。"],
        ["深度学习需要大量计算资源。"],
        ["他对这个问题有着独到的见解。"],
    ],
    theme="soft",
)

demo.launch(share=True)
# 记下输出里的 .gradio.live 链接！

# ============================================================
# Cell 10: 上传到 HuggingFace Hub
# ============================================================
from huggingface_hub import notebook_login
notebook_login()

# 重新加载并 merge
base_for_push = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    torch_dtype=torch.float16,
    trust_remote_code=True,
).to("cuda")
model_for_push = PeftModel.from_pretrained(base_for_push, "/kaggle/working/qwen-translation-lora-final")
model_for_push = model_for_push.merge_and_unload()

tokenizer_for_push = AutoTokenizer.from_pretrained("/kaggle/working/qwen-translation-lora-final", trust_remote_code=True)

REPO_NAME = "wangchao-nlp/qwen2.5-0.5b-zh-en-lora"

model_for_push.push_to_hub(REPO_NAME)
tokenizer_for_push.push_to_hub(REPO_NAME)

print(f"上传完成！https://huggingface.co/{REPO_NAME}")
