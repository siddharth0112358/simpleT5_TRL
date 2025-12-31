# simpleT5_TRL Examples

Comprehensive examples for all training methods supported by simpleT5_TRL.

## Table of Contents

- [Standard Supervised Fine-Tuning (SFT)](#standard-supervised-fine-tuning-sft)
  - [Full Finetuning](#1-full-finetuning)
  - [LoRA Finetuning](#2-lora-finetuning)
  - [QLoRA Finetuning](#3-qlora-finetuning)
- [DPO (Direct Preference Optimization)](#dpo-direct-preference-optimization)
  - [Basic DPO](#basic-dpo-training)
  - [DPO with LoRA](#dpo-with-lora)
  - [DPO with Different Loss Types](#dpo-with-different-loss-types)
- [SimPO (Simple Preference Optimization)](#simpo-simple-preference-optimization)
  - [Basic SimPO](#basic-simpo-training)
  - [SimPO with QLoRA](#simpo-with-qlora)
- [RFT (Reinforcement Fine-Tuning)](#rft-reinforcement-fine-tuning)
  - [Basic RFT](#basic-rft-training)
  - [RFT with Text Completion Format](#rft-with-text-completion-format)
- [Complete SFT + RL Pipelines](#complete-sft--rl-pipelines)
  - [SFT then DPO Pipeline](#sft-then-dpo-pipeline)
- [Advanced Examples](#advanced-examples)
  - [Resume Training](#resume-training-from-checkpoint)
  - [Merge LoRA Weights](#merge-lora-weights)
  - [Training Large Models](#training-large-models-on-consumer-gpus)
  - [Weights & Biases Logging](#weights--biases-logging)
  - [Complete Pipeline](#complete-summarization-pipeline)

---

## Standard Supervised Fine-Tuning (SFT)

### 1. Full Finetuning

Train all model parameters. Best quality but requires more GPU memory.

```python
from simplet5_trl import SimpleT5_TRL
import pandas as pd

# Prepare data with source_text and target_text columns
train_df = pd.DataFrame({
    "source_text": [
        "summarize: The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet.",
        "summarize: Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "translate English to French: Hello, how are you today?",
        "translate English to French: The weather is beautiful.",
    ],
    "target_text": [
        "A sentence using all alphabet letters.",
        "ML enables systems to learn from data.",
        "Bonjour, comment allez-vous aujourd'hui?",
        "Le temps est magnifique.",
    ]
})

eval_df = pd.DataFrame({
    "source_text": ["summarize: Python is a popular programming language."],
    "target_text": ["Python is a popular language."]
})

# Initialize and load model
model = SimpleT5_TRL()
model.from_pretrained("t5-base")

# Train with full finetuning
model.train(
    train_df=train_df,
    eval_df=eval_df,
    source_max_token_len=512,
    target_max_token_len=128,
    batch_size=8,
    max_epochs=5,
    learning_rate=1e-4,
    outputdir="outputs/full_finetuning",
    save_strategy="epoch",
    save_total_limit=2,
)

# Load and predict
model.load_model("outputs/full_finetuning/checkpoint-xxx", use_gpu=True)
predictions = model.predict("summarize: Your text here", max_length=128, num_beams=4)
print(predictions[0])
```

### 2. LoRA Finetuning

Parameter-efficient finetuning that only trains small adapter layers.

```python
from simplet5_trl import SimpleT5_TRL
import pandas as pd

train_df = pd.DataFrame({
    "source_text": ["summarize: " + text for text in your_texts],
    "target_text": your_summaries
})

model = SimpleT5_TRL()
model.from_pretrained("t5-base")

# Train with LoRA
model.train(
    train_df=train_df,
    eval_df=eval_df,
    finetuning="lora",
    lora_r=16,              # LoRA rank (higher = more parameters)
    lora_alpha=32,          # LoRA scaling factor
    lora_dropout=0.05,      # Dropout for LoRA layers
    batch_size=8,
    max_epochs=5,
    learning_rate=1e-4,
    outputdir="outputs/lora_finetuning",
)

# Load LoRA model - requires base model name
model = SimpleT5_TRL()
model.load_model(
    "outputs/lora_finetuning/checkpoint-xxx",
    finetuning="lora",
    base_model_name="t5-base",
    use_gpu=True
)
print(model.predict("summarize: Your text"))
```

### 3. QLoRA Finetuning

Train large models on consumer GPUs using 4-bit/8-bit quantization.

```python
from simplet5_trl import SimpleT5_TRL
import pandas as pd

model = SimpleT5_TRL()
model.from_pretrained("t5-large")  # Can use larger models with QLoRA

# Train with QLoRA (4-bit quantization)
model.train(
    train_df=train_df,
    eval_df=eval_df,
    finetuning="qlora",
    quantization="4bit",              # or "8bit"
    bnb_4bit_compute_dtype="float16", # Compute dtype
    bnb_4bit_quant_type="nf4",        # "nf4" or "fp4"
    bnb_4bit_use_double_quant=True,   # Double quantization for more savings
    lora_r=16,
    lora_alpha=32,
    batch_size=4,                     # Smaller batch size for large models
    gradient_accumulation_steps=4,    # Effective batch = 16
    gradient_checkpointing=True,      # Save more memory
    max_epochs=3,
    learning_rate=2e-4,
    outputdir="outputs/qlora_finetuning",
)

# Load QLoRA model
model = SimpleT5_TRL()
model.load_model(
    "outputs/qlora_finetuning/checkpoint-xxx",
    finetuning="qlora",
    base_model_name="t5-large",
    quantization="4bit",
    use_gpu=True
)
```

---

## DPO (Direct Preference Optimization)

DPO trains models to prefer chosen responses over rejected ones without a separate reward model.

### Basic DPO Training

```python
from simplet5_trl import SimpleT5_TRL
import pandas as pd

# DPO requires: prompt, chosen (preferred), rejected (not preferred)
train_df = pd.DataFrame({
    "prompt": [
        "Summarize: The quick brown fox jumps over the lazy dog.",
        "Translate to French: Good morning!",
        "Explain: What is machine learning?",
        "Summarize: Python is a versatile programming language.",
    ],
    "chosen": [
        "A fox jumps over a dog.",
        "Bonjour!",
        "Machine learning is AI that learns from data to make predictions.",
        "Python is versatile.",
    ],
    "rejected": [
        "Quick fox.",                    # Too short
        "Buenos dias!",                  # Wrong language
        "It's complicated.",             # Unhelpful
        "Programming is hard.",          # Off-topic
    ]
})

eval_df = pd.DataFrame({
    "prompt": ["Summarize: The weather is nice today."],
    "chosen": ["Nice weather today."],
    "rejected": ["Weather."]
})

model = SimpleT5_TRL()
model.from_pretrained("t5-base")

# Train with DPO
model.train_dpo(
    train_df=train_df,
    eval_df=eval_df,
    beta=0.1,                    # Lower = more deviation from reference allowed
    loss_type="sigmoid",         # Standard DPO loss
    max_length=512,
    max_prompt_length=256,
    batch_size=4,
    max_epochs=3,
    learning_rate=5e-7,          # DPO typically uses lower learning rates
    outputdir="outputs/dpo",
)
```

### DPO with LoRA

```python
model = SimpleT5_TRL()
model.from_pretrained("t5-base")

model.train_dpo(
    train_df=train_df,
    eval_df=eval_df,
    beta=0.1,
    finetuning="lora",           # Use LoRA for memory efficiency
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    batch_size=4,
    max_epochs=3,
    learning_rate=5e-7,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    outputdir="outputs/dpo_lora",
)

# Load
model.load_model(
    "outputs/dpo_lora/checkpoint-xxx",
    finetuning="lora",
    base_model_name="t5-base",
    use_gpu=True
)
```

### DPO with Different Loss Types

```python
# IPO (Identity Preference Optimization) - more robust
model.train_dpo(
    train_df=train_df,
    beta=0.1,
    loss_type="ipo",
    # ... other params
)

# Hinge loss - margin-based
model.train_dpo(
    train_df=train_df,
    beta=0.1,
    loss_type="hinge",
    # ... other params
)

# Robust DPO - handles noisy preferences
model.train_dpo(
    train_df=train_df,
    beta=0.1,
    loss_type="robust",
    label_smoothing=0.1,         # Add smoothing for robustness
    # ... other params
)
```

---

## SimPO (Simple Preference Optimization)

SimPO is a memory-efficient variant of DPO that doesn't require a reference model.

### Basic SimPO Training

```python
from simplet5_trl import SimpleT5_TRL
import pandas as pd

# SimPO uses same data format as DPO
train_df = pd.DataFrame({
    "prompt": [
        "Summarize: Long article about climate change...",
        "Translate: Hello world",
    ],
    "chosen": [
        "Climate change requires immediate action.",
        "Bonjour le monde",
    ],
    "rejected": [
        "Weather is changing.",
        "Hello world",
    ]
})

model = SimpleT5_TRL()
model.from_pretrained("t5-base")

model.train_simpo(
    train_df=train_df,
    eval_df=eval_df,
    beta=2.0,                    # SimPO uses higher beta than DPO
    simpo_gamma=0.5,             # Target reward margin
    label_smoothing=0.0,
    max_length=512,
    max_prompt_length=256,
    batch_size=4,
    max_epochs=3,
    learning_rate=5e-7,
    finetuning="lora",
    outputdir="outputs/simpo",
)
```

### SimPO with QLoRA

```python
# SimPO is already memory-efficient, QLoRA makes it even more so
model = SimpleT5_TRL()
model.from_pretrained("t5-large")

model.train_simpo(
    train_df=train_df,
    beta=2.0,
    simpo_gamma=0.5,
    finetuning="qlora",
    quantization="4bit",
    lora_r=16,
    lora_alpha=32,
    batch_size=2,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    max_epochs=3,
    outputdir="outputs/simpo_qlora",
)
```

---

## RFT (Reinforcement Fine-Tuning)

RFT is supervised fine-tuning on high-quality curated samples.

### Basic RFT Training

```python
from simplet5_trl import SimpleT5_TRL
import pandas as pd

# RFT uses same format as standard training
# The key is that your data should be high-quality, curated samples
train_df = pd.DataFrame({
    "source_text": [
        "summarize: [High quality article text...]",
        "translate English to French: Hello, how are you?",
    ],
    "target_text": [
        "[Expert-written summary]",
        "Bonjour, comment allez-vous?",
    ]
})

model = SimpleT5_TRL()
model.from_pretrained("t5-base")

model.train_rft(
    train_df=train_df,
    eval_df=eval_df,
    max_seq_length=512,
    batch_size=8,
    max_epochs=3,
    learning_rate=2e-5,
    finetuning="lora",
    lora_r=16,
    outputdir="outputs/rft",
)
```

### RFT with Text Completion Format

```python
# Alternative format using single "text" column
train_df = pd.DataFrame({
    "text": [
        "Human: What is machine learning?\nAssistant: Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed.",
        "Human: Explain Python in one sentence.\nAssistant: Python is a versatile, readable programming language used for web development, data science, and automation.",
    ]
})

model = SimpleT5_TRL()
model.from_pretrained("t5-base")

model.train_rft(
    train_df=train_df,
    dataset_text_field="text",   # Specify the column name
    packing=True,                # Pack multiple short examples together
    max_seq_length=512,
    batch_size=4,
    max_epochs=3,
    finetuning="lora",
    outputdir="outputs/rft_completion",
)
```

---

## Complete SFT + RL Pipelines

The recommended approach for training with reinforcement learning is to first perform supervised fine-tuning (SFT) on your task, then apply RL techniques to further improve the model. This two-stage approach produces better results than applying RL directly to a pretrained model.

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Pretrained     │     │   SFT Model     │     │   RL-Aligned    │
│  Model (T5)     │ ──► │  (Task-tuned)   │ ──► │     Model       │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                              │                        │
                         train()                 train_dpo()
                         train_rft()             train_simpo()
```

### SFT then DPO Pipeline

The most common pipeline: SFT for task adaptation, then DPO for preference alignment.

```python
from simplet5_trl import SimpleT5_TRL
import pandas as pd

# ============ Stage 1: Supervised Fine-Tuning ============
print("Stage 1: Supervised Fine-Tuning")

# Load SFT data
sft_train = pd.read_csv("data/sft_train.csv")
sft_val = pd.read_csv("data/sft_val.csv")

# Initialize model
model = SimpleT5_TRL()
model.from_pretrained("t5-base")

# Train with SFT
model.train(
    train_df=sft_train,
    eval_df=sft_val,
    source_max_token_len=512,
    target_max_token_len=128,
    batch_size=8,
    max_epochs=3,
    learning_rate=1e-4,
    finetuning="lora",
    lora_r=16,
    lora_alpha=32,
    outputdir="outputs/sft_model",
    save_strategy="epoch",
)

# ============ Stage 2: DPO Alignment ============
print("Stage 2: DPO Preference Alignment")

# Load preference data
dpo_train = pd.read_csv("data/dpo_train.csv")
dpo_val = pd.read_csv("data/dpo_val.csv")

# Load the SFT checkpoint for DPO training
model = SimpleT5_TRL()
model.load_model(
    "outputs/sft_model/checkpoint-best",
    finetuning="lora",
    base_model_name="t5-base",
    use_gpu=True
)

# Merge LoRA weights before DPO (recommended)
model.merge_lora_weights()

# Train with DPO
model.train_dpo(
    train_df=dpo_train,
    eval_df=dpo_val,
    beta=0.1,
    max_length=512,
    max_prompt_length=256,
    batch_size=4,
    max_epochs=2,
    learning_rate=5e-7,           # Lower LR for DPO
    finetuning="lora",            # Apply new LoRA for DPO
    lora_r=8,
    outputdir="outputs/dpo_model",
)

# ============ Stage 3: Inference ============
print("Stage 3: Inference with aligned model")

model = SimpleT5_TRL()
model.load_model(
    "outputs/dpo_model/checkpoint-best",
    finetuning="lora",
    base_model_name="t5-base",
    use_gpu=True
)

result = model.predict("summarize: Your text here", max_length=128)
print(f"Result: {result[0]}")
```

### Quick Reference: SFT + RL Combinations

| Pipeline | When to Use | Data Required |
|----------|-------------|---------------|
| **SFT → DPO** | Have preference pairs, simple setup | `source_text/target_text` + `prompt/chosen/rejected` |
| **SFT → SimPO** | Limited GPU memory, no reference model | `source_text/target_text` + `prompt/chosen/rejected` |

### Tips for SFT + RL Pipelines

1. **Always start with SFT**: RL works better when the model already understands the task
2. **Merge LoRA weights** between stages for cleaner training
3. **Use lower learning rates** for RL (5e-7 to 1e-5) compared to SFT (1e-4)
4. **Monitor KL divergence** to prevent the model from deviating too far from SFT
5. **Start with small beta** (0.1 for DPO) and increase if needed
6. **Quality of preference/reward data** matters more than quantity

---

## Advanced Examples

### Resume Training from Checkpoint

```python
model = SimpleT5_TRL()
model.from_pretrained("t5-base")

# Resume training from a specific checkpoint
model.train(
    train_df=train_df,
    eval_df=eval_df,
    resume_from_checkpoint="outputs/checkpoint-500",
    max_epochs=10,              # Continue to 10 epochs total
    outputdir="outputs",
)
```

### Merge LoRA Weights

Merge LoRA adapters into base model for faster inference:

```python
model = SimpleT5_TRL()
model.load_model(
    "outputs/lora_model/checkpoint-xxx",
    finetuning="lora",
    base_model_name="t5-base",
    use_gpu=True
)

# Merge LoRA weights into base model
model.merge_lora_weights()

# Save merged model - no longer needs PEFT for inference
model.save_merged_model("outputs/merged_model")

# Load merged model simply
model = SimpleT5_TRL()
model.load_model("outputs/merged_model", use_gpu=True)
print(model.predict("Your input"))
```

### Training Large Models on Consumer GPUs

```python
from simplet5_trl import SimpleT5_TRL

model = SimpleT5_TRL()
model.from_pretrained("t5-3b")  # 3 billion parameter model

# Use QLoRA + gradient checkpointing + small batch
model.train(
    train_df=train_df,
    eval_df=eval_df,
    finetuning="qlora",
    quantization="4bit",
    lora_r=8,                    # Lower rank for large models
    lora_alpha=16,
    batch_size=1,                # Very small batch
    gradient_accumulation_steps=16,  # Effective batch = 16
    gradient_checkpointing=True,
    precision="bf16",            # Use bfloat16 if supported
    max_epochs=3,
    learning_rate=1e-4,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    outputdir="outputs/t5_3b_qlora",
)
```

### Weights & Biases Logging

```python
import wandb
wandb.login()

model = SimpleT5_TRL()
model.from_pretrained("t5-base")

model.train(
    train_df=train_df,
    eval_df=eval_df,
    report_to=["wandb"],
    logging_steps=10,
    max_epochs=5,
    # WandB will auto-create a run and log metrics
)
```

### Complete Summarization Pipeline

```python
from simplet5_trl import SimpleT5_TRL
import pandas as pd

# ============ 1. Load and Prepare Data ============
train_df = pd.read_csv("train.csv")  # columns: article, summary
eval_df = pd.read_csv("eval.csv")

# Add task prefix for T5
train_df["source_text"] = "summarize: " + train_df["article"]
train_df["target_text"] = train_df["summary"]
eval_df["source_text"] = "summarize: " + eval_df["article"]
eval_df["target_text"] = eval_df["summary"]

# ============ 2. Initialize Model ============
model = SimpleT5_TRL()
model.from_pretrained("t5-base")

# ============ 3. Train with LoRA ============
model.train(
    train_df=train_df[["source_text", "target_text"]],
    eval_df=eval_df[["source_text", "target_text"]],
    source_max_token_len=512,
    target_max_token_len=128,
    batch_size=8,
    max_epochs=5,
    finetuning="lora",
    lora_r=16,
    lora_alpha=32,
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    save_strategy="epoch",
    save_total_limit=2,
    eval_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    outputdir="summarization_model",
)

# ============ 4. Load Best Model ============
model = SimpleT5_TRL()
model.load_model(
    "summarization_model/checkpoint-xxx",
    finetuning="lora",
    base_model_name="t5-base",
    use_gpu=True
)

# ============ 5. Generate Summaries ============
def summarize(text, max_length=128):
    return model.predict(
        f"summarize: {text}",
        max_length=max_length,
        num_beams=4,
        do_sample=False,
        early_stopping=True,
    )[0]

# Test
article = """
Your long article text goes here. It can be multiple paragraphs
and contain detailed information that needs to be summarized.
The model will generate a concise summary of this content.
"""

summary = summarize(article)
print(f"Summary: {summary}")

# ============ 6. Batch Inference ============
test_articles = ["Article 1...", "Article 2...", "Article 3..."]
summaries = [summarize(article) for article in test_articles]
for i, (article, summary) in enumerate(zip(test_articles, summaries)):
    print(f"\n--- Article {i+1} ---")
    print(f"Summary: {summary}")
```

### Training T5Gemma

```python
from simplet5_trl import SimpleT5_TRL

model = SimpleT5_TRL()

# T5Gemma requires authentication for gated models
# Available variants:
#   - google/t5gemma-2-270m-270m (0.8B params - fits on consumer GPUs)
#   - google/t5gemma-2b-2b-ul2 (larger model - requires more VRAM)
model.from_pretrained(
    "google/t5gemma-2-270m-270m",  # or "google/t5gemma-2b-2b-ul2"
    use_auth_token="your_hf_token",
    trust_remote_code=True
)

# Train with LoRA (270m variant fits on consumer GPUs without quantization)
model.train(
    train_df=train_df,
    eval_df=eval_df,
    finetuning="lora",
    lora_r=16,
    lora_alpha=32,
    batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    max_epochs=3,
    precision="bf16",
    outputdir="outputs/t5gemma",
)

# For larger variants (2b-2b), use QLoRA:
# model.train(
#     train_df=train_df,
#     finetuning="qlora",
#     quantization="4bit",
#     gradient_checkpointing=True,
#     batch_size=2,
#     ...
# )
```

### Early Stopping

```python
model.train(
    train_df=train_df,
    eval_df=eval_df,
    max_epochs=20,
    early_stopping_patience_epochs=3,  # Stop if no improvement for 3 evals
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,           # Lower loss is better
    eval_strategy="epoch",
    save_strategy="epoch",
)
```

### Custom Learning Rate Schedule

```python
# Cosine with warmup
model.train(
    train_df=train_df,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,            # 10% of steps for warmup
    learning_rate=1e-4,
    # ...
)

# Linear decay with fixed warmup steps
model.train(
    train_df=train_df,
    lr_scheduler_type="linear",
    warmup_steps=500,            # Fixed 500 warmup steps
    learning_rate=1e-4,
    # ...
)

# Constant learning rate with warmup
model.train(
    train_df=train_df,
    lr_scheduler_type="constant_with_warmup",
    warmup_steps=100,
    learning_rate=5e-5,
    # ...
)
```

---

## Tips and Best Practices

### Choosing Learning Rates

| Method | Recommended LR |
|--------|----------------|
| SFT (Full) | 1e-4 to 5e-5 |
| SFT (LoRA) | 1e-4 to 2e-4 |
| DPO | 5e-7 to 1e-6 |
| SimPO | 5e-7 to 1e-6 |

### Memory Optimization

1. **Use LoRA/QLoRA** for large models
2. **Enable gradient checkpointing** (`gradient_checkpointing=True`)
3. **Reduce batch size** and increase `gradient_accumulation_steps`
4. **Use mixed precision** (`precision="16"` or `precision="bf16"`)

### Data Quality Tips

- **DPO/SimPO**: Ensure chosen responses are clearly better than rejected
- **RFT**: Use only high-quality, verified examples
