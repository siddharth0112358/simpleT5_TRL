# simpleT5-TRL

[![PyPI version](https://badge.fury.io/py/simplet5-trl)](https://badge.fury.io/py/simplet5-trl)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Train encoder-decoder models (T5, BART, T5Gemma, etc.) with **Full Finetuning**, **LoRA**, **QLoRA**, **DPO**, **SimPO** & **RFT** in just a few lines of code.

Built on top of HuggingFace Transformers and TRL.

**Supported models:** T5, MT5, ByT5, CodeT5, BART, mBART, Pegasus, LED, T5Gemma, and more!

---

## Installation

### From PyPI

```bash
pip install simplet5-trl
```

### From Source

```bash
git clone https://github.com/siddharth0112358/simpleT5_TRL.git
cd simpleT5_TRL
pip install -e .
```

---

## Quick Start

```python
from simplet5_trl import SimpleT5_TRL
import pandas as pd

# Prepare data
train_df = pd.DataFrame({
    "source_text": ["summarize: This is a long article..."],
    "target_text": ["Short summary"]
})

# Train
model = SimpleT5_TRL()
model.from_pretrained("t5-base")
model.train(train_df=train_df, eval_df=train_df, max_epochs=3)

# Predict
model.load_model("outputs/checkpoint-xxx", use_gpu=True)
print(model.predict("summarize: Your text here"))
```

---

## Training Methods

| Method | Function | Data Columns | Use When |
|--------|----------|--------------|----------|
| **SFT** | `train()` | `source_text`, `target_text` | Standard supervised finetuning |
| **DPO** | `train_dpo()` | `prompt`, `chosen`, `rejected` | You have preference pairs |
| **SimPO** | `train_simpo()` | `prompt`, `chosen`, `rejected` | Limited GPU memory (no ref model) |
| **RFT** | `train_rft()` | `source_text`, `target_text` | Curated high-quality samples |

### Choosing the Right Method

```
Have preference pairs (chosen vs rejected)?
├── Yes → Have enough GPU memory?
│         ├── Yes → DPO
│         └── No  → SimPO
└── No  → RFT (or standard train)
```

---

## Finetuning Modes

All training methods support three finetuning modes via `finetuning=` parameter:

| Mode | Memory | Quality | Use Case |
|------|--------|---------|----------|
| `"full"` | High | Best | Small models, enough VRAM |
| `"lora"` | Low | Good | Large models, limited VRAM |
| `"qlora"` | Very Low | Good | Very large models, consumer GPUs |

---

## Hyperparameters

<details>
<summary><strong>Common Parameters (All Methods)</strong></summary>

```python
# Training
max_epochs=3                    # Number of epochs
max_steps=-1                    # Max steps (-1 = use epochs)
batch_size=8                    # Batch size
learning_rate=1e-4              # Learning rate
precision="32"                  # "32", "16", "bf16"
seed=42                         # Random seed

# Optimizer
optim="adamw_torch"             # "adamw_torch", "sgd", "adafactor"
weight_decay=0.0                # Weight decay
warmup_steps=0                  # Warmup steps
warmup_ratio=0.0                # Warmup ratio
lr_scheduler_type="linear"      # "linear", "cosine", "constant", "polynomial"

# Gradient
gradient_accumulation_steps=1   # Gradient accumulation
gradient_checkpointing=False    # Memory-saving checkpointing

# Saving
outputdir="outputs"             # Output directory
save_strategy="epoch"           # "epoch", "steps", "no"
save_steps=500                  # Save every N steps
save_total_limit=None           # Max checkpoints to keep

# Evaluation
eval_strategy="epoch"           # "epoch", "steps", "no"
eval_steps=500                  # Eval every N steps

# Logging
logging_steps=1                 # Log every N steps
report_to=["tensorboard"]       # "tensorboard", "wandb"

# Finetuning
finetuning="full"               # "full", "lora", "qlora"
```

</details>

<details>
<summary><strong>LoRA Parameters</strong></summary>

```python
lora_r=16                       # LoRA rank
lora_alpha=32                   # LoRA alpha
lora_dropout=0.05               # LoRA dropout
lora_target_modules=None        # Auto-detected if None
```

</details>

<details>
<summary><strong>QLoRA Parameters</strong></summary>

```python
quantization="4bit"             # "4bit" or "8bit"
bnb_4bit_compute_dtype="float16"
bnb_4bit_quant_type="nf4"       # "nf4" or "fp4"
bnb_4bit_use_double_quant=True
```

</details>

<details>
<summary><strong>DPO Parameters</strong></summary>

```python
model.train_dpo(
    beta=0.1,                   # Deviation from reference (lower = more deviation)
    loss_type="sigmoid",        # "sigmoid", "hinge", "ipo", "robust"
    label_smoothing=0.0,        # Label smoothing
    max_length=512,             # Max sequence length
    max_prompt_length=256,      # Max prompt length
)
```

</details>

<details>
<summary><strong>SimPO Parameters</strong></summary>

```python
model.train_simpo(
    beta=2.0,                   # SimPO beta (higher than DPO)
    simpo_gamma=0.5,            # Target reward margin
    label_smoothing=0.0,
    max_length=512,
    max_prompt_length=256,
)
```

</details>

<details>
<summary><strong>RFT Parameters</strong></summary>

```python
model.train_rft(
    max_seq_length=512,         # Max sequence length
    packing=False,              # Pack multiple examples
    dataset_text_field="text",  # Column for text data
)
```

</details>

<details>
<summary><strong>Prediction Parameters</strong></summary>

```python
model.predict(
    source_text="input",
    max_length=512,
    num_beams=2,
    do_sample=True,
    temperature=1.0,
    top_k=50,
    top_p=0.95,
    repetition_penalty=2.5,
)
```

</details>

---

## Loading Models

```python
# Full finetuned
model.load_model("outputs/checkpoint-xxx", use_gpu=True)

# LoRA
model.load_model("outputs/checkpoint-xxx", finetuning="lora", base_model_name="t5-base")

# QLoRA
model.load_model("outputs/checkpoint-xxx", finetuning="qlora", base_model_name="t5-large", quantization="4bit")
```

---

## Supported Models

| Family | Examples |
|--------|----------|
| T5 | `t5-small`, `t5-base`, `t5-large`, `t5-3b` |
| MT5 | `google/mt5-small`, `google/mt5-base` |
| BART | `facebook/bart-base`, `facebook/bart-large` |
| CodeT5 | `Salesforce/codet5-base` |
| Pegasus | `google/pegasus-xsum` |
| LongT5 | `google/long-t5-local-base` |
| T5Gemma | `google/t5gemma-2-270m-270m`, `google/t5gemma-2b-2b-ul2` |

---

## Troubleshooting

<details>
<summary><strong>Missing embed_positions.weight Warning</strong></summary>

When loading BART or Pegasus models, you may see:
```
model.decoder.embed_positions.weight | MISSING
model.encoder.embed_positions.weight | MISSING
```
**This is expected and harmless.** These models use sinusoidal positional embeddings computed at runtime, not learned weights.

</details>

<details>
<summary><strong>LongT5-tglobal NaN Issues</strong></summary>

The `google/long-t5-tglobal-*` models may produce NaN values during training. Use the local variant instead:
```python
model.from_pretrained("google/long-t5-local-base")  # Stable
# instead of "google/long-t5-tglobal-base"          # May produce NaN
```

</details>

---

## Acknowledgements

- [simpleT5](https://github.com/Shivanandroy/simpleT5) by Shivanandroy
- [Transformers](https://huggingface.co/transformers/) by HuggingFace
- [TRL](https://github.com/huggingface/trl) by HuggingFace
- [PEFT](https://github.com/huggingface/peft) by HuggingFace
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)

---

## License

MIT License
