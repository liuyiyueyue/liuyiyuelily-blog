---
title: "How to Use Hugging Face"
date: 2022-07-23
tags: ["llm", "hugging-face"]
---


Hugging Face is a higher-level library built on top of frameworks such as PyTorch, TensorFlow, and JAX. It provides rich pretrained models and fine-tuning support, with the goal of simplifying the development and use of Transformer models.
The three core parts of Hugging Face are: Transformers, Tokenizer, and Datasets.

### Transformers

Import the model library from `transformers`, for example `from transformers import AutoModelForCausalLM`
- Want representations → AutoModel
- Want generation / loss → AutoModelForCausalLM
- Want classification → AutoModelForSequenceClassification
- Want seq2seq → AutoModelForSeq2SeqLM

To specify which model to download and run from Hugging Face Hub, you need to provide the model repository name in your code, e.g. `model_id = "gpt2"`

### Tokenizer

A tokenizer splits text into tokens and maps them to input IDs. In practice, you import `AutoTokenizer`, load it with a model ID, tokenize a prompt, and pass the result to the model for generation.

### Examples

To run inference:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

prompt = "Once upon a time"
inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(**inputs, max_length=50)

print(tokenizer.decode(outputs[0]))
```

Or if you want to run training:
```python
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments
)

model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # GPT2 has no pad by default

model = AutoModelForCausalLM.from_pretrained(model_name)

# Dummy training data
texts = [
    "Hello, this is a test",
    "Hugging Face makes transformers easy",
    "We are training GPT-2"
]

dataset = Dataset.from_dict({"text": texts})

def preprocess(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=64)

dataset = dataset.map(preprocess, batched=True, remove_columns=["text"])

# labels = input_ids (causal LM)
dataset = dataset.map(lambda x: {"labels": x["input_ids"]})

training_args = TrainingArguments(
    output_dir="./gpt2-continued-pretrain",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_steps=1,
    save_steps=10,
    fp16=torch.cuda.is_available(),
)

trainer = Trainer(model=model, args=training_args, train_dataset=dataset)

trainer.train()
```
