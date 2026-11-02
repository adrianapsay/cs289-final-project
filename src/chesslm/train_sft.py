"""Supervised fine-tuning (SFT) with LoRA, driven by configs/sft.yaml.

The model sees a position (the same prompt used in evaluation) and learns to answer
with the move that was played. The loss is computed on the answer only, not the prompt.
"""

import time

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

from chesslm.model import load_model, load_tokenizer, pick_device
from chesslm.prompts import build_messages


def to_sft_example(row, prompt_cfg):
    """One training example: the question as the prompt, the move played as the completion."""
    answer = row[f"move_{prompt_cfg['notation']}"]
    return {
        "prompt": build_messages(row, prompt_cfg),
        "completion": [{"role": "assistant", "content": answer}],
    }


def make_dataset(positions, n, prompt_cfg, seed):
    """A seeded random sample of n positions, as SFT examples."""
    sample = positions.sample(n=min(n, len(positions)), random_state=seed)
    return Dataset.from_list([to_sft_example(row, prompt_cfg) for row in sample.to_dict("records")])


def train(train_ds, val_ds, sft_cfg, output_dir):
    """Train a LoRA adapter, save it to output_dir, and return (loss log, compute used)."""
    t = sft_cfg["training"]
    device = pick_device()
    use_4bit = sft_cfg["lora"]["quantize_4bit"] and device == "cuda"

    trainer = SFTTrainer(
        model=load_model(sft_cfg["model"], quantize_4bit=use_4bit),
        processing_class=load_tokenizer(sft_cfg["model"]),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=LoraConfig(
            r=sft_cfg["lora"]["rank"],
            lora_alpha=sft_cfg["lora"]["alpha"],
            lora_dropout=sft_cfg["lora"]["dropout"],
            target_modules=sft_cfg["lora"]["target_modules"],
            task_type="CAUSAL_LM",
        ),
        args=SFTConfig(
            output_dir=str(output_dir),
            num_train_epochs=t["epochs"],
            learning_rate=t["learning_rate"],
            lr_scheduler_type=t["lr_scheduler"],
            per_device_train_batch_size=t["batch_size"],
            per_device_eval_batch_size=t["batch_size"],
            gradient_accumulation_steps=t["effective_batch_size"] // t["batch_size"],
            gradient_checkpointing=t["gradient_checkpointing"],
            max_length=t["max_seq_length"],
            completion_only_loss=True,  # learn the answer, not the prompt
            bf16=device == "cuda",
            logging_steps=t["logging_steps"],
            eval_strategy="steps",
            eval_steps=t["eval_steps"],
            save_strategy="no",  # we save the final adapter ourselves
            report_to="none",
            seed=t["seed"],
        ),
    )

    start = time.time()
    trainer.train()
    minutes = (time.time() - start) / 60
    trainer.save_model(str(output_dir))  # saves the LoRA adapter only (a few MB), not the full model

    compute = {
        "device": torch.cuda.get_device_name() if device == "cuda" else device,
        "quantized_4bit": use_4bit,
        "wall_clock_minutes": round(minutes, 1),
        "gpu_hours": round(minutes / 60, 3),
        "examples_seen": len(train_ds) * t["epochs"],
        "tokens_seen": sum(len(ids) for ids in trainer.train_dataset["input_ids"]) * t["epochs"],
    }
    return pd.DataFrame(trainer.state.log_history), compute
