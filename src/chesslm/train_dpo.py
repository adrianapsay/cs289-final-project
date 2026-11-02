"""Direct preference optimization (DPO), starting from the SFT model. Driven by configs/dpo.yaml.

The model sees a position (the same prompt used in evaluation) and learns to prefer
Stockfish's best move over a clearly worse one. The SFT model is the reference: DPO
rewards moving away from it only towards the preferred move.
"""

import time

import pandas as pd
import torch
from trl import DPOConfig, DPOTrainer

from chesslm.model import load_tokenizer, load_trainable_adapter, pick_device
from chesslm.preferences import to_dpo_dataset


def split_pairs(pairs, val_fraction, seed):
    """Split the preference pairs into train and validation datasets (by position; each position has one pair)."""
    val = pairs.sample(frac=val_fraction, random_state=seed)
    return to_dpo_dataset(pairs.drop(val.index)), to_dpo_dataset(val)


def train(train_ds, val_ds, dpo_cfg, model_cfg, lora_cfg, start_adapter, output_dir):
    """Train the SFT adapter further with DPO, save it to output_dir, and return (log, compute used)."""
    t = dpo_cfg["training"]
    device = pick_device()
    use_4bit = lora_cfg["quantize_4bit"] and device == "cuda"

    trainer = DPOTrainer(
        model=load_trainable_adapter(model_cfg, start_adapter, quantize_4bit=use_4bit),
        processing_class=load_tokenizer(model_cfg),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=DPOConfig(
            output_dir=str(output_dir),
            beta=t["beta"],
            num_train_epochs=t["epochs"],
            learning_rate=t["learning_rate"],
            lr_scheduler_type=t["lr_scheduler"],
            warmup_steps=t["warmup_steps"],
            per_device_train_batch_size=t["batch_size"],
            per_device_eval_batch_size=t["batch_size"],
            gradient_accumulation_steps=t["effective_batch_size"] // t["batch_size"],
            gradient_checkpointing=t["gradient_checkpointing"],
            max_length=t["max_seq_length"],
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
    trainer.save_model(str(output_dir))  # saves the trained adapter only, not the frozen reference copy

    data = trainer.train_dataset
    compute = {
        "device": torch.cuda.get_device_name() if device == "cuda" else device,
        "quantized_4bit": use_4bit,
        "wall_clock_minutes": round(minutes, 1),
        "gpu_hours": round(minutes / 60, 3),
        "pairs_seen": len(train_ds) * t["epochs"],
        # Each pair is two sequences: the prompt with the chosen move, and the prompt with the rejected move.
        "tokens_seen": sum(2 * len(p) + len(c) + len(r) for p, c, r in
                           zip(data["prompt_ids"], data["chosen_ids"], data["rejected_ids"])) * t["epochs"],
    }
    return pd.DataFrame(trainer.state.log_history), compute
