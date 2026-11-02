"""Reinforcement learning with GRPO, starting from the SFT model. Driven by configs/rl.yaml.

For each position the model samples several answers, Stockfish scores each one
(see rewards.py), and the model is pushed towards the answers that scored above
the group's average. No example answers are needed.
"""

import time

import pandas as pd
import torch
from trl import GRPOConfig, GRPOTrainer

from chesslm.model import load_tokenizer, load_trainable_adapter, pick_device
from chesslm.rewards import make_reward_function


def train(train_ds, rl_cfg, stockfish_cfg, model_cfg, lora_cfg, start_adapter, output_dir):
    """Train the SFT adapter further with GRPO, save it to output_dir, and return (log, compute used)."""
    t = rl_cfg["training"]
    device = pick_device()
    use_4bit = lora_cfg["quantize_4bit"] and device == "cuda"
    reward_fn = make_reward_function(rl_cfg["reward"], stockfish_cfg)

    trainer = GRPOTrainer(
        model=load_trainable_adapter(model_cfg, start_adapter, quantize_4bit=use_4bit),
        processing_class=load_tokenizer(model_cfg),
        reward_funcs=reward_fn,
        train_dataset=train_ds,
        args=GRPOConfig(
            output_dir=str(output_dir),
            num_generations=t["num_generations"],
            max_completion_length=t["max_completion_length"],
            temperature=t["temperature"],
            beta=t["beta"],
            num_train_epochs=1,  # each position is used once
            learning_rate=t["learning_rate"],
            lr_scheduler_type=t["lr_scheduler"],
            per_device_train_batch_size=t["batch_size"],
            gradient_accumulation_steps=t["effective_batch_size"] // t["batch_size"],
            gradient_checkpointing=t["gradient_checkpointing"],
            bf16=device == "cuda",
            logging_steps=t["logging_steps"],
            save_strategy="no",  # we save the final adapter ourselves
            report_to="none",
            seed=t["seed"],
        ),
    )

    start = time.time()
    try:
        trainer.train()
    finally:
        reward_fn.scorer.engine.quit()
    minutes = (time.time() - start) / 60
    trainer.save_model(str(output_dir))  # saves the LoRA adapter only

    compute = {
        "device": torch.cuda.get_device_name() if device == "cuda" else device,
        "quantized_4bit": use_4bit,
        "wall_clock_minutes": round(minutes, 1),
        "gpu_hours": round(minutes / 60, 3),
        "positions_seen": len(train_ds),
        "answers_generated": len(train_ds) * t["num_generations"],
        "tokens_seen": int(trainer.state.num_input_tokens_seen),  # prompts plus generated answers
    }
    return pd.DataFrame(trainer.state.log_history), compute
