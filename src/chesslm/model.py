"""Load Gemma's tokenizer and its text-only model on whatever hardware is available."""

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer, Gemma4ForCausalLM
from transformers.utils import logging


def pick_device():
    """Use an NVIDIA GPU if there is one, then the Mac GPU, then the CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_tokenizer(model_cfg):
    return AutoTokenizer.from_pretrained(model_cfg["id"], revision=model_cfg["revision"])


def load_model(model_cfg, quantize_4bit=False):
    """Load only the language part of Gemma, optionally compressed to 4 bits (QLoRA, NVIDIA GPUs only).

    The checkpoint also contains image and audio encoders (~0.95 GB) that we never use.
    Its language weights are stored as 'model.language_model.*'; mapping them to the
    text model's 'model.*' loads them and skips everything else.
    """
    kwargs = {
        "revision": model_cfg["revision"],
        "config": AutoConfig.from_pretrained(model_cfg["id"], revision=model_cfg["revision"]).text_config,
        "dtype": torch.bfloat16,
        "key_mapping": {r"^model\.language_model\.": "model."},
    }
    if quantize_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
        )
        kwargs["device_map"] = {"": 0}

    # Hide the long list of skipped image/audio weights; they are skipped on purpose.
    previous = logging.get_verbosity()
    logging.set_verbosity_error()
    model = Gemma4ForCausalLM.from_pretrained(model_cfg["id"], **kwargs)
    logging.set_verbosity(previous)
    if quantize_4bit:
        keep_per_layer_embeddings_on_cpu(model)
        return model
    return model.to(pick_device())


class CpuEmbedding(torch.nn.Module):
    """A frozen, scaled embedding table that stays on the CPU and returns its rows on the input's device."""

    def __init__(self, embedding):
        super().__init__()
        # A plain tensor, not a parameter, so model.to("cuda") (e.g. by the trainer) leaves it on the CPU.
        self.table = embedding.weight.detach().to("cpu")
        self.scale = embedding.scalar_embed_scale
        self.padding_idx = embedding.padding_idx

    def forward(self, input_ids):
        rows = F.embedding(input_ids.cpu(), self.table, self.padding_idx).to(input_ids.device)
        return rows * torch.tensor(self.scale, dtype=rows.dtype, device=rows.device)


def keep_per_layer_embeddings_on_cpu(model):
    """Move Gemma's per-layer embedding table (2.35B parameters, 4.7 GB) from the GPU to the CPU.

    4-bit loading leaves this table unquantized, and on a 12 GB GPU it pushes training
    into slow shared memory. It is only a lookup table (each step reads a few rows), so
    looking it up on the CPU costs little. The weights and results are unchanged.
    """
    model.model.embed_tokens_per_layer = CpuEmbedding(model.model.embed_tokens_per_layer)
    torch.cuda.empty_cache()


def load_trainable_adapter(model_cfg, adapter_path, quantize_4bit=False):
    """Load the model with a trained LoRA adapter (e.g. the SFT model) whose weights keep training.

    DPO and RL start from the SFT model this way. TRL then keeps a frozen copy of the
    adapter as the reference model, so the reference is the SFT model, not the base model.
    """
    from peft import PeftModel
    model = load_model(model_cfg, quantize_4bit=quantize_4bit)
    return PeftModel.from_pretrained(model, str(adapter_path), is_trainable=True)
