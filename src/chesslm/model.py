"""Load Gemma's tokenizer and its text-only model on whatever hardware is available."""

import torch
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
    return model if quantize_4bit else model.to(pick_device())
