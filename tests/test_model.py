import pandas as pd
import torch
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextScaledWordEmbedding

from chesslm.model import CpuEmbedding
from chesslm.train_dpo import split_pairs


def test_cpu_embedding_matches_the_original_table():
    table = Gemma4TextScaledWordEmbedding(50, 8, padding_idx=0, embed_scale=4.0).to(torch.bfloat16)
    ids = torch.tensor([[0, 3, 49], [7, 7, 1]])
    assert torch.equal(CpuEmbedding(table)(ids), table(ids))


def test_cpu_embedding_stays_on_the_cpu_when_the_model_is_moved():
    model = torch.nn.Sequential(CpuEmbedding(Gemma4TextScaledWordEmbedding(10, 4, padding_idx=0)))
    model.to(torch.float64)  # .to() of any kind leaves the plain tensor alone
    assert model[0].table.dtype == torch.float32
    assert list(model.parameters()) == []  # frozen: nothing for the optimizer to train


def test_dpo_validation_pairs_are_held_out():
    pairs = pd.DataFrame({"prompt": [[{"role": "user", "content": str(i)}] for i in range(20)],
                          "chosen": [[{"role": "assistant", "content": "e2e4"}]] * 20,
                          "rejected": [[{"role": "assistant", "content": "a2a3"}]] * 20})
    train, val = split_pairs(pairs, 0.25, seed=0)
    assert (len(train), len(val)) == (15, 5)
    assert not {p[0]["content"] for p in train["prompt"]} & {p[0]["content"] for p in val["prompt"]}
