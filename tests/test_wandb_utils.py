from cellworldmodel.script.wandb_utils import flatten_numeric


def test_flatten_numeric_keeps_only_numeric_values():
    out = flatten_numeric({
        "train": {"loss": 1.2, "name": "x"},
        "eval": [{"w2": 0.3}, {"nan": float("nan")}],
        "flag": True,
    })
    assert out == {
        "train/loss": 1.2,
        "eval/0/w2": 0.3,
    }
