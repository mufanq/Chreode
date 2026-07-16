from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


EXPECTED_N_GENES = 16485
EXPECTED_GENE_LIST_SHA1 = "17481f015e4fdc6220f7764d8c5341a52b164bfa"


def test_bundled_gene_vocab_matches_released_vae_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    table = pd.read_parquet(root / "artifacts" / "gene_vocab.parquet")

    assert len(table) == EXPECTED_N_GENES
    assert table["canonical_index"].astype(int).tolist() == list(range(EXPECTED_N_GENES))
    assert table["canonical_gene"].is_unique
    assert table["mouse_symbol"].is_unique
    assert table["human_symbol"].is_unique

    genes = table["canonical_gene"].astype(str).tolist()
    digest = hashlib.sha1("\0".join(genes).encode("utf-8")).hexdigest()
    assert digest == EXPECTED_GENE_LIST_SHA1
