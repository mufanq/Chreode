from pathlib import Path

import h5py
import pandas as pd
import yaml

from cellworldmodel.foundation.catalog import build_catalog
from cellworldmodel.foundation.config import load_foundation_config
from cellworldmodel.foundation.expression_dataset import FoundationExpressionDataset
from cellworldmodel.foundation.gene_vocab import build_gene_vocab, write_gene_vocab
from cellworldmodel.foundation.catalog import write_catalog


def write_h5ad(path: Path, genes: list[str], n_obs: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        x = handle.create_group("X")
        x.attrs["shape"] = (n_obs, len(genes))
        x.create_dataset("data", data=[1.0, 2.0, 3.0])
        x.create_dataset("indices", data=[0, 1, 2])
        x.create_dataset("indptr", data=[0, 1, 2, 3])
        obs = handle.create_group("obs")
        obs.create_dataset("barcode", data=[f"cell{i}".encode() for i in range(n_obs)])
        obs.create_dataset("cell_type", data=[b"a", b"b", b"c"][:n_obs])
        obs.create_dataset("time_of_sampling", data=[b"1.0"] * n_obs)
        var = handle.create_group("var")
        var.create_dataset("gene_name", data=[g.encode() for g in genes])
        handle.create_group("obsm")
        handle.create_group("layers")


def write_config(tmp_path: Path, data_root: Path, ortholog_path: Path) -> Path:
    cfg = {
        "output_root": str(tmp_path / "out"),
        "data_root": str(data_root),
        "gene_vocab": {
            "source": str(ortholog_path),
            "canonical_order": "mouse_unified_order_filtered",
        },
    }
    path = tmp_path / "foundation.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def test_build_gene_vocab_uses_mouse_unified_order(tmp_path):
    data_root = tmp_path / "data"
    write_h5ad(data_root / "GSE1" / "leaf" / "day_1p0.h5ad", ["B", "A", "B", "C"])
    ortholog = pd.DataFrame({
        "mouse_symbol": ["A", "B"],
        "human_symbol": ["HA", "HB"],
        "mouse_ensembl": ["MA", "MB"],
        "human_ensembl": ["HAE", "HBE"],
        "orthology_type": ["one2one", "one2one"],
        "source": ["test", "test"],
    })
    ortholog_path = tmp_path / "ortholog.parquet"
    ortholog.to_parquet(ortholog_path, index=False)
    cfg = load_foundation_config(write_config(tmp_path, data_root, ortholog_path))

    result = build_gene_vocab(cfg)

    assert result.table["canonical_gene"].tolist() == ["B", "A"]
    assert result.table["human_symbol"].tolist() == ["HB", "HA"]
    assert result.n_duplicate_unified_genes_skipped == 1
    assert len(result.sha1) == 40


def test_build_catalog_reads_h5ad_metadata(tmp_path):
    data_root = tmp_path / "data"
    write_h5ad(data_root / "GSE1" / "leaf" / "day_1p0.h5ad", ["A", "B"], n_obs=2)
    write_h5ad(data_root / "GSE1" / "leaf" / "day_2p0.h5ad", ["A", "B"], n_obs=3)
    ortholog = pd.DataFrame({"mouse_symbol": ["A"], "human_symbol": ["HA"]})
    ortholog_path = tmp_path / "ortholog.parquet"
    ortholog.to_parquet(ortholog_path, index=False)
    cfg = load_foundation_config(write_config(tmp_path, data_root, ortholog_path))

    catalog = build_catalog(cfg)

    assert len(catalog.h5ad_files) == 2
    assert len(catalog.cell_index) == 5
    assert catalog.manifest["n_cells"] == 5
    assert catalog.manifest["n_h5ad_files"] == 2
    assert set(catalog.cell_index["foundation_split"]).issubset({"train", "val", "test"})
    assert set(catalog.cell_index["barcode"]) == {"cell0", "cell1", "cell2"}


def test_build_catalog_marks_heldout_family(tmp_path):
    data_root = tmp_path / "data"
    write_h5ad(data_root / "GSE275562" / "leaf" / "day_1p0.h5ad", ["A", "B"], n_obs=2)
    ortholog = pd.DataFrame({"mouse_symbol": ["A"], "human_symbol": ["HA"]})
    ortholog_path = tmp_path / "ortholog.parquet"
    ortholog.to_parquet(ortholog_path, index=False)
    cfg = load_foundation_config(write_config(tmp_path, data_root, ortholog_path))

    catalog = build_catalog(cfg)

    assert set(catalog.cell_index["foundation_split"]) == {"heldout"}
    assert catalog.manifest["split_counts"] == {"heldout": 2}


def test_expression_dataset_reads_ortholog_log1p_batches(tmp_path):
    data_root = tmp_path / "data"
    write_h5ad(data_root / "GSE1" / "leaf" / "day_1p0.h5ad", ["B", "A", "C"], n_obs=3)
    ortholog = pd.DataFrame({
        "mouse_symbol": ["A", "B"],
        "human_symbol": ["HA", "HB"],
    })
    ortholog_path = tmp_path / "ortholog.parquet"
    ortholog.to_parquet(ortholog_path, index=False)
    cfg = load_foundation_config(write_config(tmp_path, data_root, ortholog_path))
    catalog_dir = tmp_path / "catalog"
    write_gene_vocab(build_gene_vocab(cfg), catalog_dir)
    write_catalog(build_catalog(cfg), catalog_dir)
    dataset = FoundationExpressionDataset(catalog_dir, target_sum=1e4)

    batch = dataset.load_cells([0, 1, 2])

    assert batch.x.shape == (3, 2)
    assert batch.cell_ids.tolist() == [0, 1, 2]
    # Gene vocab follows h5ad order filtered to orthologs: B, A.
    assert dataset.gene_vocab["canonical_gene"].tolist() == ["B", "A"]
    assert float(batch.x[0, 0]) > 0.0
    assert float(batch.x[0, 1]) == 0.0
