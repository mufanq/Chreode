rule validate_config:
    output:
        f"{OUT_ROOT}/manifest/config.validated.txt",
    shell:
        """
        mkdir -p {OUT_ROOT}/manifest
        PYTHONPATH=src python -m cellworldmodel.script.run_foundation validate-config \
          --config {CONFIGFILE_PATH} > {output}
        """


rule build_gene_vocab:
    input:
        f"{OUT_ROOT}/manifest/config.validated.txt",
    output:
        f"{OUT_ROOT}/catalog/gene_vocab.parquet",
    shell:
        """
        PYTHONPATH=src python -m cellworldmodel.script.run_foundation build-gene-vocab \
          --config {CONFIGFILE_PATH} \
          --output-dir {OUT_ROOT}/catalog
        """


rule build_catalog:
    input:
        f"{OUT_ROOT}/catalog/gene_vocab.parquet",
    output:
        f"{OUT_ROOT}/catalog/data_manifest.json",
        f"{OUT_ROOT}/catalog/cell_index.parquet",
        f"{OUT_ROOT}/catalog/split_manifest.json",
    shell:
        """
        PYTHONPATH=src python -m cellworldmodel.script.run_foundation build-catalog \
          --config {CONFIGFILE_PATH} \
          --output-dir {OUT_ROOT}/catalog
        """
