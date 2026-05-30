rule eval_vae_arch_embedding:
    input:
        checkpoint=f"{OUT_ROOT}/vae/arch_{{name}}/model.pt",
        catalog=f"{OUT_ROOT}/catalog/data_manifest.json",
    output:
        metrics=f"{OUT_ROOT}/vae/arch_{{name}}/embedding_eval/metrics.json",
        labels=f"{OUT_ROOT}/vae/arch_{{name}}/embedding_eval/label_metrics.tsv",
        centroids=f"{OUT_ROOT}/vae/arch_{{name}}/embedding_eval/centroid_metrics.tsv",
        metadata=f"{OUT_ROOT}/vae/arch_{{name}}/embedding_eval/embedding_metadata.tsv",
        embeddings=f"{OUT_ROOT}/vae/arch_{{name}}/embedding_eval/embedding_sample.npz",
        manifest=f"{OUT_ROOT}/vae/arch_{{name}}/embedding_eval/manifest.json",
        umap_leiden=f"{OUT_ROOT}/vae/arch_{{name}}/embedding_eval/plots/umap_by_leiden.png",
        umap_cell_type=f"{OUT_ROOT}/vae/arch_{{name}}/embedding_eval/plots/umap_by_cell_type.png",
        metric_plot=f"{OUT_ROOT}/vae/arch_{{name}}/embedding_eval/plots/metric_summary.png",
    shell:
        """
        source 3rdparty/BranchSBM/.branchsbm-venv/bin/activate
        PYTHONPATH=src python -m cellworldmodel.script.run_foundation eval-vae-embedding \
          --checkpoint {input.checkpoint} \
          --catalog-dir {OUT_ROOT}/catalog \
          --output-dir {OUT_ROOT}/vae/arch_{wildcards.name}/embedding_eval \
          --sample-size {VAE_EMBEDDING_EVAL_SAMPLE_SIZE} \
          --batch-size {VAE_EMBEDDING_EVAL_BATCH_SIZE} \
          --device cpu
        """
