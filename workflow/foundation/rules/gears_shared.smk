rule export_gears_official_predictions:
    input:
        model=f"{GEARS_OFFICIAL_MODEL_DIR}/model.pt",
        config=f"{GEARS_OFFICIAL_MODEL_DIR}/config.pkl",
        adata=GEARS_ADATA,
        subgroup=GEARS_SUBGROUP,
    output:
        prediction=f"{GEARS_OFFICIAL_EXPORT_DIR}/predictions.npz",
        summary=f"{GEARS_OFFICIAL_EXPORT_DIR}/export_summary.json",
        test_res=f"{GEARS_OFFICIAL_EXPORT_DIR}/test_res.pkl",
        slurm=f"{GEARS_OFFICIAL_EXPORT_DIR}/slurm_job.json",
    params:
        submit=lambda wc: sbatch_wait(
            job_name="gears-export-norman",
            partition=RES.get("gears_export_partition", "a100"),
            runtime_min=RES.get("gears_export_runtime_min", 60),
            cpus=RES.get("cpus", 8),
            mem_mb=RES.get("gears_export_mem_mb", RES.get("mem_mb", 60000)),
            stdout="logs/foundation/gears_export_norman.out",
            stderr="logs/foundation/gears_export_norman.err",
            metadata=f"{GEARS_OFFICIAL_EXPORT_DIR}/slurm_job.json",
            gres=RES.get("gres", "gpu:1"),
            qos=RES.get("qos"),
            command=(
                f"{GEARS_VENV_PYTHON} scripts/gears/export_norman_gears_predictions.py "
                f"--model-dir {GEARS_OFFICIAL_MODEL_DIR} "
                f"--output-dir {GEARS_OFFICIAL_EXPORT_DIR} "
                f"--data-dir {GEARS_DATA_DIR} "
                "--device cuda:0"
            ),
        ),
    shell:
        "{params.submit}"


rule eval_gears_shared_vocab:
    input:
        prediction=f"{GEARS_OFFICIAL_EXPORT_DIR}/predictions.npz",
        adata=GEARS_ADATA,
        subgroup=GEARS_SUBGROUP,
        genes=f"{OUT_ROOT}/catalog/gene_vocab.parquet",
    output:
        summary=f"{GEARS_SHARED_EVAL_DIR}/shared_vocab_summary.json",
        conditions=f"{GEARS_SHARED_EVAL_DIR}/shared_vocab_conditions.tsv",
        coverage=f"{GEARS_SHARED_EVAL_DIR}/shared_vocab_gene_coverage.tsv",
        genes=f"{GEARS_SHARED_EVAL_DIR}/shared_vocab_genes.txt",
    shell:
        """
        PYTHONPATH=src python -m cellworldmodel.script.run_foundation eval-gears-shared-vocab \
          --gears-adata {input.adata} \
          --prediction {input.prediction} \
          --ours-genes {input.genes} \
          --subgroup {input.subgroup} \
          --output-dir {GEARS_SHARED_EVAL_DIR}
        """


rule export_foundation_gears_predictions:
    input:
        checkpoint=f"{PERTURB_DIR}/{{arm}}/model.pt",
        vae=f"{VAE_FULL_DIR}/epoch_{VAE_FULL_EPOCHS}.pt",
        genes=f"{OUT_ROOT}/catalog/gene_vocab.parquet",
        adata=GEARS_ADATA,
        subgroup=GEARS_SUBGROUP,
    output:
        prediction=f"{GEARS_FOUNDATION_EVAL_DIR}/{{arm}}_export/predictions.npz",
        summary=f"{GEARS_FOUNDATION_EVAL_DIR}/{{arm}}_export/export_summary.json",
        slurm=f"{GEARS_FOUNDATION_EVAL_DIR}/{{arm}}_export/slurm_job.json",
    params:
        submit=lambda wc: sbatch_wait(
            job_name=f"foundation-gears-export-{wc.arm}",
            partition=RES.get("gears_foundation_export_partition", "blackwell,a100"),
            runtime_min=RES.get("gears_foundation_export_runtime_min", 60),
            cpus=RES.get("cpus", 8),
            mem_mb=RES.get("gears_foundation_export_mem_mb", RES.get("mem_mb", 60000)),
            stdout=f"logs/foundation/gears_foundation_export_{wc.arm}.out",
            stderr=f"logs/foundation/gears_foundation_export_{wc.arm}.err",
            metadata=f"{GEARS_FOUNDATION_EVAL_DIR}/{wc.arm}_export/slurm_job.json",
            gres=RES.get("gres", "gpu:1"),
            qos=RES.get("qos"),
            command=(
                f"PYTHONPATH=src {GEARS_DOWNSTREAM_PYTHON} -m cellworldmodel.script.run_foundation export-gears-foundation-predictions "
                f"--config {CONFIGFILE_PATH} "
                f"--gears-adata {GEARS_ADATA} "
                f"--gene-vocab {OUT_ROOT}/catalog/gene_vocab.parquet "
                f"--vae-checkpoint {VAE_FULL_DIR}/epoch_{VAE_FULL_EPOCHS}.pt "
                f"--perturbation-checkpoint {PERTURB_DIR}/{wc.arm}/model.pt "
                f"--subgroup {GEARS_SUBGROUP} "
                f"--output-dir {GEARS_FOUNDATION_EVAL_DIR}/{wc.arm}_export "
                f"--experiment {DYNAMICS_EXPERIMENT} "
                f"--dit-size {DYNAMICS_DIT_SIZE} "
                f"--batch-size {PERT_CFG.get('gears_export_batch_size', 512)} "
                f"--k-samples {DYNAMICS_K} "
                f"--action-dim {PERT_CFG.get('action_dim', 64)} "
                "--device cuda"
            ),
        ),
    shell:
        "{params.submit}"


rule eval_foundation_gears_shared_vocab:
    input:
        prediction=f"{GEARS_FOUNDATION_EVAL_DIR}/{{arm}}_export/predictions.npz",
        adata=GEARS_ADATA,
        subgroup=GEARS_SUBGROUP,
        genes=f"{OUT_ROOT}/catalog/gene_vocab.parquet",
    output:
        summary=f"{GEARS_FOUNDATION_EVAL_DIR}/{{arm}}_shared_eval/shared_vocab_summary.json",
        conditions=f"{GEARS_FOUNDATION_EVAL_DIR}/{{arm}}_shared_eval/shared_vocab_conditions.tsv",
        coverage=f"{GEARS_FOUNDATION_EVAL_DIR}/{{arm}}_shared_eval/shared_vocab_gene_coverage.tsv",
        genes=f"{GEARS_FOUNDATION_EVAL_DIR}/{{arm}}_shared_eval/shared_vocab_genes.txt",
    shell:
        """
        PYTHONPATH=src python -m cellworldmodel.script.run_foundation eval-gears-shared-vocab \
          --gears-adata {input.adata} \
          --prediction {input.prediction} \
          --ours-genes {input.genes} \
          --subgroup {input.subgroup} \
          --output-dir {GEARS_FOUNDATION_EVAL_DIR}/{wildcards.arm}_shared_eval
        """


rule train_gears_downstream:
    input:
        vae=f"{VAE_FULL_DIR}/epoch_{VAE_FULL_EPOCHS}.pt",
        genes=f"{OUT_ROOT}/catalog/gene_vocab.parquet",
        adata=GEARS_ADATA,
        split=GEARS_SPLIT,
        subgroup=GEARS_SUBGROUP,
        init=lambda wc: gears_init_checkpoint(wc.arm) or [],
    output:
        summary=f"{GEARS_DOWNSTREAM_DIR}/{{arm}}/summary.json",
        model=f"{GEARS_DOWNSTREAM_DIR}/{{arm}}/model.pt",
        history=f"{GEARS_DOWNSTREAM_DIR}/{{arm}}/history.tsv",
        predictions=f"{GEARS_DOWNSTREAM_DIR}/{{arm}}/predictions.npz",
        shared=f"{GEARS_DOWNSTREAM_DIR}/{{arm}}/shared_eval/shared_vocab_summary.json",
        slurm=f"{GEARS_DOWNSTREAM_DIR}/{{arm}}/slurm_job.json",
    params:
        submit=lambda wc: sbatch_wait(
            job_name=f"foundation-gears-downstream-{wc.arm}",
            partition=RES.get("gears_downstream_partition", "blackwell,a100"),
            runtime_min=RES.get("gears_downstream_runtime_min", 180),
            cpus=RES.get("cpus", 8),
            mem_mb=min(int(RES.get("gears_downstream_mem_mb", RES.get("mem_mb", 60000))), 60000),
            stdout=f"logs/foundation/gears_downstream_{wc.arm}.out",
            stderr=f"logs/foundation/gears_downstream_{wc.arm}.err",
            metadata=f"{GEARS_DOWNSTREAM_DIR}/{wc.arm}/slurm_job.json",
            gres=RES.get("gres", "gpu:1"),
            qos=RES.get("qos"),
            command=gears_downstream_command(wc.arm),
        ),
    shell:
        "{params.submit}"


rule train_gears_downstream_hybrid:
    input:
        vae=f"{VAE_FULL_DIR}/epoch_{VAE_FULL_EPOCHS}.pt",
        genes=f"{OUT_ROOT}/catalog/gene_vocab.parquet",
        adata=GEARS_ADATA,
        split=GEARS_SPLIT,
        subgroup=GEARS_SUBGROUP,
        init=lambda wc: gears_init_checkpoint(wc.arm) or [],
    output:
        summary=f"{GEARS_HYBRID_DIR}/{{arm}}/summary.json",
        model=f"{GEARS_HYBRID_DIR}/{{arm}}/model.pt",
        history=f"{GEARS_HYBRID_DIR}/{{arm}}/history.tsv",
        predictions=f"{GEARS_HYBRID_DIR}/{{arm}}/predictions.npz",
        shared=f"{GEARS_HYBRID_DIR}/{{arm}}/shared_eval/shared_vocab_summary.json",
        slurm=f"{GEARS_HYBRID_DIR}/{{arm}}/slurm_job.json",
    params:
        submit=lambda wc: sbatch_wait(
            job_name=f"foundation-gears-hybrid-{wc.arm}",
            partition=RES.get("gears_downstream_partition", "blackwell,a100"),
            runtime_min=RES.get("gears_downstream_runtime_min", 180),
            cpus=RES.get("cpus", 8),
            mem_mb=min(int(RES.get("gears_downstream_mem_mb", RES.get("mem_mb", 60000))), 60000),
            stdout=f"logs/foundation/gears_hybrid_{wc.arm}.out",
            stderr=f"logs/foundation/gears_hybrid_{wc.arm}.err",
            metadata=f"{GEARS_HYBRID_DIR}/{wc.arm}/slurm_job.json",
            gres=RES.get("gres", "gpu:1"),
            qos=RES.get("qos"),
            command=gears_downstream_command(
                wc.arm,
                model_type="hybrid_kick_rollout",
                output_dir=f"{GEARS_HYBRID_DIR}/{wc.arm}",
            ),
        ),
    shell:
        "{params.submit}"


rule train_gears_foundation_embedding:
    input:
        vae=f"{VAE_FULL_DIR}/epoch_{VAE_FULL_EPOCHS}.pt",
        genes=f"{OUT_ROOT}/catalog/gene_vocab.parquet",
        adata=GEARS_ADATA,
        subgroup=GEARS_SUBGROUP,
        static=lambda wc: f"{DYNAMICS_STATIC_DIR}/model.pt" if wc.arm == "static" else [],
        dynamics=lambda wc: f"{DYNAMICS_TEMPORAL_DIR}/model.pt" if wc.arm == "dynamics" else [],
    output:
        summary=f"{GEARS_FOUNDATION_EMB_PREFIX}_{{arm}}_{GEARS_FOUNDATION_EMB_MODE}{GEARS_FOUNDATION_EMB_EPOCHS}/summary.json",
        test_res=f"{GEARS_FOUNDATION_EMB_PREFIX}_{{arm}}_{GEARS_FOUNDATION_EMB_MODE}{GEARS_FOUNDATION_EMB_EPOCHS}/test_res.pkl",
        shared=f"{GEARS_FOUNDATION_EMB_PREFIX}_{{arm}}_{GEARS_FOUNDATION_EMB_MODE}{GEARS_FOUNDATION_EMB_EPOCHS}/shared_eval/shared_vocab_summary.json",
        slurm=f"{GEARS_FOUNDATION_EMB_PREFIX}_{{arm}}_{GEARS_FOUNDATION_EMB_MODE}{GEARS_FOUNDATION_EMB_EPOCHS}/slurm_job.json",
    params:
        submit=lambda wc: sbatch_wait(
            job_name=f"gears-foundation-emb-{wc.arm}",
            partition=RES.get("gears_foundation_embedding_partition", "a100"),
            runtime_min=RES.get("gears_foundation_embedding_runtime_min", 240),
            cpus=RES.get("cpus", 8),
            mem_mb=min(int(RES.get("gears_foundation_embedding_mem_mb", RES.get("mem_mb", 60000))), 60000),
            stdout=f"logs/foundation/gears_foundation_emb_{wc.arm}_{GEARS_FOUNDATION_EMB_MODE}{GEARS_FOUNDATION_EMB_EPOCHS}.out",
            stderr=f"logs/foundation/gears_foundation_emb_{wc.arm}_{GEARS_FOUNDATION_EMB_MODE}{GEARS_FOUNDATION_EMB_EPOCHS}.err",
            metadata=f"{GEARS_FOUNDATION_EMB_PREFIX}_{wc.arm}_{GEARS_FOUNDATION_EMB_MODE}{GEARS_FOUNDATION_EMB_EPOCHS}/slurm_job.json",
            gres=RES.get("gres", "gpu:1"),
            qos=RES.get("qos"),
            command=gears_foundation_embedding_command(wc.arm),
        ),
    shell:
        "{params.submit}"
