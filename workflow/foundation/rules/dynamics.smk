rule build_foundation_transition_index:
    input:
        f"{OUT_ROOT}/catalog/data_manifest.json",
        f"{OUT_ROOT}/catalog/cell_index.parquet",
    output:
        index=f"{TRANSITION_INDEX_DIR}/transition_index.parquet",
        manifest=f"{TRANSITION_INDEX_DIR}/transition_manifest.json",
    shell:
        """
        PYTHONPATH=src python -m cellworldmodel.script.run_foundation build-transition-index \
          --catalog-dir {OUT_ROOT}/catalog \
          --output-dir {TRANSITION_INDEX_DIR} \
          --split train \
          --pair-policy {TRANSITION_PAIR_POLICY}
        """


rule train_static_dit_pretrain:
    input:
        checkpoint=f"{VAE_FULL_DIR}/epoch_{VAE_FULL_EPOCHS}.pt",
        catalog=f"{OUT_ROOT}/catalog/data_manifest.json",
    output:
        summary=f"{DYNAMICS_STATIC_DIR}/summary.json",
        model=f"{DYNAMICS_STATIC_DIR}/model.pt",
        history=f"{DYNAMICS_STATIC_DIR}/history.tsv",
        slurm=f"{DYNAMICS_STATIC_DIR}/slurm_job.json",
    params:
        submit=lambda wc: sbatch_wait(
            job_name=f"foundation-dyn-{DYNAMICS_STATIC_NAME}",
            partition=RES.get("partition", "blackwell,a100"),
            runtime_min=RES.get("dynamics_runtime_min", RES.get("runtime_min", 720)),
            cpus=RES.get("cpus", 8),
            mem_mb=RES.get("dynamics_mem_mb", RES.get("mem_mb", 60000)),
            stdout=f"logs/foundation/dynamics_{DYNAMICS_STATIC_NAME}.out",
            stderr=f"logs/foundation/dynamics_{DYNAMICS_STATIC_NAME}.err",
            metadata=f"{DYNAMICS_STATIC_DIR}/slurm_job.json",
            gres=RES.get("gres", "gpu:1"),
            qos=RES.get("qos"),
            command=foundation_dynamics_command(
                DYNAMICS_STATIC_NAME,
                DYNAMICS_STATIC_OBJECTIVE,
                DYNAMICS_STATIC_DIR,
                extra_args=f"--vae-checkpoint {VAE_FULL_DIR}/epoch_{VAE_FULL_EPOCHS}.pt --static-delta {DYNAMICS_STATIC_DELTA}",
            ),
        ),
    shell:
        "{params.submit}"


rule train_temporal_dit_pretrain:
    input:
        latent=f"{LATENT_CACHE_DIR}/manifest.json",
        transition=f"{TRANSITION_INDEX_DIR}/transition_index.parquet",
    output:
        summary=f"{DYNAMICS_TEMPORAL_DIR}/summary.json",
        model=f"{DYNAMICS_TEMPORAL_DIR}/model.pt",
        history=f"{DYNAMICS_TEMPORAL_DIR}/history.tsv",
        slurm=f"{DYNAMICS_TEMPORAL_DIR}/slurm_job.json",
    params:
        submit=lambda wc: sbatch_wait(
            job_name=f"foundation-dyn-{DYNAMICS_TEMPORAL_NAME}",
            partition=RES.get("partition", "blackwell,a100"),
            runtime_min=RES.get("dynamics_runtime_min", RES.get("runtime_min", 720)),
            cpus=RES.get("cpus", 8),
            mem_mb=RES.get("dynamics_mem_mb", RES.get("mem_mb", 60000)),
            stdout=f"logs/foundation/dynamics_{DYNAMICS_TEMPORAL_NAME}.out",
            stderr=f"logs/foundation/dynamics_{DYNAMICS_TEMPORAL_NAME}.err",
            metadata=f"{DYNAMICS_TEMPORAL_DIR}/slurm_job.json",
            gres=RES.get("gres", "gpu:1"),
            qos=RES.get("qos"),
            command=foundation_dynamics_command(
                DYNAMICS_TEMPORAL_NAME,
                DYNAMICS_TEMPORAL_OBJECTIVE,
                DYNAMICS_TEMPORAL_DIR,
                extra_args=f"--latent-cache-dir {LATENT_CACHE_DIR} --transition-index-dir {TRANSITION_INDEX_DIR}",
            ),
        ),
    shell:
        "{params.submit}"
