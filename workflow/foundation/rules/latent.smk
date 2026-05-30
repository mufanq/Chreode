rule encode_latent_cache:
    input:
        checkpoint=f"{VAE_FULL_DIR}/epoch_{VAE_FULL_EPOCHS}.pt",
        catalog=f"{OUT_ROOT}/catalog/data_manifest.json",
    output:
        manifest=f"{LATENT_CACHE_DIR}/manifest.json",
        index=f"{LATENT_CACHE_DIR}/index.parquet",
        slurm=f"{LATENT_CACHE_DIR}/slurm_job.json",
    params:
        submit=lambda wc: sbatch_wait(
            job_name=f"foundation-latent-{VAE_FULL_NAME}",
            partition=RES.get("partition", "blackwell,a100"),
            runtime_min=RES.get("latent_cache_runtime_min", RES.get("runtime_min", 720)),
            cpus=RES.get("cpus", 8),
            mem_mb=RES.get("latent_cache_mem_mb", RES.get("vae_smoke_mem_mb", RES.get("mem_mb", 60000))),
            stdout=f"logs/foundation/latent_cache_{VAE_FULL_NAME}.out",
            stderr=f"logs/foundation/latent_cache_{VAE_FULL_NAME}.err",
            metadata=f"{LATENT_CACHE_DIR}/slurm_job.json",
            gres=RES.get("gres", "gpu:1"),
            qos=RES.get("qos"),
            command=(
                "source 3rdparty/BranchSBM/.branchsbm-venv/bin/activate && "
                "PYTHONPATH=src python -m cellworldmodel.script.run_foundation encode-latent-cache "
                f"--checkpoint {VAE_FULL_DIR}/epoch_{VAE_FULL_EPOCHS}.pt "
                f"--catalog-dir {OUT_ROOT}/catalog "
                f"--output-dir {LATENT_CACHE_DIR} "
                f"--splits {LATENT_CACHE_SPLITS_ARG} "
                f"--batch-size {LATENT_CACHE_BATCH_SIZE} "
                f"--shard-size {LATENT_CACHE_SHARD_SIZE} "
                "--device cuda"
            ),
        ),
    shell:
        "{params.submit}"
