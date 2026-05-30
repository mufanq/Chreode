rule train_vae_smoke:
    input:
        f"{OUT_ROOT}/catalog/data_manifest.json",
        f"{OUT_ROOT}/catalog/cell_index.parquet",
        f"{OUT_ROOT}/catalog/gene_vocab.parquet",
    output:
        qc=f"{OUT_ROOT}/vae/smoke_{{strategy}}_latent{{latent_dim}}/qc.json",
        model=f"{OUT_ROOT}/vae/smoke_{{strategy}}_latent{{latent_dim}}/model.pt",
        history=f"{OUT_ROOT}/vae/smoke_{{strategy}}_latent{{latent_dim}}/history.tsv",
        slurm=f"{OUT_ROOT}/vae/smoke_{{strategy}}_latent{{latent_dim}}/slurm_job.json",
    params:
        submit=lambda wc: sbatch_wait(
            job_name=f"foundation-vae-smoke-{wc.strategy}-l{wc.latent_dim}",
            partition=RES.get("partition", "blackwell,a100"),
            runtime_min=RES.get("vae_smoke_runtime_min", RES.get("runtime_min", 240)),
            cpus=RES.get("cpus", 8),
            mem_mb=RES.get("vae_smoke_mem_mb", RES.get("mem_mb", 60000)),
            stdout=f"logs/foundation/vae_smoke_{wc.strategy}_latent{wc.latent_dim}.out",
            stderr=f"logs/foundation/vae_smoke_{wc.strategy}_latent{wc.latent_dim}.err",
            metadata=f"{OUT_ROOT}/vae/smoke_{wc.strategy}_latent{wc.latent_dim}/slurm_job.json",
            gres=RES.get("gres", "gpu:1"),
            qos=RES.get("qos"),
            command=(
                "source 3rdparty/BranchSBM/.branchsbm-venv/bin/activate && "
                "export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True} && "
                "PYTHONPATH=src python -m cellworldmodel.script.run_foundation train-vae-smoke "
                f"--config {CONFIGFILE_PATH} "
                f"--catalog-dir {OUT_ROOT}/catalog "
                f"--output-dir {OUT_ROOT}/vae/smoke_{wc.strategy}_latent{wc.latent_dim} "
                f"--architecture {VAE_CFG.get('architecture', 'mlp512')} "
                f"--latent-dim {wc.latent_dim} "
                f"--batch-strategy {wc.strategy} "
                f"--max-steps {VAE_CFG.get('smoke_steps', 200)} "
                f"--batch-size {VAE_CFG.get('smoke_batch_size', 256)} "
                f"--qc-sample-size {VAE_CFG.get('qc_sample_size', 2048)} "
                "--seed 0 "
                f"{wandb_args(f'foundation-vae-smoke-{wc.strategy}-l{wc.latent_dim}', tags_extra='vae_smoke')}"
            ),
        ),
    shell:
        "{params.submit}"


rule train_vae_throughput:
    input:
        f"{OUT_ROOT}/catalog/data_manifest.json",
        f"{OUT_ROOT}/catalog/cell_index.parquet",
        f"{OUT_ROOT}/catalog/gene_vocab.parquet",
    output:
        qc=f"{OUT_ROOT}/vae/throughput_{{strategy}}_latent{{latent_dim}}_batch{{batch_size}}/qc.json",
        model=f"{OUT_ROOT}/vae/throughput_{{strategy}}_latent{{latent_dim}}_batch{{batch_size}}/model.pt",
        history=f"{OUT_ROOT}/vae/throughput_{{strategy}}_latent{{latent_dim}}_batch{{batch_size}}/history.tsv",
        slurm=f"{OUT_ROOT}/vae/throughput_{{strategy}}_latent{{latent_dim}}_batch{{batch_size}}/slurm_job.json",
    params:
        submit=lambda wc: sbatch_wait(
            job_name=f"foundation-vae-throughput-{wc.strategy}-l{wc.latent_dim}-b{wc.batch_size}",
            partition=RES.get("partition", "blackwell,a100"),
            runtime_min=RES.get("vae_smoke_runtime_min", RES.get("runtime_min", 240)),
            cpus=RES.get("cpus", 8),
            mem_mb=RES.get("vae_smoke_mem_mb", RES.get("mem_mb", 60000)),
            stdout=f"logs/foundation/vae_throughput_{wc.strategy}_latent{wc.latent_dim}_batch{wc.batch_size}.out",
            stderr=f"logs/foundation/vae_throughput_{wc.strategy}_latent{wc.latent_dim}_batch{wc.batch_size}.err",
            metadata=f"{OUT_ROOT}/vae/throughput_{wc.strategy}_latent{wc.latent_dim}_batch{wc.batch_size}/slurm_job.json",
            gres=RES.get("gres", "gpu:1"),
            qos=RES.get("qos"),
            command=(
                "source 3rdparty/BranchSBM/.branchsbm-venv/bin/activate && "
                "export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True} && "
                "PYTHONPATH=src python -m cellworldmodel.script.run_foundation train-vae-smoke "
                f"--config {CONFIGFILE_PATH} "
                f"--catalog-dir {OUT_ROOT}/catalog "
                f"--output-dir {OUT_ROOT}/vae/throughput_{wc.strategy}_latent{wc.latent_dim}_batch{wc.batch_size} "
                f"--architecture {VAE_CFG.get('architecture', 'mlp512')} "
                f"--latent-dim {wc.latent_dim} "
                f"--batch-strategy {wc.strategy} "
                f"--max-steps {VAE_CFG.get('throughput_steps', 50)} "
                f"--batch-size {wc.batch_size} "
                f"--qc-sample-size {VAE_CFG.get('qc_sample_size', 2048)} "
                "--seed 0 "
                f"{wandb_args(f'foundation-vae-throughput-{wc.strategy}-l{wc.latent_dim}-b{wc.batch_size}', tags_extra='vae_throughput')}"
            ),
        ),
    shell:
        "{params.submit}"


rule train_vae_arch_search:
    input:
        f"{OUT_ROOT}/catalog/data_manifest.json",
        f"{OUT_ROOT}/catalog/cell_index.parquet",
        f"{OUT_ROOT}/catalog/gene_vocab.parquet",
    output:
        qc=f"{OUT_ROOT}/vae/arch_{{name}}/qc.json",
        model=f"{OUT_ROOT}/vae/arch_{{name}}/model.pt",
        history=f"{OUT_ROOT}/vae/arch_{{name}}/history.tsv",
        slurm=f"{OUT_ROOT}/vae/arch_{{name}}/slurm_job.json",
    params:
        submit=lambda wc: sbatch_wait(
            job_name=f"foundation-vae-arch-{wc.name}",
            partition=RES.get("partition", "blackwell,a100"),
            runtime_min=RES.get("vae_arch_runtime_min", RES.get("vae_smoke_runtime_min", RES.get("runtime_min", 360))),
            cpus=RES.get("cpus", 8),
            mem_mb=RES.get("vae_smoke_mem_mb", RES.get("mem_mb", 60000)),
            stdout=f"logs/foundation/vae_arch_{wc.name}.out",
            stderr=f"logs/foundation/vae_arch_{wc.name}.err",
            metadata=f"{OUT_ROOT}/vae/arch_{wc.name}/slurm_job.json",
            gres=RES.get("gres", "gpu:1"),
            qos=RES.get("qos"),
            command=vae_arch_command(wc),
        ),
    shell:
        "{params.submit}"


rule train_vae_full:
    input:
        f"{OUT_ROOT}/catalog/data_manifest.json",
        f"{OUT_ROOT}/catalog/cell_index.parquet",
        f"{OUT_ROOT}/catalog/gene_vocab.parquet",
    output:
        qc=f"{VAE_FULL_DIR}/qc.json",
        model=f"{VAE_FULL_DIR}/model.pt",
        history=f"{VAE_FULL_DIR}/history.tsv",
        epochs=VAE_FULL_EPOCH_OUTPUTS,
        slurm=f"{VAE_FULL_DIR}/slurm_job.json",
    params:
        submit=lambda wc: sbatch_wait(
            job_name=f"foundation-vae-full-{VAE_FULL_NAME}",
            partition=RES.get("partition", "blackwell,a100"),
            runtime_min=RES.get("vae_full_runtime_min", RES.get("runtime_min", 720)),
            cpus=RES.get("cpus", 8),
            mem_mb=RES.get("vae_full_mem_mb", RES.get("vae_smoke_mem_mb", RES.get("mem_mb", 60000))),
            stdout=f"logs/foundation/vae_full_{VAE_FULL_NAME}.out",
            stderr=f"logs/foundation/vae_full_{VAE_FULL_NAME}.err",
            metadata=f"{VAE_FULL_DIR}/slurm_job.json",
            gres=RES.get("gres", "gpu:1"),
            qos=RES.get("qos"),
            command=vae_full_command(),
        ),
    shell:
        "{params.submit}"
