rule train_perturbation_vae2_only:
    input:
        checkpoint=f"{VAE_FULL_DIR}/epoch_{VAE_FULL_EPOCHS}.pt",
        catalog=f"{OUT_ROOT}/catalog/data_manifest.json",
    output:
        summary=f"{PERTURB_DIR}/vae2_only/summary.json",
        model=f"{PERTURB_DIR}/vae2_only/model.pt",
        history=f"{PERTURB_DIR}/vae2_only/history.tsv",
        slurm=f"{PERTURB_DIR}/vae2_only/slurm_job.json",
    params:
        submit=lambda wc: sbatch_wait(
            job_name="foundation-perturb-vae2-only",
            partition=RES.get("partition", "blackwell,a100"),
            runtime_min=RES.get("perturbation_runtime_min", RES.get("runtime_min", 720)),
            cpus=RES.get("cpus", 8),
            mem_mb=RES.get("perturbation_mem_mb", RES.get("mem_mb", 60000)),
            stdout="logs/foundation/perturbation_vae2_only.out",
            stderr="logs/foundation/perturbation_vae2_only.err",
            metadata=f"{PERTURB_DIR}/vae2_only/slurm_job.json",
            gres=RES.get("gres", "gpu:1"),
            qos=RES.get("qos"),
            command=perturbation_command(
                "vae2_only",
                "random",
                f"{PERTURB_DIR}/vae2_only",
            ),
        ),
    shell:
        "{params.submit}"


rule train_perturbation_staticdit:
    input:
        checkpoint=f"{VAE_FULL_DIR}/epoch_{VAE_FULL_EPOCHS}.pt",
        init=f"{DYNAMICS_STATIC_DIR}/model.pt",
        catalog=f"{OUT_ROOT}/catalog/data_manifest.json",
    output:
        summary=f"{PERTURB_DIR}/vae2_staticdit2/summary.json",
        model=f"{PERTURB_DIR}/vae2_staticdit2/model.pt",
        history=f"{PERTURB_DIR}/vae2_staticdit2/history.tsv",
        slurm=f"{PERTURB_DIR}/vae2_staticdit2/slurm_job.json",
    params:
        submit=lambda wc: sbatch_wait(
            job_name="foundation-perturb-staticdit",
            partition=RES.get("partition", "blackwell,a100"),
            runtime_min=RES.get("perturbation_runtime_min", RES.get("runtime_min", 720)),
            cpus=RES.get("cpus", 8),
            mem_mb=RES.get("perturbation_mem_mb", RES.get("mem_mb", 60000)),
            stdout="logs/foundation/perturbation_staticdit.out",
            stderr="logs/foundation/perturbation_staticdit.err",
            metadata=f"{PERTURB_DIR}/vae2_staticdit2/slurm_job.json",
            gres=RES.get("gres", "gpu:1"),
            qos=RES.get("qos"),
            command=perturbation_command(
                "vae2_staticdit2",
                "static_dit",
                f"{PERTURB_DIR}/vae2_staticdit2",
                init_checkpoint=f"{DYNAMICS_STATIC_DIR}/model.pt",
            ),
        ),
    shell:
        "{params.submit}"


rule train_perturbation_dynamicsdit:
    input:
        checkpoint=f"{VAE_FULL_DIR}/epoch_{VAE_FULL_EPOCHS}.pt",
        init=f"{DYNAMICS_TEMPORAL_DIR}/model.pt",
        catalog=f"{OUT_ROOT}/catalog/data_manifest.json",
    output:
        summary=f"{PERTURB_DIR}/vae2_dynamicsdit2/summary.json",
        model=f"{PERTURB_DIR}/vae2_dynamicsdit2/model.pt",
        history=f"{PERTURB_DIR}/vae2_dynamicsdit2/history.tsv",
        slurm=f"{PERTURB_DIR}/vae2_dynamicsdit2/slurm_job.json",
    params:
        submit=lambda wc: sbatch_wait(
            job_name="foundation-perturb-dynamicsdit",
            partition=RES.get("partition", "blackwell,a100"),
            runtime_min=RES.get("perturbation_runtime_min", RES.get("runtime_min", 720)),
            cpus=RES.get("cpus", 8),
            mem_mb=RES.get("perturbation_mem_mb", RES.get("mem_mb", 60000)),
            stdout="logs/foundation/perturbation_dynamicsdit.out",
            stderr="logs/foundation/perturbation_dynamicsdit.err",
            metadata=f"{PERTURB_DIR}/vae2_dynamicsdit2/slurm_job.json",
            gres=RES.get("gres", "gpu:1"),
            qos=RES.get("qos"),
            command=perturbation_command(
                "vae2_dynamicsdit2",
                "temporal_dynamics",
                f"{PERTURB_DIR}/vae2_dynamicsdit2",
                init_checkpoint=f"{DYNAMICS_TEMPORAL_DIR}/model.pt",
            ),
        ),
    shell:
        "{params.submit}"


rule run_perturbation_baselines:
    input:
        checkpoint=f"{VAE_FULL_DIR}/epoch_{VAE_FULL_EPOCHS}.pt",
        dynamics=f"{DYNAMICS_TEMPORAL_DIR}/model.pt",
        catalog=f"{OUT_ROOT}/catalog/data_manifest.json",
    output:
        summary=f"{PERTURB_BASELINE_DIR}/summary.json",
        table=f"{PERTURB_BASELINE_DIR}/baseline_summary.tsv",
        conditions=f"{PERTURB_BASELINE_DIR}/baseline_conditions.tsv",
        slurm=f"{PERTURB_BASELINE_DIR}/slurm_job.json",
    params:
        submit=lambda wc: sbatch_wait(
            job_name="foundation-perturb-baselines",
            partition=RES.get("partition", "blackwell,a100"),
            runtime_min=RES.get("perturbation_baseline_runtime_min", 120),
            cpus=RES.get("cpus", 8),
            mem_mb=RES.get("perturbation_mem_mb", RES.get("mem_mb", 60000)),
            stdout="logs/foundation/perturbation_baselines.out",
            stderr="logs/foundation/perturbation_baselines.err",
            metadata=f"{PERTURB_BASELINE_DIR}/slurm_job.json",
            gres=RES.get("gres", "gpu:1"),
            qos=RES.get("qos"),
            command=perturbation_baseline_command(),
        ),
    shell:
        "{params.submit}"
