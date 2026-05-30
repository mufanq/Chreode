from argparse import Namespace

from cellworldmodel.benchmark.config_overrides import apply_common_overrides
from cellworldmodel.benchmark.experiment_registry import get_experiment_spec


def test_g2a_recipe_sets_selected_configuration():
    cfg = {}
    spec = get_experiment_spec("g2a_m10_md_adamw")
    spec.apply_to_cfg(cfg)
    assert spec.method == "m10"
    assert spec.epochs == 5000
    assert cfg["batch_size"] == 512
    assert cfg["K"] == 8
    assert cfg["multi_delta"] is True
    assert cfg["optimizer"] == "adamw"
    assert cfg["lr_schedule"] == "warmup_cosine"
    assert cfg["split_policy"] == "per_timepoint"


def test_common_overrides_only_apply_explicit_values():
    args = Namespace(
        hidden_dim=None,
        n_layers=None,
        noise_dim=None,
        batch_size=16,
        K=None,
        lr=None,
        lambda_drift=None,
        lambda_down=None,
        state_chunk_dim=None,
        learned_state_tokens=None,
        disable_rope=False,
        dit_size=None,
        waddington_dit=False,
        curl_rank=None,
        wdit_curl_update=None,
        wdit_curl_time_mode=None,
        wdit_hybrid_delta0=None,
        wdit_hybrid_slope=None,
        wdit_hard_delta0=None,
        wdit_time_embedding=None,
        wdit_time_delta_transform=None,
        wdit_time_delta_scale=None,
        wdit_curl_time_embedding=None,
        wdit_curl_time_delta_transform=None,
        wdit_curl_time_delta_scale=None,
        loss_balancer=None,
        loss_balancer_temperature=None,
        loss_balancer_lookback_prob=None,
        loss_balancer_alpha=None,
        loss_balancer_max_multiplier=None,
        lambda_wdit_a_fro=None,
        lambda_wdit_curl=None,
        drift_pos_ratio=None,
        drift_balance_sample_counts=False,
        optimizer="adamw",
        weight_decay=None,
        lr_schedule="none",
        warmup_frac=None,
        ema_decay=None,
        down_n_mc=None,
        down_antithetic=False,
        multi_delta=False,
        md_endpoint_prob=None,
        split_policy=None,
    )
    cfg = {"batch_size": 512, "optimizer": "adam"}
    apply_common_overrides(args, cfg)
    assert cfg["batch_size"] == 16
    assert cfg["optimizer"] == "adamw"
    assert "lr_schedule" not in cfg


def test_waddington_dit_recipe_sets_explicit_residual():
    cfg = {}
    spec = get_experiment_spec("g2a_m10_wdit_adamw")
    spec.apply_to_cfg(cfg)
    assert spec.method == "m10"
    assert cfg["waddington_dit"] is True
    assert cfg["curl_rank"] == 16
    assert cfg["multi_delta"] is True
    assert cfg["optimizer"] == "adamw"


def test_waddington_dit_cayley_recipes_set_update_modes():
    cfg = {}
    spec = get_experiment_spec("g2a_m10_wdit_cayley_direct_adamw")
    spec.apply_to_cfg(cfg)
    assert cfg["waddington_dit"] is True
    assert cfg["wdit_curl_update"] == "cayley_direct"
    cfg = {}
    get_experiment_spec("g2a_m10_wdit_cayley_residual_adamw").apply_to_cfg(cfg)
    assert cfg["wdit_curl_update"] == "cayley_residual"


def test_waddington_dit_new_fix_recipes():
    cfg = {}
    get_experiment_spec("g2a_m10_wdit_hybrid_delta_adamw").apply_to_cfg(cfg)
    assert cfg["wdit_curl_update"] == "hybrid_delta"
    assert cfg["wdit_hybrid_delta0"] == 36.0
    cfg = {}
    get_experiment_spec("g2a_m10_wdit_curlpen1e5_adamw").apply_to_cfg(cfg)
    assert cfg["lambda_wdit_curl"] == 1e-5
    cfg = {}
    get_experiment_spec("g2a_m10_wdit_statecurl_adamw").apply_to_cfg(cfg)
    assert cfg["wdit_curl_time_mode"] == "state_only"


def test_waddington_dit_statecurl_cayley_residual_recipe():
    cfg = {}
    get_experiment_spec("g2a_m10_wdit_statecurl_cayley_residual_adamw").apply_to_cfg(cfg)
    assert cfg["waddington_dit"] is True
    assert cfg["wdit_curl_update"] == "cayley_residual"
    assert cfg["wdit_curl_time_mode"] == "state_only"
    assert cfg["multi_delta"] is True


def test_waddington_dit_time_embedding_recipes():
    cfg = {}
    get_experiment_spec("g2a_m10_wdit_lowfreqtime_adamw").apply_to_cfg(cfg)
    assert cfg["wdit_time_embedding"] == "bounded_lowfreq_fourier"
    assert cfg["wdit_time_delta_transform"] == "normalized"
    assert cfg["wdit_curl_update"] == "additive"
    cfg = {}
    get_experiment_spec("g2a_m10_wdit_time2vec_adamw").apply_to_cfg(cfg)
    assert cfg["wdit_time_embedding"] == "time2vec"
    cfg = {}
    get_experiment_spec("g2a_m10_wdit_statecurl_cayley_lowfreqtime_adamw").apply_to_cfg(cfg)
    assert cfg["wdit_time_embedding"] == "bounded_lowfreq_fourier"
    assert cfg["wdit_curl_update"] == "cayley_residual"
    assert cfg["wdit_curl_time_mode"] == "state_only"
    cfg = {}
    get_experiment_spec("g2a_m10_wdit_statecurl_cayley_time2vec_adamw").apply_to_cfg(cfg)
    assert cfg["wdit_time_embedding"] == "time2vec"
    assert cfg["wdit_curl_update"] == "cayley_residual"
    assert cfg["wdit_curl_time_mode"] == "state_only"


def test_waddington_dit_long_delta_and_branch_time_recipes():
    cfg = {}
    get_experiment_spec("g2a_m10_wdit_lowfreq_hardcayley_adamw").apply_to_cfg(cfg)
    assert cfg["wdit_curl_update"] == "hard_delta_cayley_residual"
    assert cfg["wdit_hard_delta0"] == 30.0
    assert cfg["wdit_time_embedding"] == "bounded_lowfreq_fourier"
    cfg = {}
    get_experiment_spec("g2a_m10_wdit_legacyu_lowfreqcurl_adamw").apply_to_cfg(cfg)
    assert cfg["wdit_curl_time_mode"] == "separate"
    assert cfg["wdit_time_embedding"] == "legacy_fourier"
    assert cfg["wdit_curl_time_embedding"] == "bounded_lowfreq_fourier"
    cfg = {}
    get_experiment_spec("g2a_m10_wdit_time2vecu_lowfreqcurl_adamw").apply_to_cfg(cfg)
    assert cfg["wdit_curl_time_mode"] == "separate"
    assert cfg["wdit_time_embedding"] == "time2vec"
    assert cfg["wdit_curl_time_embedding"] == "bounded_lowfreq_fourier"
    cfg = {}
    get_experiment_spec("g2a_m10_wdit_cayley_time2vecu_lowfreqcurl_adamw").apply_to_cfg(cfg)
    assert cfg["wdit_curl_update"] == "cayley_residual"
    assert cfg["wdit_curl_time_mode"] == "separate"
    assert cfg["wdit_time_embedding"] == "time2vec"
    assert cfg["wdit_curl_time_embedding"] == "bounded_lowfreq_fourier"


def test_selected_wdit_loss_balancer_recipes():
    expected = {
        "g2a_m10_wdit_time2vecu_lowfreqcurl_uncertainty_adamw": "uncertainty",
        "g2a_m10_wdit_time2vecu_lowfreqcurl_relobralo_adamw": "relobralo",
        "g2a_m10_wdit_time2vecu_lowfreqcurl_dwa_adamw": "dwa",
        "g2a_m10_wdit_time2vecu_lowfreqcurl_gradnormlite_adamw": "gradnorm_lite",
        "g2a_m10_wdit_time2vecu_lowfreqcurl_rlw_adamw": "rlw",
    }
    for name, balancer in expected.items():
        cfg = {}
        get_experiment_spec(name).apply_to_cfg(cfg)
        assert cfg["loss_balancer"] == balancer
        assert cfg["wdit_curl_time_mode"] == "separate"
        assert cfg["wdit_time_embedding"] == "time2vec"
        assert cfg["wdit_curl_time_embedding"] == "bounded_lowfreq_fourier"
        assert cfg["multi_delta"] is True
