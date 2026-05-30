#!/usr/bin/env python
"""
CellStream 参数搜索脚本

并行运行多组参数实验，比较结果

Usage:
    python param_search_cellstream.py \
        --data /path/to/data.csv \
        --output-dir /path/to/output \
        --gpus 4 5 6 7
"""

import sys
import os
import argparse
import logging
import json
import subprocess
import time
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 参数搜索空间
PARAM_SEARCH_SPACE = {
    # 实验1: 默认参数 (baseline)
    "default": {
        "description": "默认参数",
        "config": {}
    },

    # 实验2: 增加训练迭代次数
    "more_iters": {
        "description": "增加训练迭代次数 (2000 iters)",
        "config": {
            "net_train_iters": 2000,
            "num_refinements": 5,
            "refine_ae_iters": 200,
            "refine_net_iters": 200
        }
    },

    # 实验3: 更大的网络
    "larger_net": {
        "description": "更大的网络结构 (hidden=20)",
        "config": {
            "hidden_dim": 20,
            "n_hiddens_net": 5,
            "n_hiddens_ae": 4
        }
    },

    # 实验4: 调整 Loss 权重 - 降低 AE 权重
    "lower_ae_weight": {
        "description": "降低 AE Loss 权重",
        "config": {
            "para_ae": 5,
            "para_ae_refine": 25
        }
    },

    # 实验5: 调整 Loss 权重 - 增加 Match 权重
    "higher_match": {
        "description": "增加 Match Loss 权重",
        "config": {
            "para_match": 10,
            "para_wfr": 2
        }
    },

    # 实验6: 综合优化参数
    "optimized": {
        "description": "综合优化 (更多迭代 + 更大网络)",
        "config": {
            "net_train_iters": 2000,
            "num_refinements": 5,
            "refine_ae_iters": 200,
            "refine_net_iters": 200,
            "hidden_dim": 15,
            "n_hiddens_net": 5
        }
    },

    # 实验7: 高变基因选择 (需要在数据预处理阶段完成)
    # 这里只调整模型参数
    "high_lr": {
        "description": "更高学习率",
        "config": {
            "net_train_lr": 0.01,
            "refine_lr": 0.01
        }
    },

    # 实验8: 低学习率 + 更多迭代
    "low_lr_more_iters": {
        "description": "低学习率 + 更多迭代",
        "config": {
            "net_train_iters": 3000,
            "num_refinements": 6,
            "net_train_lr": 0.001,
            "refine_lr": 0.001
        }
    }
}


def run_single_experiment(exp_name: str, exp_config: dict, data_path: str,
                          output_base: str, gpu_id: int) -> dict:
    """运行单个实验"""
    from datetime import datetime

    output_dir = Path(output_base) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # 写入配置文件
    config_path = output_dir / "config.yaml"

    # 基础配置
    base_config = {
        "latent_dim": 2,
        "hidden_dim": 10,
        "n_hiddens_net": 4,
        "n_hiddens_ae": 3,
        "init_encoder_epochs": 3000,
        "init_encoder_lr": 0.001,
        "init_decoder_epochs": 10000,
        "init_decoder_lr": 0.003,
        "net_train_iters": 1000,
        "net_train_lr": 0.003,
        "refine_ae_iters": 100,
        "refine_net_iters": 100,
        "refine_lr": 0.003,
        "num_refinements": 3,
        "para_wfr": 1,
        "para_match": 5,
        "para_ae": 10,
        "para_ae_refine": 50,
        "para_v_g": [1, 1],
        "eval_radius": 0.05
    }

    # 更新配置
    config = {**base_config, **exp_config.get("config", {})}

    import yaml
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    # 构建命令
    script_path = Path(__file__).parent / "train_cellstream.py"
    cmd = [
        sys.executable, str(script_path),
        "--data", data_path,
        "--output-dir", str(output_dir),
        "--config", str(config_path),
        "--device", f"cuda:{gpu_id}"
    ]

    # 运行
    log_path = output_dir / "train.log"
    start_time = time.time()

    logger.info(f"[GPU {gpu_id}] 开始实验: {exp_name} - {exp_config.get('description', '')}")

    try:
        with open(log_path, 'w') as log_file:
            result = subprocess.run(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                timeout=3600,  # 1小时超时
                cwd=str(Path(__file__).parent.parent.parent)  # 工作目录
            )

        elapsed = time.time() - start_time

        # 读取结果
        result_path = output_dir / "results.json"
        if result_path.exists():
            with open(result_path) as f:
                metrics = json.load(f)
        else:
            # 从日志中提取
            metrics = extract_metrics_from_log(log_path)

        logger.info(f"[GPU {gpu_id}] 完成实验: {exp_name} - TC={metrics.get('TC', 'N/A'):.4f}, VC={metrics.get('VC', 'N/A'):.4f} ({elapsed/60:.1f}min)")

        return {
            "name": exp_name,
            "description": exp_config.get("description", ""),
            "config": config,
            "metrics": metrics,
            "elapsed_seconds": elapsed,
            "status": "success"
        }

    except subprocess.TimeoutExpired:
        logger.error(f"[GPU {gpu_id}] 实验超时: {exp_name}")
        return {
            "name": exp_name,
            "description": exp_config.get("description", ""),
            "config": config,
            "metrics": {},
            "status": "timeout"
        }
    except Exception as e:
        logger.error(f"[GPU {gpu_id}] 实验失败: {exp_name} - {e}")
        return {
            "name": exp_name,
            "description": exp_config.get("description", ""),
            "config": config,
            "metrics": {},
            "status": f"error: {e}"
        }


def extract_metrics_from_log(log_path: Path) -> dict:
    """从日志文件提取指标"""
    metrics = {}
    try:
        with open(log_path) as f:
            for line in f:
                if "Temporal Consistency (TC):" in line:
                    metrics["TC"] = float(line.split(":")[-1].strip())
                elif "Velocity Consistency (VC):" in line:
                    metrics["VC"] = float(line.split(":")[-1].strip())
    except:
        pass
    return metrics


def run_experiments_parallel(data_path: str, output_dir: str, gpus: list,
                             experiments: list = None):
    """并行运行多个实验"""
    if experiments is None:
        experiments = list(PARAM_SEARCH_SPACE.keys())

    output_base = Path(output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    results = []
    gpu_pool = gpus.copy()

    # 分配 GPU
    exp_gpu_pairs = []
    for i, exp_name in enumerate(experiments):
        gpu_id = gpu_pool[i % len(gpu_pool)]
        exp_gpu_pairs.append((exp_name, gpu_id))

    logger.info(f"开始参数搜索: {len(experiments)} 个实验, 使用 GPU: {gpus}")

    # 并行运行
    with ProcessPoolExecutor(max_workers=len(gpus)) as executor:
        futures = {}
        for exp_name, gpu_id in exp_gpu_pairs:
            exp_config = PARAM_SEARCH_SPACE[exp_name]
            future = executor.submit(
                run_single_experiment,
                exp_name, exp_config, data_path, str(output_base), gpu_id
            )
            futures[future] = exp_name

        for future in as_completed(futures):
            exp_name = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"实验 {exp_name} 出错: {e}")
                results.append({
                    "name": exp_name,
                    "status": f"error: {e}"
                })

    return results


def summarize_results(results: list, output_dir: Path):
    """汇总并比较结果"""
    logger.info("\n" + "="*60)
    logger.info("参数搜索结果汇总")
    logger.info("="*60)

    # 按 TC+VC 排序
    valid_results = [r for r in results if r.get("status") == "success" and r.get("metrics")]
    valid_results.sort(key=lambda x: x["metrics"].get("TC", 0) + x["metrics"].get("VC", 0), reverse=True)

    # 打印表格
    logger.info(f"\n{'实验名称':<20} {'TC':<10} {'VC':<10} {'TC+VC':<10} {'耗时(min)':<10} {'描述'}")
    logger.info("-"*80)

    for r in valid_results:
        tc = r["metrics"].get("TC", 0)
        vc = r["metrics"].get("VC", 0)
        elapsed = r.get("elapsed_seconds", 0) / 60
        logger.info(f"{r['name']:<20} {tc:<10.4f} {vc:<10.4f} {tc+vc:<10.4f} {elapsed:<10.1f} {r.get('description', '')[:30]}")

    # 最优结果
    if valid_results:
        best = valid_results[0]
        logger.info(f"\n最优结果: {best['name']}")
        logger.info(f"  TC = {best['metrics'].get('TC', 0):.4f}")
        logger.info(f"  VC = {best['metrics'].get('VC', 0):.4f}")
        logger.info(f"  配置: {json.dumps(best.get('config', {}), indent=2)}")

    # 保存汇总
    summary_path = output_dir / "param_search_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "best": valid_results[0] if valid_results else None
        }, f, indent=2)

    logger.info(f"\n结果保存到: {summary_path}")

    return valid_results


def main():
    parser = argparse.ArgumentParser(description='CellStream 参数搜索')
    parser.add_argument('--data', type=str, required=True, help='数据文件路径')
    parser.add_argument('--output-dir', type=str, required=True, help='输出目录')
    parser.add_argument('--gpus', type=int, nargs='+', default=[4, 5, 6, 7], help='可用 GPU 列表')
    parser.add_argument('--experiments', type=str, nargs='*', default=None,
                        help='要运行的实验列表 (默认全部)')

    args = parser.parse_args()

    # 运行实验
    results = run_experiments_parallel(
        args.data,
        args.output_dir,
        args.gpus,
        args.experiments
    )

    # 汇总结果
    output_dir = Path(args.output_dir)
    summarize_results(results, output_dir)

    logger.info("\n参数搜索完成!")


if __name__ == '__main__':
    main()
