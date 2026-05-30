#!/usr/bin/env python
"""
将 h5ad 格式数据转换为 CellStream 格式

CellStream 输入格式:
- CSV 文件
- 第一列 'samples' (或 'time'): 时间标签 (0, 1, 2, ...)
- 其余列: 特征 (基因表达值), 归一化到 [0.05, 0.95]

Usage:
    python convert_h5ad_to_cellstream.py \
        --input-dir /path/to/h5ad/files \
        --output /path/to/output.csv \
        [--time-col time_of_sampling] \
        [--normalize] \
        [--select-genes gene1,gene2,...]
"""

import os
import re
import argparse
import logging
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
import anndata as ad

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_time_label(label: str) -> float:
    """
    将时间标签转换为数值

    支持格式:
    - Day0, Day1, Day1.5, ... -> 0, 1, 1.5, ...
    - d0, d1, d1.5, ... -> 0, 1, 1.5, ...
    - 0h, 8h, 24h, ... -> 0, 8, 24, ...
    - 纯数字: 0, 1, 2, ...
    """
    label = str(label).strip()

    # Day0, Day1.5 格式
    match = re.match(r'[Dd]ay?(\d+\.?\d*)', label)
    if match:
        return float(match.group(1))

    # 小时格式: 0h, 8h, 24h
    match = re.match(r'(\d+\.?\d*)h', label)
    if match:
        return float(match.group(1))

    # 纯数字
    try:
        return float(label)
    except ValueError:
        pass

    raise ValueError(f"无法解析时间标签: {label}")


def normalize_to_range(X: np.ndarray, low: float = 0.05, high: float = 0.95) -> np.ndarray:
    """
    将数据归一化到 [low, high] 范围
    对每个特征单独进行归一化
    """
    X_min = X.min(axis=0, keepdims=True)
    X_max = X.max(axis=0, keepdims=True)

    # 避免除以零
    X_range = X_max - X_min
    X_range[X_range == 0] = 1

    # 归一化到 [0, 1]
    X_norm = (X - X_min) / X_range

    # 缩放到 [low, high]
    X_scaled = X_norm * (high - low) + low

    return X_scaled


def load_h5ad_files(
    input_dir: str,
    time_col: str = 'time_of_sampling',
    pattern: str = '*.h5ad'
) -> tuple[np.ndarray, np.ndarray, List[str]]:
    """
    加载目录下所有 h5ad 文件并合并

    Returns:
        X: 表达矩阵 (cells x genes)
        time_labels: 时间标签数组 (cells,)
        gene_names: 基因名称列表
    """
    input_path = Path(input_dir)
    h5ad_files = sorted(input_path.glob(pattern))

    if not h5ad_files:
        raise FileNotFoundError(f"未找到 h5ad 文件: {input_dir}/{pattern}")

    logger.info(f"找到 {len(h5ad_files)} 个 h5ad 文件")

    all_X = []
    all_times = []
    gene_names = None

    # 收集所有时间点及其数值
    time_mapping: Dict[str, float] = {}

    for h5ad_file in h5ad_files:
        adata = ad.read_h5ad(h5ad_file)
        logger.info(f"  加载 {h5ad_file.name}: {adata.shape[0]} cells x {adata.shape[1]} genes")

        # 获取基因名称 (第一个文件)
        if gene_names is None:
            gene_names = list(adata.var_names)
        else:
            # 检查基因顺序一致性
            current_genes = list(adata.var_names)
            if current_genes != gene_names:
                logger.warning(f"基因顺序不一致: {h5ad_file.name}")
                # 重新排序
                adata = adata[:, gene_names]

        # 获取时间标签
        if time_col in adata.obs.columns:
            time_labels = adata.obs[time_col].values
        else:
            # 从文件名推断
            match = re.search(r'day[_]?([Dd]ay?\d+\.?\d*)', h5ad_file.name)
            if match:
                time_label = match.group(1)
                time_labels = np.array([time_label] * adata.shape[0])
            else:
                raise ValueError(f"无法确定时间标签: {h5ad_file.name}")

        # 解析时间数值
        for label in np.unique(time_labels):
            if label not in time_mapping:
                time_mapping[label] = parse_time_label(label)

        # 转换为数值
        time_values = np.array([time_mapping[str(t)] for t in time_labels])

        all_X.append(adata.X)
        all_times.append(time_values)

    # 合并
    X = np.vstack(all_X)
    time_labels = np.concatenate(all_times)

    logger.info(f"合并后: {X.shape[0]} cells x {X.shape[1]} genes")
    logger.info(f"时间点: {sorted(np.unique(time_labels))}")

    return X, time_labels, gene_names


def select_variable_genes(
    X: np.ndarray,
    gene_names: List[str],
    n_top_genes: int = 50,
    method: str = 'variance'
) -> tuple[np.ndarray, List[str]]:
    """
    选择高变基因

    Args:
        method: 'variance' 或 'cv' (coefficient of variation)
    """
    if method == 'variance':
        scores = X.var(axis=0)
    elif method == 'cv':
        means = X.mean(axis=0)
        stds = X.std(axis=0)
        scores = stds / (means + 1e-6)
    else:
        raise ValueError(f"未知方法: {method}")

    # 选择 top n 基因
    top_indices = np.argsort(scores)[::-1][:n_top_genes]
    top_indices = np.sort(top_indices)  # 保持原始顺序

    X_selected = X[:, top_indices]
    selected_genes = [gene_names[i] for i in top_indices]

    logger.info(f"选择了 {len(selected_genes)} 个高变基因")

    return X_selected, selected_genes


def convert_to_cellstream_format(
    X: np.ndarray,
    time_labels: np.ndarray,
    gene_names: List[str],
    output_path: str,
    normalize: bool = True,
    norm_low: float = 0.05,
    norm_high: float = 0.95
) -> pd.DataFrame:
    """
    转换为 CellStream 格式并保存
    """
    # 归一化
    if normalize:
        logger.info(f"归一化到 [{norm_low}, {norm_high}]")
        X = normalize_to_range(X, norm_low, norm_high)

    # 将时间标签转换为整数索引 (CellStream 期望)
    unique_times = sorted(np.unique(time_labels))
    time_to_idx = {t: i for i, t in enumerate(unique_times)}
    time_indices = np.array([time_to_idx[t] for t in time_labels])

    logger.info(f"时间点映射: {time_to_idx}")

    # 创建 DataFrame
    df = pd.DataFrame(X, columns=gene_names)
    df.insert(0, 'samples', time_indices)

    # 保存
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info(f"保存到: {output_path}")
    logger.info(f"数据形状: {df.shape}")

    # 保存元数据
    meta_path = output_path.with_suffix('.meta.json')
    import json
    metadata = {
        'time_mapping': time_to_idx,
        'gene_names': gene_names,
        'n_cells': int(df.shape[0]),
        'n_genes': int(df.shape[1] - 1),
        'n_timepoints': len(unique_times),
        'cells_per_timepoint': {str(t): int((time_labels == t).sum()) for t in unique_times},
        'normalized': normalize,
        'norm_range': [norm_low, norm_high] if normalize else None
    }
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"元数据保存到: {meta_path}")

    return df


def main():
    parser = argparse.ArgumentParser(description='将 h5ad 格式转换为 CellStream 格式')
    parser.add_argument('--input-dir', type=str, required=True, help='h5ad 文件目录')
    parser.add_argument('--output', type=str, required=True, help='输出 CSV 文件路径')
    parser.add_argument('--time-col', type=str, default='time_of_sampling', help='时间列名')
    parser.add_argument('--normalize', action='store_true', help='是否归一化')
    parser.add_argument('--norm-low', type=float, default=0.05, help='归一化下界')
    parser.add_argument('--norm-high', type=float, default=0.95, help='归一化上界')
    parser.add_argument('--select-genes', type=str, default=None, help='选择的基因 (逗号分隔)')
    parser.add_argument('--n-top-genes', type=int, default=None, help='选择 top N 高变基因')
    parser.add_argument('--gene-selection-method', type=str, default='variance',
                        choices=['variance', 'cv'], help='基因选择方法')

    args = parser.parse_args()

    # 加载数据
    X, time_labels, gene_names = load_h5ad_files(args.input_dir, args.time_col)

    # 基因选择
    if args.select_genes:
        selected = args.select_genes.split(',')
        indices = [gene_names.index(g) for g in selected if g in gene_names]
        X = X[:, indices]
        gene_names = [gene_names[i] for i in indices]
        logger.info(f"选择了 {len(gene_names)} 个指定基因")
    elif args.n_top_genes:
        X, gene_names = select_variable_genes(
            X, gene_names, args.n_top_genes, args.gene_selection_method
        )

    # 转换并保存
    convert_to_cellstream_format(
        X, time_labels, gene_names, args.output,
        normalize=args.normalize,
        norm_low=args.norm_low,
        norm_high=args.norm_high
    )

    logger.info("转换完成!")


if __name__ == '__main__':
    main()
