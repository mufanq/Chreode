#!/usr/bin/env python
"""
CellStream 可视化脚本

生成:
1. Embedding 散点图 (按时间点着色)
2. 速度场可视化
3. 轨迹可视化
4. 生长动力学可视化

Usage:
    python visualize_cellstream.py \
        --data /path/to/data.csv \
        --model-dir /path/to/model \
        --output-dir /path/to/figures \
        --device cuda:0
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from datetime import datetime

# 添加 CellStream 路径
CELLSTREAM_PATH = Path(__file__).parent.parent.parent / "baseline" / "CellStream"
sys.path.insert(0, str(CELLSTREAM_PATH / "CellStream"))

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data_and_model(data_path: str, model_dir: str, device: str):
    """加载数据和模型"""
    from TranscriptomicsData import TranscriptomicsData
    from Net import Net_UOT
    from Autoencoder import Autoencoder
    from Train import load_params

    logger.info(f"加载数据: {data_path}")
    df = pd.read_csv(data_path)

    # 从数据推断名称
    name = Path(data_path).stem.replace('_normalized', '')

    data = TranscriptomicsData(
        name=name,
        values=df.values[:, 1:],
        labels=df.values[:, 0],
        device=device
    )

    # 创建模型
    latent_dim = 2
    net = Net_UOT(in_out_dim=latent_dim, hidden_dim=10, n_hiddens=4).to(device)
    autoencoder = Autoencoder(input_dim=data.dim + 1, hidden_dim=10, latent_dim=latent_dim).to(device)

    # 加载参数
    model_path = Path(model_dir)
    if model_path.is_dir():
        # 查找最新的模型
        param_dirs = list(model_path.glob(f"{name}"))
        if param_dirs:
            param_dir = param_dirs[0]
            # 查找 pth 文件
            pth_files = list(param_dir.glob("*.pth"))
            if pth_files:
                # 提取 name
                ae_file = [f for f in pth_files if 'AE_' in f.name]
                if ae_file:
                    model_name = ae_file[0].name.replace('AE_', '').replace('.pth', '')
                    load_params(data, net, autoencoder, path=str(model_path) + '/', name=model_name)
                    logger.info(f"加载模型: {model_path}/{name}/{model_name}")

    return data, net, autoencoder


def plot_embedding(data, autoencoder, output_dir: Path, title_suffix: str = ""):
    """绘制 embedding 散点图"""
    logger.info("绘制 embedding 散点图...")

    z = autoencoder.encoded(data.info).detach().cpu().numpy()
    labels = data.labels.cpu().numpy()

    unique_times = np.unique(labels)
    n_times = len(unique_times)

    # 使用颜色映射
    colors = cm.viridis(np.linspace(0, 1, n_times))

    fig, ax = plt.subplots(figsize=(10, 8))

    for i, t in enumerate(unique_times):
        mask = labels == t
        ax.scatter(z[mask, 0], z[mask, 1], c=[colors[i]], label=f'Time {int(t)}',
                   alpha=0.7, s=30, edgecolors='white', linewidth=0.5)

    ax.set_xlabel('Embedding Dim 1', fontsize=12)
    ax.set_ylabel('Embedding Dim 2', fontsize=12)
    ax.set_title(f'CellStream Embedding {title_suffix}', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f'embedding{title_suffix.replace(" ", "_")}.png', dpi=150)
    plt.close()

    logger.info(f"  保存到: {output_dir}/embedding{title_suffix.replace(' ', '_')}.png")


def plot_velocity_field(data, net, autoencoder, output_dir: Path, title_suffix: str = ""):
    """绘制速度场"""
    logger.info("绘制速度场...")

    z = autoencoder.encoded(data.info).detach().cpu().numpy()
    labels = data.labels.cpu().numpy()

    # 计算速度
    n_cells = int(data.mass.sum().item())
    z_tensor = autoencoder.encoded(data.info)

    velocities = []
    for i in range(n_cells):
        v = net.v(data.labels[i], z_tensor[i].unsqueeze(0)).squeeze(0)
        velocities.append(v.detach().cpu().numpy())
    velocities = np.array(velocities)

    unique_times = np.unique(labels)
    n_times = len(unique_times)
    colors = cm.viridis(np.linspace(0, 1, n_times))

    fig, ax = plt.subplots(figsize=(12, 10))

    # 绘制点
    for i, t in enumerate(unique_times):
        mask = labels == t
        ax.scatter(z[mask, 0], z[mask, 1], c=[colors[i]], label=f'Time {int(t)}',
                   alpha=0.5, s=20)

    # 绘制速度箭头 (采样)
    step = max(1, n_cells // 200)  # 最多绘制 200 个箭头
    ax.quiver(z[::step, 0], z[::step, 1],
              velocities[::step, 0], velocities[::step, 1],
              alpha=0.6, scale=15, width=0.003, color='red')

    ax.set_xlabel('Embedding Dim 1', fontsize=12)
    ax.set_ylabel('Embedding Dim 2', fontsize=12)
    ax.set_title(f'Velocity Field {title_suffix}', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f'velocity_field{title_suffix.replace(" ", "_")}.png', dpi=150)
    plt.close()

    logger.info(f"  保存到: {output_dir}/velocity_field{title_suffix.replace(' ', '_')}.png")


def plot_trajectories(data, net, autoencoder, output_dir: Path, num_trajectories: int = 30,
                      title_suffix: str = ""):
    """绘制轨迹"""
    logger.info("绘制轨迹...")
    from torchdiffeq import odeint

    z = autoencoder.encoded(data.info)
    labels = data.labels

    # 获取时间范围
    unique_times = torch.unique(labels)
    t_min, t_max = unique_times.min().item(), unique_times.max().item()

    # 随机选择起始点 (从第一个时间点)
    first_time_mask = labels == t_min
    first_time_indices = torch.where(first_time_mask)[0]

    if len(first_time_indices) < num_trajectories:
        selected_indices = first_time_indices
    else:
        perm = torch.randperm(len(first_time_indices))[:num_trajectories]
        selected_indices = first_time_indices[perm]

    z_start = z[selected_indices]

    # 积分轨迹
    t_span = torch.linspace(t_min, t_max, 50).to(z.device)

    trajectories = []
    for i in range(len(z_start)):
        z0 = z_start[i:i+1]
        try:
            traj = odeint(lambda t, z: net.v(t, z), z0, t_span, method='dopri5')
            trajectories.append(traj.squeeze(1).detach().cpu().numpy())
        except:
            pass

    # 绘制
    fig, ax = plt.subplots(figsize=(10, 8))

    z_np = z.detach().cpu().numpy()
    labels_np = labels.cpu().numpy()

    unique_times_np = np.unique(labels_np)
    n_times = len(unique_times_np)
    colors = cm.viridis(np.linspace(0, 1, n_times))

    # 绘制数据点
    for i, t in enumerate(unique_times_np):
        mask = labels_np == t
        ax.scatter(z_np[mask, 0], z_np[mask, 1], c=[colors[i]], alpha=0.3, s=10)

    # 绘制轨迹
    for traj in trajectories:
        ax.plot(traj[:, 0], traj[:, 1], 'r-', alpha=0.5, linewidth=1)
        ax.scatter(traj[0, 0], traj[0, 1], c='green', s=30, zorder=5)
        ax.scatter(traj[-1, 0], traj[-1, 1], c='red', s=30, zorder=5)

    ax.set_xlabel('Embedding Dim 1', fontsize=12)
    ax.set_ylabel('Embedding Dim 2', fontsize=12)
    ax.set_title(f'Cell Trajectories {title_suffix}', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f'trajectories{title_suffix.replace(" ", "_")}.png', dpi=150)
    plt.close()

    logger.info(f"  保存到: {output_dir}/trajectories{title_suffix.replace(' ', '_')}.png")


def plot_time_distribution(data, autoencoder, output_dir: Path, title_suffix: str = ""):
    """绘制各时间点在 embedding 空间的分布"""
    logger.info("绘制时间分布...")

    z = autoencoder.encoded(data.info).detach().cpu().numpy()
    labels = data.labels.cpu().numpy()

    unique_times = np.unique(labels)
    n_times = len(unique_times)

    # 计算每个时间点的中心
    centers = []
    for t in unique_times:
        mask = labels == t
        center = z[mask].mean(axis=0)
        centers.append(center)
    centers = np.array(centers)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左图: 各时间点的分布
    colors = cm.viridis(np.linspace(0, 1, n_times))
    for i, t in enumerate(unique_times):
        mask = labels == t
        axes[0].scatter(z[mask, 0], z[mask, 1], c=[colors[i]], label=f'Time {int(t)}',
                        alpha=0.5, s=20)

    # 绘制中心轨迹
    axes[0].plot(centers[:, 0], centers[:, 1], 'k--', linewidth=2, label='Center path')
    axes[0].scatter(centers[:, 0], centers[:, 1], c='black', s=100, zorder=5, marker='*')

    axes[0].set_xlabel('Embedding Dim 1', fontsize=12)
    axes[0].set_ylabel('Embedding Dim 2', fontsize=12)
    axes[0].set_title(f'Time Point Distribution {title_suffix}', fontsize=14)
    axes[0].legend(loc='best', fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # 右图: 各时间点的细胞数量
    cell_counts = [np.sum(labels == t) for t in unique_times]
    axes[1].bar(unique_times, cell_counts, color=colors, edgecolor='black')
    axes[1].set_xlabel('Time Point', fontsize=12)
    axes[1].set_ylabel('Number of Cells', fontsize=12)
    axes[1].set_title('Cells per Time Point', fontsize=14)
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / f'time_distribution{title_suffix.replace(" ", "_")}.png', dpi=150)
    plt.close()

    logger.info(f"  保存到: {output_dir}/time_distribution{title_suffix.replace(' ', '_')}.png")


def plot_metrics_comparison(results: dict, output_dir: Path):
    """绘制参数搜索结果对比图"""
    logger.info("绘制参数对比图...")

    if not results:
        logger.warning("没有结果可绘制")
        return

    names = list(results.keys())
    tc_values = [results[n]['TC'] for n in names]
    vc_values = [results[n]['VC'] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    x = np.arange(len(names))
    width = 0.35

    # TC
    bars1 = axes[0].bar(x, tc_values, width, color='steelblue', edgecolor='black')
    axes[0].set_xlabel('Experiment', fontsize=12)
    axes[0].set_ylabel('Temporal Consistency (TC)', fontsize=12)
    axes[0].set_title('TC Comparison', fontsize=14)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=45, ha='right')
    axes[0].axhline(y=0.99, color='r', linestyle='--', label='EMT Reference')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for bar, val in zip(bars1, tc_values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    # VC
    bars2 = axes[1].bar(x, vc_values, width, color='coral', edgecolor='black')
    axes[1].set_xlabel('Experiment', fontsize=12)
    axes[1].set_ylabel('Velocity Consistency (VC)', fontsize=12)
    axes[1].set_title('VC Comparison', fontsize=14)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=45, ha='right')
    axes[1].axhline(y=0.97, color='r', linestyle='--', label='EMT Reference')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars2, vc_values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_comparison.png', dpi=150)
    plt.close()

    logger.info(f"  保存到: {output_dir}/metrics_comparison.png")


def main():
    parser = argparse.ArgumentParser(description='CellStream 可视化')
    parser.add_argument('--data', type=str, required=True, help='数据文件路径')
    parser.add_argument('--model-dir', type=str, required=True, help='模型目录')
    parser.add_argument('--output-dir', type=str, required=True, help='输出目录')
    parser.add_argument('--device', type=str, default='cuda:0', help='计算设备')
    parser.add_argument('--title-suffix', type=str, default='', help='图标题后缀')

    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 设置设备
    if args.device.startswith('cuda') and torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device('cpu')

    # 加载数据和模型
    data, net, autoencoder = load_data_and_model(args.data, args.model_dir, str(device))

    # 生成可视化
    plot_embedding(data, autoencoder, output_dir, args.title_suffix)
    plot_velocity_field(data, net, autoencoder, output_dir, args.title_suffix)
    plot_trajectories(data, net, autoencoder, output_dir, title_suffix=args.title_suffix)
    plot_time_distribution(data, autoencoder, output_dir, args.title_suffix)

    logger.info("可视化完成!")


if __name__ == '__main__':
    main()
