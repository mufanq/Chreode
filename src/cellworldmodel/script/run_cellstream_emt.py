#!/usr/bin/env python
"""
CellStream EMT Tutorial 复现脚本
用于验证环境配置和复现论文结果

Usage:
    python run_cellstream_emt.py [--device cuda:4] [--train] [--eval-only]
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
import torch.nn as nn

# Logging: file handler is opt-in (set CHREODE_LOG_DIR if you want a log file).
# Importing this module from a fresh checkout must not require an `agent/`
# directory to exist on disk.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
_log_dir = os.environ.get("CHREODE_LOG_DIR")
if _log_dir:
    Path(_log_dir).mkdir(parents=True, exist_ok=True)
    _fh = logging.FileHandler(Path(_log_dir) / "run_emt.log")
    _fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(_fh)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")

def setup_device(device_str):
    """设置计算设备"""
    if device_str.startswith('cuda') and torch.cuda.is_available():
        device = torch.device(device_str)
        logger.info(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")
    return device

def load_data(device):
    """加载 EMT 数据"""
    from TranscriptomicsData import TranscriptomicsData

    df_path = CELLSTREAM_PATH / 'data' / 'EMT' / 'emt_normalized.csv'
    logger.info(f"Loading data from {df_path}")

    df = pd.read_csv(df_path)
    data = TranscriptomicsData(
        name='EMT',
        values=df.values[:, 1:],
        labels=df.values[:, 0],
        device=str(device)
    )
    data.abstract()

    logger.info(f"Data loaded: {int(data.mass.sum().item())} cells, {data.dim} features, {len(data.labels.unique())} time points")
    return data

def init_models(data, latent_dim=2, device='cpu'):
    """初始化模型"""
    from Net import Net_UOT
    from Autoencoder import Autoencoder
    from sklearn.decomposition import PCA

    logger.info("Initializing models...")

    net = Net_UOT(in_out_dim=latent_dim, hidden_dim=10, n_hiddens=4).to(device)
    autoencoder = Autoencoder(input_dim=data.dim+1, hidden_dim=10, latent_dim=latent_dim).to(device)

    # PCA 初始化
    logger.info("Initializing encoder with PCA...")
    pca = PCA(n_components=latent_dim, random_state=100)
    data_PCA = pca.fit_transform(data.values.cpu().detach().numpy())
    M = data_PCA.max(axis=0)
    m = data_PCA.min(axis=0)
    data_PCA = (data_PCA - m) / (M - m)

    # 训练 encoder
    for param in autoencoder.parameters():
        param.requires_grad = True

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)

    target = torch.tensor(data_PCA).type(torch.float32).to(device)
    for epoch in range(3000):
        optimizer.zero_grad()
        loss = criterion(target, autoencoder.encoded(data.info))
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 1000 == 0:
            logger.info(f'Encoder epoch {epoch+1}, Loss: {loss.item():.6f}')
        if loss.item() < 0.00005:
            break

    # 训练 decoder
    logger.info("Training decoder...")
    z = autoencoder.encoded(data.info).detach()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=3e-3)

    for epoch in range(10000):
        optimizer.zero_grad()
        loss = criterion(data.info, autoencoder.decoded(z))
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 2000 == 0:
            logger.info(f'Decoder epoch {epoch+1}, Loss: {loss.item():.6f}')
        if loss.item() < 0.001:
            break

    return net, autoencoder

def train_cellstream(data, net, autoencoder, num_refinements=3):
    """训练 CellStream 模型"""
    from Train import train

    time_indices_list = [(0, 1, 2, 3), (1, 2), (2, 3)]

    # 初始训练 Network
    logger.info("Training Network (initial)...")
    train(
        data, net, autoencoder, time_indices_list, train_mode='net',
        WFR_mode='local', lr=30e-4, iters=1000, para=(1, 5, 10, 0), para_v_g=(1, 1), output_gap=200
    )

    # 交替精炼
    for i in range(num_refinements):
        logger.info(f"Refinement round {i+1}/{num_refinements}...")

        # Refine Autoencoder
        logger.info("  Refining Autoencoder...")
        train(
            data, net, autoencoder, time_indices_list, train_mode='autoencoder',
            WFR_mode='local', lr=30e-4, iters=100, para=(1, 5, 10, 50), para_v_g=(1, 1), output_gap=50
        )

        # Refine Network
        logger.info("  Refining Network...")
        train(
            data, net, autoencoder, time_indices_list, train_mode='net',
            WFR_mode='local', lr=30e-4, iters=100, para=(1, 5, 10, 0), para_v_g=(1, 1), output_gap=50
        )

    logger.info("Training completed!")
    return net, autoencoder

def evaluate(data, net, autoencoder):
    """评估模型"""
    from Evaluation import temperal_consistency, velocity_consistency

    logger.info("Evaluating model...")

    z = autoencoder.encoded(data.info)

    # TC (Temporal Consistency)
    tc = temperal_consistency(z, data.labels, radius=0.05)
    tc_val = tc.item() if hasattr(tc, 'item') else float(tc)
    logger.info(f"Temporal Consistency (TC): {tc_val:.4f}")

    # VC (Velocity Consistency)
    velocity = torch.stack([
        net.v(data.labels[i], z[i].unsqueeze(0)).squeeze(0)
        for i in range(int(data.mass.sum().item()))
    ])
    vc = velocity_consistency(z, data.labels, velocity, radius=0.05)
    vc_val = float(vc)
    logger.info(f"Velocity Consistency (VC): {vc_val:.4f}")

    return {'TC': tc_val, 'VC': vc_val}

def save_results(results, output_path):
    """保存结果"""
    with open(output_path, 'a') as f:
        f.write(f"\n## EMT 复现结果 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- TC (Temporal Consistency): {results['TC']:.4f} (论文报告: 0.99)\n")
        f.write(f"- VC (Velocity Consistency): {results['VC']:.4f} (论文报告: 0.97)\n")
    logger.info(f"Results saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Run CellStream EMT Tutorial')
    parser.add_argument('--device', type=str, default='cuda:4', help='Device to use')
    parser.add_argument('--train', action='store_true', help='Train from scratch')
    parser.add_argument('--eval-only', action='store_true', help='Only evaluate pretrained model')
    parser.add_argument('--output', type=str, default=None, help='Output file for results')
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("CellStream EMT Tutorial - 环境验证与复现")
    logger.info("=" * 60)

    # 设置设备
    device = setup_device(args.device)

    # 加载数据
    data = load_data(device)

    # 初始化或加载模型
    from Net import Net_UOT
    from Autoencoder import Autoencoder

    latent_dim = 2
    net = Net_UOT(in_out_dim=latent_dim, hidden_dim=10, n_hiddens=4).to(device)
    autoencoder = Autoencoder(input_dim=data.dim+1, hidden_dim=10, latent_dim=latent_dim).to(device)

    if args.eval_only:
        # 加载预训练模型
        from Train import load_params
        path = str(CELLSTREAM_PATH / 'params') + '/'
        name = 'example'
        logger.info(f"Loading pretrained model from {path}{data.name}/{name}...")
        load_params(data, net, autoencoder, path=path, name=name)
    else:
        # 初始化并训练
        net, autoencoder = init_models(data, latent_dim, device)
        net, autoencoder = train_cellstream(data, net, autoencoder)

        # 保存模型
        from Train import save_params
        save_path = str(CELLSTREAM_PATH / 'params/')
        save_name = f'emt_reproduced_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        save_params(data, net, autoencoder, path=save_path, name=save_name)
        logger.info(f"Model saved to {save_path}{save_name}")

    # 评估
    results = evaluate(data, net, autoencoder)

    # 保存结果
    output_path = args.output or str(Path(__file__).parent.parent.parent / "agent" / "baseline.md")
    save_results(results, output_path)

    logger.info("=" * 60)
    logger.info("完成!")
    logger.info("=" * 60)

    return results

if __name__ == '__main__':
    main()
