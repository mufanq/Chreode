#!/usr/bin/env python
"""
CellStream 通用训练脚本

支持任意数据集的训练，包括:
- 自动数据加载
- 可配置的超参数
- 训练进度日志
- 模型保存和评估

Usage:
    python train_cellstream.py \
        --data /path/to/data.csv \
        --name my_dataset \
        --device cuda:4 \
        [--config /path/to/config.yaml]
"""

import sys
import os
import argparse
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# 添加 CellStream 路径
CELLSTREAM_PATH = Path(__file__).parent.parent.parent / "baseline" / "CellStream"
sys.path.insert(0, str(CELLSTREAM_PATH / "CellStream"))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA

# 设置日志
def setup_logging(log_file: Optional[str] = None):
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    return logging.getLogger(__name__)


# 默认配置
DEFAULT_CONFIG = {
    # 模型结构
    'latent_dim': 2,
    'hidden_dim': 10,
    'n_hiddens_net': 4,
    'n_hiddens_ae': 3,

    # 训练参数
    'init_encoder_epochs': 3000,
    'init_encoder_lr': 1e-3,
    'init_decoder_epochs': 10000,
    'init_decoder_lr': 3e-3,

    'net_train_iters': 1000,
    'net_train_lr': 30e-4,

    'refine_ae_iters': 100,
    'refine_net_iters': 100,
    'refine_lr': 30e-4,
    'num_refinements': 3,

    # Loss 权重
    'para_wfr': 1,
    'para_match': 5,
    'para_ae': 10,
    'para_ae_refine': 50,
    'para_v_g': (1, 1),

    # 评估参数
    'eval_radius': 0.05,
}


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """加载配置文件"""
    config = DEFAULT_CONFIG.copy()
    if config_path:
        import yaml
        with open(config_path, 'r') as f:
            user_config = yaml.safe_load(f)
        config.update(user_config)
    return config


def load_data(data_path: str, name: str, device: str):
    """加载数据"""
    from TranscriptomicsData import TranscriptomicsData

    logger = logging.getLogger(__name__)
    logger.info(f"加载数据: {data_path}")

    df = pd.read_csv(data_path)
    data = TranscriptomicsData(
        name=name,
        values=df.values[:, 1:],
        labels=df.values[:, 0],
        device=device
    )
    data.abstract()

    n_cells = int(data.mass.sum().item())
    n_times = len(data.labels.unique())
    logger.info(f"数据加载完成: {n_cells} cells, {data.dim} genes, {n_times} time points")

    return data


def init_autoencoder(data, config: Dict, device: str):
    """初始化 Autoencoder"""
    from Autoencoder import Autoencoder

    logger = logging.getLogger(__name__)
    latent_dim = config['latent_dim']

    autoencoder = Autoencoder(
        input_dim=data.dim + 1,
        hidden_dim=config['hidden_dim'],
        latent_dim=latent_dim
    ).to(device)

    # PCA 初始化
    logger.info("使用 PCA 初始化 encoder...")
    pca = PCA(n_components=latent_dim, random_state=100)
    data_PCA = pca.fit_transform(data.values.cpu().detach().numpy())
    M = data_PCA.max(axis=0)
    m = data_PCA.min(axis=0)
    data_PCA = (data_PCA - m) / (M - m)

    for param in autoencoder.parameters():
        param.requires_grad = True

    # 训练 encoder
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=config['init_encoder_lr'])
    target = torch.tensor(data_PCA).type(torch.float32).to(device)

    for epoch in range(config['init_encoder_epochs']):
        optimizer.zero_grad()
        loss = criterion(target, autoencoder.encoded(data.info))
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 1000 == 0:
            logger.info(f'  Encoder epoch {epoch+1}, Loss: {loss.item():.6f}')
        if loss.item() < 0.00005:
            break

    # 训练 decoder
    logger.info("训练 decoder...")
    z = autoencoder.encoded(data.info).detach()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=config['init_decoder_lr'])

    for epoch in range(config['init_decoder_epochs']):
        optimizer.zero_grad()
        loss = criterion(data.info, autoencoder.decoded(z))
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 2000 == 0:
            logger.info(f'  Decoder epoch {epoch+1}, Loss: {loss.item():.6f}')
        if loss.item() < 0.001:
            break

    return autoencoder


def create_time_indices_list(n_times: int):
    """
    创建时间索引列表用于训练

    包括:
    - 全部时间点
    - 相邻时间点对
    """
    time_indices_list = [tuple(range(n_times))]  # 全部时间点

    # 相邻时间点对
    for i in range(n_times - 1):
        time_indices_list.append((i, i + 1))

    return time_indices_list


def train_model(data, net, autoencoder, config: Dict):
    """训练模型"""
    from Train import train

    logger = logging.getLogger(__name__)

    n_times = len(data.labels.unique())
    time_indices_list = create_time_indices_list(n_times)
    logger.info(f"时间索引列表: {time_indices_list}")

    # 初始训练 Network
    logger.info("=" * 50)
    logger.info("阶段 1: 训练 Network (初始)")
    logger.info("=" * 50)

    train(
        data, net, autoencoder, time_indices_list, train_mode='net',
        WFR_mode='local', lr=config['net_train_lr'],
        iters=config['net_train_iters'],
        para=(config['para_wfr'], config['para_match'], config['para_ae'], 0),
        para_v_g=config['para_v_g'],
        output_gap=200
    )

    # 交替精炼
    for i in range(config['num_refinements']):
        logger.info("=" * 50)
        logger.info(f"阶段 2.{i+1}: 精炼 (第 {i+1}/{config['num_refinements']} 轮)")
        logger.info("=" * 50)

        # Refine Autoencoder
        logger.info("  精炼 Autoencoder...")
        train(
            data, net, autoencoder, time_indices_list, train_mode='autoencoder',
            WFR_mode='local', lr=config['refine_lr'],
            iters=config['refine_ae_iters'],
            para=(config['para_wfr'], config['para_match'], config['para_ae'], config['para_ae_refine']),
            para_v_g=config['para_v_g'],
            output_gap=50
        )

        # Refine Network
        logger.info("  精炼 Network...")
        train(
            data, net, autoencoder, time_indices_list, train_mode='net',
            WFR_mode='local', lr=config['refine_lr'],
            iters=config['refine_net_iters'],
            para=(config['para_wfr'], config['para_match'], config['para_ae'], 0),
            para_v_g=config['para_v_g'],
            output_gap=50
        )

    logger.info("训练完成!")
    return net, autoencoder


def evaluate_model(data, net, autoencoder, config: Dict) -> Dict[str, float]:
    """评估模型"""
    from Evaluation import temperal_consistency, velocity_consistency

    logger = logging.getLogger(__name__)
    logger.info("评估模型...")

    z = autoencoder.encoded(data.info)
    radius = config['eval_radius']

    # TC
    tc = temperal_consistency(z, data.labels, radius=radius)
    tc_val = tc.item() if hasattr(tc, 'item') else float(tc)

    # VC
    n_cells = int(data.mass.sum().item())
    velocity = torch.stack([
        net.v(data.labels[i], z[i].unsqueeze(0)).squeeze(0)
        for i in range(n_cells)
    ])
    vc = velocity_consistency(z, data.labels, velocity, radius=radius)
    vc_val = float(vc)

    logger.info(f"Temporal Consistency (TC): {tc_val:.4f}")
    logger.info(f"Velocity Consistency (VC): {vc_val:.4f}")

    return {'TC': tc_val, 'VC': vc_val}


def save_model(data, net, autoencoder, output_dir: str, name: str):
    """保存模型"""
    from Train import save_params

    logger = logging.getLogger(__name__)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    save_params(data, net, autoencoder, path=str(output_path) + '/', name=name)
    logger.info(f"模型保存到: {output_path}/{data.name}/")


def main():
    parser = argparse.ArgumentParser(description='CellStream 训练脚本')
    parser.add_argument('--data', type=str, required=True, help='数据文件路径 (CSV)')
    parser.add_argument('--name', type=str, required=True, help='数据集名称')
    parser.add_argument('--device', type=str, default='cuda:0', help='计算设备')
    parser.add_argument('--config', type=str, default=None, help='配置文件路径 (YAML)')
    parser.add_argument('--output-dir', type=str, default=None, help='输出目录')
    parser.add_argument('--log-file', type=str, default=None, help='日志文件路径')

    args = parser.parse_args()

    # 设置默认输出目录
    if args.output_dir is None:
        args.output_dir = str(Path(__file__).parent.parent.parent / "output" / "models")

    # 设置默认日志文件
    if args.log_file is None:
        args.log_file = str(Path(__file__).parent.parent.parent / "agent" / f"train_{args.name}.log")

    # 设置日志
    logger = setup_logging(args.log_file)

    logger.info("=" * 60)
    logger.info(f"CellStream 训练 - {args.name}")
    logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # 加载配置
    config = load_config(args.config)
    logger.info(f"配置: {json.dumps(config, indent=2, default=str)}")

    # 设置设备
    if args.device.startswith('cuda') and torch.cuda.is_available():
        device = torch.device(args.device)
        logger.info(f"使用 GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device('cpu')
        logger.info("使用 CPU")

    # 加载数据
    data = load_data(args.data, args.name, str(device))

    # 创建模型
    from Net import Net_UOT
    net = Net_UOT(
        in_out_dim=config['latent_dim'],
        hidden_dim=config['hidden_dim'],
        n_hiddens=config['n_hiddens_net']
    ).to(device)

    # 初始化 Autoencoder
    autoencoder = init_autoencoder(data, config, device)

    # 训练
    net, autoencoder = train_model(data, net, autoencoder, config)

    # 评估
    results = evaluate_model(data, net, autoencoder, config)

    # 保存模型
    model_name = f"trained_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_model(data, net, autoencoder, args.output_dir, model_name)

    # 保存结果到 JSON (用于参数搜索)
    results_json_path = Path(args.output_dir) / "results.json"
    with open(results_json_path, 'w') as f:
        json.dump({
            "TC": results['TC'],
            "VC": results['VC'],
            "name": args.name,
            "model_path": f"{args.output_dir}/{data.name}/{model_name}",
            "timestamp": datetime.now().isoformat(),
            "config": config
        }, f, indent=2, default=str)

    # 保存结果到 baseline.md
    baseline_path = Path(__file__).parent.parent.parent / "agent" / "baseline.md"
    with open(baseline_path, 'a') as f:
        f.write(f"\n## {args.name} 训练结果 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- TC (Temporal Consistency): {results['TC']:.4f}\n")
        f.write(f"- VC (Velocity Consistency): {results['VC']:.4f}\n")
        f.write(f"- 模型保存: {args.output_dir}/{data.name}/{model_name}\n")

    logger.info("=" * 60)
    logger.info(f"训练完成!")
    logger.info(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    return results


if __name__ == '__main__':
    main()
