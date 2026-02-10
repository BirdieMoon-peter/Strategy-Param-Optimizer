# -*- coding: utf-8 -*-
"""
期货配置模块
包含期货手续费类、品种配置和经纪商配置数据类

支持两种使用方式:
1. 内置品种配置: --asset-type futures --contract-code AG
2. 自定义JSON配置: --broker-config broker.json
"""

import json
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import backtrader as bt


# ==================== 经纪商配置数据类 ====================
@dataclass
class BrokerConfig:
    """
    经纪商配置（支持股票和期货）
    """
    asset_type: str = 'stock'        # 'stock' or 'futures'
    mult: int = 1                    # 合约乘数（期货专用）
    margin: float = 1.0              # 保证金比例（期货专用，stock默认1.0即无杠杆）
    comm_type: str = 'PERC'          # 手续费类型: 'FIXED' 或 'PERC'
    commission: float = 0.001        # 手续费（固定金额或费率）
    initial_cash: float = 100000.0   # 初始资金
    contract_code: str = ''          # 合约代码（如 'AG', 'SC'）
    contract_name: str = ''          # 合约名称（如 '白银', '原油'）

    @property
    def is_futures(self) -> bool:
        return self.asset_type.lower() == 'futures'


# ==================== 期货手续费类 ====================
class FutureCommInfoFixed(bt.CommInfoBase):
    """
    固定金额手续费类（适用于SC原油、AU黄金等）
    手续费 = 交易手数 × 每手手续费
    """
    params = (
        ('stocklike', False),
        ('commtype', bt.CommInfoBase.COMM_FIXED),
        ('commission', 20.0),
        ('mult', 1000),
        ('margin', 0.15),
    )

    def _getcommission(self, size, price, pseudoexec):
        return abs(size) * self.p.commission


class FutureCommInfoPerc(bt.CommInfoBase):
    """
    百分比手续费类（适用于AG白银、CU铜等）
    手续费 = 交易手数 × 价格 × 合约乘数 × 手续费率
    """
    params = (
        ('stocklike', False),
        ('commtype', bt.CommInfoBase.COMM_PERC),
        ('percabs', True),
        ('commission', 0.0001),
        ('mult', 1),
        ('margin', 0.1),
    )

    def _getcommission(self, size, price, pseudoexec):
        return abs(size) * price * self.p.mult * self.p.commission


# ==================== 内置期货品种配置 ====================
FUTURES_CONFIG = {
    'SC': {
        'name': '原油', 'exchange': 'INE',
        'mult': 1000, 'margin': 0.10,
        'comm_type': 'FIXED', 'commission': 20.0,
    },
    'AG': {
        'name': '白银', 'exchange': 'SHFE',
        'mult': 15, 'margin': 0.12,
        'comm_type': 'PERC', 'commission': 0.0005,
    },
    'AU': {
        'name': '黄金', 'exchange': 'SHFE',
        'mult': 1000, 'margin': 0.10,
        'comm_type': 'FIXED', 'commission': 10.0,
    },
    'CU': {
        'name': '铜', 'exchange': 'SHFE',
        'mult': 5, 'margin': 0.10,
        'comm_type': 'PERC', 'commission': 0.0001,
    },
    'RB': {
        'name': '螺纹钢', 'exchange': 'SHFE',
        'mult': 10, 'margin': 0.10,
        'comm_type': 'PERC', 'commission': 0.0001,
    },
    'IF': {
        'name': '沪深300股指', 'exchange': 'CFFEX',
        'mult': 300, 'margin': 0.12,
        'comm_type': 'PERC', 'commission': 0.000023,
    },
}


# ==================== 工厂函数 ====================
def build_broker_config(
    asset_type: str = 'stock',
    contract_code: str = None,
    broker_config_path: str = None,
    initial_cash: float = 100000.0
) -> BrokerConfig:
    """
    构建 BrokerConfig

    优先级: broker_config_path > contract_code > 默认stock

    Args:
        asset_type: 'stock' or 'futures'
        contract_code: 内置期货合约代码 (如 'AG')
        broker_config_path: 自定义 JSON 配置文件路径
        initial_cash: 初始资金

    Returns:
        BrokerConfig 实例
    """
    # 方式1: 从JSON文件加载（最高优先级）
    if broker_config_path:
        path = Path(broker_config_path)
        if not path.exists():
            raise FileNotFoundError(f"经纪商配置文件不存在: {broker_config_path}")

        with open(path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)

        cfg_asset = cfg.get('asset_type', 'futures')
        if cfg_asset == 'futures':
            required = ['mult', 'margin', 'comm_type', 'commission']
            missing = [k for k in required if k not in cfg]
            if missing:
                raise ValueError(
                    f"期货配置文件缺少必需字段: {missing}。"
                    f"必需字段: {required}"
                )

        return BrokerConfig(
            asset_type=cfg_asset,
            mult=cfg.get('mult', 1),
            margin=cfg.get('margin', 1.0),
            comm_type=cfg.get('comm_type', 'PERC').upper(),
            commission=cfg.get('commission', 0.001),
            initial_cash=cfg.get('initial_cash', initial_cash),
            contract_code=cfg.get('contract_code', ''),
            contract_name=cfg.get('contract_name', ''),
        )

    # 方式2: 从内置配置查找
    if asset_type.lower() == 'futures' and contract_code:
        code_upper = contract_code.upper()
        if code_upper not in FUTURES_CONFIG:
            supported = ', '.join(sorted(FUTURES_CONFIG.keys()))
            raise ValueError(
                f"不支持的合约代码: '{contract_code}'。"
                f"支持的合约: {supported}。"
                f"如需自定义品种，请使用 --broker-config 指定JSON文件。"
            )

        fc = FUTURES_CONFIG[code_upper]
        return BrokerConfig(
            asset_type='futures',
            mult=fc['mult'],
            margin=fc['margin'],
            comm_type=fc['comm_type'],
            commission=fc['commission'],
            initial_cash=initial_cash,
            contract_code=code_upper,
            contract_name=fc['name'],
        )

    # 方式3: 默认股票配置
    return BrokerConfig(
        asset_type='stock',
        commission=0.001,
        initial_cash=initial_cash,
    )


def create_commission_info(broker_config: BrokerConfig) -> Optional[bt.CommInfoBase]:
    """
    根据 BrokerConfig 创建 backtrader CommInfoBase 实例

    Returns:
        bt.CommInfoBase 实例（期货），或 None（股票）
    """
    if not broker_config.is_futures:
        return None

    if broker_config.comm_type.upper() == 'FIXED':
        return FutureCommInfoFixed(
            commission=broker_config.commission,
            mult=broker_config.mult,
            margin=broker_config.margin,
        )
    else:
        return FutureCommInfoPerc(
            commission=broker_config.commission,
            mult=broker_config.mult,
            margin=broker_config.margin,
        )