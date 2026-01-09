# 导入必要模块
import qlib
from qlib.constant import REG_CN
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.processor import DropnaLabel, CSZScoreNorm, DropnaProcessor
from qlib.contrib.data.loader import QlibDataLoader
import pandas as pd
import traceback

def setup_qlib():
    """Initializes Qlib."""
    try:
        qlib.init(provider_uri="~/.qlib/qlib_data/cn_etf_data", region=REG_CN)
        print("Qlib初始化成功，使用路径: ~/.qlib/qlib_data/cn_etf_data")
        return True
    except Exception as e:
        print(f"Qlib初始化失败: {e}")
        print("请确保您已经按照GEMINI.md中的步骤正确转换了ETF数据。")
        return False

# 自定义DataHandler
class ETFDataHandler(DataHandlerLP):
    def __init__(self, instruments, start_time, end_time, **kwargs):
        data_loader_config = {
            "feature": [
                "$close", "$volume", "Ref($close, 1)", "Mean($volume, 20)",
                "Std($close / Ref($close, 1) - 1, 20)",
            ],
            "label": ["Ref($close, -1) / $close - 1"],
        }
        data_loader = {"class": "QlibDataLoader", "kwargs": {"config": data_loader_config}}
        processors = [DropnaProcessor(fields_group="feature"), DropnaLabel(), CSZScoreNorm()]
        
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            **kwargs
        )
        self.processors = processors

def main():
    """主执行函数"""
    # 设置Pandas显示选项
    pd.set_option('display.max_rows', 100)

    # 原始ETF列表
    ETF_LIST = [
        "510300.SH", "159339.SH", "512480.SH", "516160.SH", "512690.SH",
        "512800.SH", "512010.SH", "510630.SH", "515790.SH", "512880.SH",
    ]

    # 原始时间范围
    START_TIME = "2015-01-01"
    END_TIME = "2026-01-09"

    print("开始创建DataHandler实例...")
    handler = ETFDataHandler(
        instruments=ETF_LIST,
        start_time=START_TIME,
        end_time=END_TIME,
    )
    print("DataHandler实例创建成功。")

    # 原始数据段
    segments = {
        "train": ("2015-01-01", "2022-12-31"),
        "valid": ("2023-01-01", "2024-12-31"),
        "test": ("2025-01-01", END_TIME),
    }

    print("开始创建DatasetH实例...")
    dataset = DatasetH(handler=handler, segments=segments)
    print("DatasetH实例创建成功。")

    # --- 获取并打印数据 ---
    print("\n--- 准备训练集数据 ---")
    try:
        train_features = dataset.prepare("train", col_set="feature")
        print("训练集特征数据形状:", train_features.shape)
        if not train_features.empty:
            print("训练集特征示例:\n", train_features.head(10))

        train_label = dataset.prepare("train", col_set="label")
        print("\n训练集标签形状:", train_label.shape)
        if not train_label.empty:
            print("训练集标签示例:\n", train_label.head(10))
    except Exception as e:
        print(f"准备训练集时出错: {e}\n{traceback.format_exc()}")

    print("\n--- 准备测试集数据 ---")
    try:
        test_data = dataset.prepare("test", col_set=["feature", "label"])
        print("\n测试集数据形状:", test_data.shape)
        if not test_data.empty:
            print("测试集数据示例:\n", test_data.head(20))
            instruments_count = test_data.groupby(level="instrument").size()
            print("\n测试集每个ETF样本数:\n", instruments_count)
        else:
            print("测试集为空。")
    except Exception as e:
        print(f"准备测试集时出错: {e}\n{traceback.format_exc()}")

# 使用 if __name__ == '__main__' 来防止多进程问题
if __name__ == '__main__':
    if setup_qlib():
        main()