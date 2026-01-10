# 导入（接前步骤）
from qlib.data.dataset.processor import CSZScoreNorm, DropnaLabel
from qlib.contrib.data.loader import QlibDataLoader

# 更新ETF列表（强烈建议将A500替换为主流代码"512050.SH"，规模大、数据完整）
ETF_LIST = [
    "510300.SH",    # 沪深300ETF
    "512050.SH",    # 中证A500ETF（推荐替换为华夏主流代码，2026年规模超400亿；原159339.SH若数据少请换此）
    "512480.SH",    # 半导体
    "516160.SH",    # 新能源车
    "512690.SH",    # 白酒
    "512800.SH",    # 银行
    "512010.SH",    # 医药
    "510630.SH",    # 消费
    "515790.SH",    # 光伏
    "512880.SH",    # 证券
]

# 更新DataHandler：修正processors和label表达式
class ETFDataHandler(DataHandlerLP):
    def __init__(self, instruments, start_time, end_time, **kwargs):
        data_loader_config = {
            "feature": [  # 10个常见因子
                "Log(Mean($volume * $close, 20))",                  # 市值/流动性代理
                "$close / Ref($close, 20) - 1",                 # 20日动量
                "$close / Ref($close, 60) - 1",                 # 60日动量
                "$close / Ref($close, 120) - 1",               # 半年动量
                "($close / Ref($close, 5) - 1) * -1",                # 5日反转
                "Std($close / Ref($close, 1) - 1, 20)",         # 20日波动率
                "Std($close / Ref($close, 1) - 1, 60)",         # 60日波动率
                "$volume / Mean($volume, 20)",                       # 短期量比
                "$volume / Mean($volume, 60)",                       # 中期量比
                "Skew($close / Ref($close, 1) - 1, 20)",       # 20日偏度
            ],
            "label": [
                "Ref($close, -1) / $close - 1"  # 修正：下一交易日收益（未来收益，正向标签）
            ],
        }
        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": data_loader_config,
                "freq": "day",
            }
        }
        # 修正processors：移除无效类，仅保留标准处理器
        processors = [
            DropnaLabel(),                  # 丢弃标签NaN行（确保有未来收益）
            CSZScoreNorm(fields_group="feature"),  # 横截面Z-score标准化
        ]
        super().__init__(
            data_loader=data_loader,
            learn_processors=processors,
            **kwargs
        )

# 重新创建handler和dataset
handler = ETFDataHandler(
    instruments=ETF_LIST,
    start_time=START_TIME,
    end_time=END_TIME,
)

dataset = DatasetH(
    handler=handler,
    segments=segments,
)

# 测试数据（同前）
train_features = dataset.prepare("train", col_set="feature")
print("训练集因子数量:", len(train_features.columns))
print("因子名称:", list(train_features.columns))
print("\n训练集形状:", train_features.shape)

test_features = dataset.prepare("test", col_set="feature")
print("\n测试集形状:", test_features.shape)
print("测试集每个ETF样本数:\n", test_features.groupby(level="instrument").size())

test_label = dataset.prepare("test", col_set="label")
print("\n测试集标签示例（前10行）:\n", test_label.head(10))

print("\n训练集因子相关性:\n", train_features.corr().round(2))