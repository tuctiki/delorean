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

# ==========================================
# 步骤4: Alpha 模型构建（LightGBM）
# ==========================================
from qlib.contrib.model.gbdt import LGBModel
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
import pandas as pd
import matplotlib.pyplot as plt

# 定义LightGBM模型
model = LGBModel(
    loss="mse",               # 回归任务，预测收益
    colsample_bytree=0.887,   # 默认超参
    learning_rate=0.05,
    subsample=0.7,
    lambda_l1=1,
    lambda_l2=1,
    max_depth=-1,
    num_leaves=31,
    min_data_in_leaf=20,
    early_stopping_rounds=50,
)

# 使用Qlib Recorder记录实验
# 使用 context manager 自动管理 run 的生命周期
print("\n开始训练 LightGBM 模型...")
with R.start(experiment_name="ETF_Strategy") as recorder:
    # 训练模型
    model.fit(dataset)

    # 生成测试集预测分数
    print("生成测试集预测分数...")
    pred = model.predict(dataset)
    # pred columns usually is 'score'

    # 保存预测记录
    R.save_objects(pred=pred)

    # 特征重要性
    feature_importance = pd.DataFrame({
        'feature': dataset.prepare("train", col_set="feature").columns,
        'importance': model.get_feature_importance()
    }).sort_values('importance', ascending=False)

    print("\n特征重要性（前10）:\n", feature_importance.head(10))

    # 绘制重要性图 (如果支持显示)
    try:
        plt.figure(figsize=(10, 6))
        plt.bar(feature_importance['feature'][:10], feature_importance['importance'][:10])
        plt.xticks(rotation=45, ha='right')
        plt.title("LightGBM Factor Importance Top 10")
        plt.tight_layout()
        # plt.show() # Skip show to avoid blocking in non-interactive environment
        print("特征重要性图绘制完成 (skipped plt.show())")
    except Exception as e:
        print(f"绘图失败: {e}")

    # 查看测试集预测示例
    print("\n测试集预测分数示例（前20行）:\n", pred.head(20))

print("模型构建完成。")