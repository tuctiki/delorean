import os
import sys
import pandas as pd
import numpy as np
import logging
from gplearn.genetic import SymbolicTransformer
from gplearn.functions import make_function

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from delorean.config import ETF_LIST, START_TIME, QLIB_PROVIDER_URI
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
import qlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("delorean.leadership_miner")

def convert_to_qlib(program_str, feature_names):
    """Converts a gplearn string to a Qlib expression."""
    expr = program_str
    # Sort feature indices descending to avoid X1 replacing X10
    for i in range(len(feature_names)-1, -1, -1):
        expr = expr.replace(f"X{i}", feature_names[i])
        
    replacements = {'add': '+', 'sub': '-', 'mul': '*', 'div': '/', 'neg': '-1*', 'abs': 'Abs', 'log': 'Log', 'sqrt': 'Sqrt', 'inv': 'Inv'}
    
    return _parse_and_translate(expr, replacements)

def _parse_and_translate(expr, replacements):
    if '(' not in expr:
        # Match literal replacements for solo terms if any (e.g. neg(X0))
        return expr
    
    func_name_end = expr.index('(')
    func_name = expr[:func_name_end]
    
    count = 0
    args_start = func_name_end + 1
    args_end = -1
    for i, char in enumerate(expr[args_start:], start=args_start):
        if char == '(': count += 1
        elif char == ')':
            count -= 1
            if count == -1:
                args_end = i
                break
    if args_end == -1: return expr
    
    args_str = expr[args_start:args_end]
    args = []
    current_arg = []
    depth = 0
    for char in args_str:
        if char == ',' and depth == 0:
            args.append("".join(current_arg).strip())
            current_arg = []
        else:
            if char == '(': depth += 1
            if char == ')': depth -= 1
            current_arg.append(char)
    args.append("".join(current_arg).strip())
    
    trans_args = [_parse_and_translate(a, replacements) for a in args]
    
    if func_name in replacements:
        mapped = replacements[func_name]
        if mapped in ['+', '-', '*', '/']:
            return f"({trans_args[0]} {mapped} {trans_args[1]})"
        elif mapped == 'Inv':
            return f"(1 / ({trans_args[0]}))"
        elif mapped == '-1*':
            return f"(-1 * ({trans_args[0]}))"
        else:
            return f"{mapped}({', '.join(trans_args)})"
    else:
        return f"{func_name.capitalize()}({', '.join(trans_args)})"

def main():
    qlib.init(provider_uri=QLIB_PROVIDER_URI, region="cn")
    
    TRAIN_START = "2015-01-01"
    TRAIN_END = "2022-12-31" 
    
    # LEADERSHIP Oriented Input Features
    features = [
        "($close / Ref($close, 1)) - 1",            # Daily Ret
        "($close / Ref($close, 20)) - 1",           # 20d Ret
        "($close / Ref($close, 60)) - 1",           # 60d Ret
        "($close - Min($low, 40)) / (Max($high, 40) - Min($low, 40) + 1e-4)", # Range Pos 40
        "Std($close / Ref($close, 1), 20)",         # Vol 20
        "$volume / Mean($volume, 20)",              # Rel Vol
        "($close / Mean($close, 60)) - 1",          # Dist MA60
        "Corr($close, $volume, 10)",                 # Price-Vol Corr
        "Abs($close / $open - 1)",                   # Intraday Magnitude
        "Mean($volume * ($close - $low - ($high - $close)) / ($high - $low + 1e-4), 20)", # Money Flow
    ]
    
    feature_names = [
        "RET1", "RET20", "RET60", "RANGE40", "VOL20", "RELVOL", "DIST60", "PVCORR", "BODY", "MF20"
    ]
    
    # Label: 10-day forward return (Excess Owning Window)
    label = ["Ref($close, -10) / $close - 1"]
    
    logger.info("Initializing Leadership DataHandler...")
    
    dh_config = {
        "start_time": TRAIN_START,
        "end_time": TRAIN_END,
        "instruments": ETF_LIST,
        "learn_processors": [
             {"class": "DropnaLabel"},
             {"class": "CSZScoreNorm", "kwargs": {"fields_group": "feature"}} 
        ]
    }
    
    dataset = DatasetH(
        handler={
            "class": "DataHandlerLP",
            "module_path": "qlib.data.dataset.handler",
            "kwargs": {
                "data_loader": {
                    "class": "QlibDataLoader",
                    "kwargs": {
                        "config": {"feature": features, "label": label},
                        "freq": "day",
                    },
                },
                **dh_config
            },
        },
        segments={"train": (TRAIN_START, TRAIN_END)}
    )
    
    logger.info("Loading Training Data...")
    df_train = dataset.prepare("train", col_set=["feature", "label"])
    df_train = df_train.dropna()
    
    X = df_train["feature"].values
    y = df_train["label"].values.ravel()
    
    logger.info(f"Input Shape: {X.shape}")
    
    function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv']
    
    est = SymbolicTransformer(
        generations=40,
        population_size=4000,
        hall_of_fame=100,
        n_components=20,
        function_set=function_set,
        parsimony_coefficient=0.001,
        max_samples=0.8,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    
    logger.info("Mining Leadership Signals...")
    est.fit(X, y)
    
    print("\n" + "="*60)
    print("  TOP LEADERSHIP FACTORS (Target: 10d Forward Return)")
    print("="*60)
    
    results = []
    for i, program in enumerate(est):
        qlib_expr = convert_to_qlib(str(program), feature_names)
        results.append(qlib_expr)
        print(f"\n[Alpha_{i+1}]")
        print(f"Qlib: {qlib_expr}")
    
    # Save to file
    with open("artifacts/mined_leadership_candidates.txt", "w") as f:
        for expr in results:
            f.write(expr + "\n")

if __name__ == "__main__":
    main()
