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
logger = logging.getLogger("delorean.miner")

# -------------------------------------------------------------------------
# 1. Translator Logic : GP -> Qlib
# -------------------------------------------------------------------------
def convert_to_qlib(program_str, feature_names):
    """
    Converts a gplearn Lisp-like string to a Qlib expression.
    e.g. 'add(X0, div(X1, X2))' -> '($open + ($high / $low))'
    """
    # Map feature names: X0 -> $open, etc.
    # Replace X{i} with a temporary placeholder that won't conflict
    # We use a primitive parsing approach or just simple replacement if we trust the format
    
    # 1. Replace Variables
    # We must do this carefully. 'X10' contains 'X1'.
    # We sort feature indices descending.
    
    expr = program_str
    for i in range(len(feature_names)-1, -1, -1):
        # Explicit word boundary might be needed, but gplearn usually outputs clean args like X0
        # standard replace is risky if we have X1 and X10. 
        # But loop order descending handles X10 before X1.
        expr = expr.replace(f"X{i}", feature_names[i])
        
    # 2. Map Functions
    # Standard format: function(arg1, arg2)
    # We can use simple text replacement for operators if we are careful about nesting.
    # However, transforming 'add(A, B)' to '(A + B)' recursively is better.
    
    # Let's simple-replace known binary operators to infix for readability
    # Qlib supports many functions like Mean($close, 5), but GP outputs fundamental math
    
    replacements = {
        'add': '+',
        'sub': '-',
        'mul': '*',
        'div': '/',
    }
    
    # Recursive parser is cleaner.
    # Tokenize by finding balanced parentheses
    
    return _parse_and_translate(expr, replacements)

def _parse_and_translate(expr, replacements):
    # This is a heuristic translation for standard gplearn output
    # It might be brittle for very complex nested structures but works for standard trees
    
    # Base case: if no '(' in string, it's a leaf (variable or constant)
    if '(' not in expr:
        return expr
    
    # Find the first function call
    func_name_end = expr.index('(')
    func_name = expr[:func_name_end]
    
    # Find the matching closing parenthesis
    count = 0
    args_start = func_name_end + 1
    args_end = -1
    
    for i, char in enumerate(expr[args_start:], start=args_start):
        if char == '(':
            count += 1
        elif char == ')':
            count -= 1
            if count == -1:
                args_end = i
                break
    
    if args_end == -1:
        return expr # Should not happen if well-formed
        
    # Extract args string
    args_str = expr[args_start:args_end]
    
    # Split args by comma, honoring nested parens
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
    
    # Recursively translate args
    trans_args = [_parse_and_translate(a, replacements) for a in args]
    
    # Construct new expression
    if func_name in replacements and len(trans_args) == 2:
        op = replacements[func_name]
        return f"({trans_args[0]} {op} {trans_args[1]})"
    elif func_name == 'neg':
        return f"-1 * ({trans_args[0]})"
    elif func_name == 'inv':
        return f"1 / ({trans_args[0]})"
    elif func_name == 'abs':
        return f"Abs({trans_args[0]})"
    elif func_name == 'sqrt':
        return f"Power(Abs({trans_args[0]}), 0.5)" # Qlib Sqrt safety using Power
    elif func_name == 'log':
        return f"Log(Abs({trans_args[0]}))" # Qlib Log safety 
    else:
        # Fallback for unknown functions -> keep syntax func(A, B)
        # Qlib supports many numpy ufuncs directy
        joined_args = ", ".join(trans_args)
        return f"{func_name.capitalize()}({joined_args})"

# -------------------------------------------------------------------------
# 2. Main Mining Logic
# -------------------------------------------------------------------------
def main():
    qlib.init(provider_uri=QLIB_PROVIDER_URI, region="cn")
    
    # Config
    TRAIN_START = "2015-01-01"
    TRAIN_END = "2021-12-31" # Training period for mining
    
    # RAW input features
    features = [
        "$open", "$high", "$low", "$close", "$volume",
        "Mean($close, 5)", "Mean($close, 20)", "Mean($close, 60)",
        "Std($close, 5)", "Std($close, 20)", "Std($close, 60)",
        "Ref($close, 1)", "Ref($close, 5)", "Ref($close, 10)",
        "($close - Mean($close, 20)) / Std($close, 20)", # BB Z-Score
        "($close / Ref($close, 1)) - 1" # Daily Return
    ]
    
    feature_names = [
        "$open", "$high", "$low", "$close", "$volume",
        "MA5", "MA20", "MA60",
        "STD5", "STD20", "STD60",
        "REF1", "REF5", "REF10",
        "BB_SCORE",
        "RETURN1"
    ]
    
    # Goal: Predict 5-day return
    # Rank Label to keep it consistent with LightGBM strategy
    label = ["Ref($close, -5) / $close - 1"]
    
    logger.info("Initializing Raw DataHandler...")
    
    # Manual DataHandler config
    dh_config = {
        "start_time": TRAIN_START,
        "end_time": TRAIN_END,
        "instruments": ETF_LIST,
        "infer_processors": [],
        "learn_processors": [
             {"class": "DropnaLabel"},
             {"class": "CSZScoreNorm", "kwargs": {"fields_group": "feature"}} 
        ]
    }
    
    # We use Qlib's standard handler but inject our config
    # To do this cleanly without a class, we pass 'config' to DataHandlerLP
    
    dataset = DatasetH(
        handler={
            "class": "DataHandlerLP",
            "module_path": "qlib.data.dataset.handler",
            "kwargs": {
                "data_loader": {
                    "class": "QlibDataLoader",
                    "kwargs": {
                        "config": {
                            "feature": features,
                            "label": label,
                        },
                        "freq": "day",
                    },
                },
                **dh_config
            },
        },
        segments={
            "train": (TRAIN_START, TRAIN_END),
        }
    )
    
    # Load Data
    logger.info("Loading Training Data...")
    df_train = dataset.prepare("train", col_set=["feature", "label"])
    
    # Cleaning
    df_train = df_train.dropna()
    if df_train.empty:
        logger.error("No data found!")
        return

    X = df_train["feature"].values
    y = df_train["label"].values.ravel()
    
    logger.info(f"Input Data Shape: {X.shape}")
    
    # GP Configuration
    # We use a smaller set for 'turbo' mining in this script
    # Standard function set + some domain specific logic if possible
    # gplearn doesn't support stateful functions (like Ref, Mean) easily out of the box
    # WITHOUT custom functions.
    # So we strictly mine 'Price Action Patterns' (High/Low relationships, etc)
    # To get TSA (Mean, Ref), we would need to pre-compute basic rolling features or use a custom GP library.
    # For this iteration: First Principles Price Relations ($close vs $high, etc.)
    
    function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv']
    
    est = SymbolicTransformer(
        generations=50,
        population_size=5000,
        hall_of_fame=50,
        n_components=20, # Keep top 20
        function_set=function_set,
        parsimony_coefficient=0.0005,
        max_samples=0.9,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    
    logger.info("Evolving Factors...")
    est.fit(X, y)
    
    print("\n" + "="*60)
    print("  TOP DISCOVERED FACTORS")
    print("="*60)
    
    for i, program in enumerate(est):
        raw_expr = str(program)
        qlib_expr = convert_to_qlib(raw_expr, feature_names)
        
        # Heuristic scoring (correlation with y) already done by GP selection
        # We just list them
        print(f"\n[Factor {i+1}]")
        print(f"Raw:  {raw_expr}")
        print(f"Qlib: {qlib_expr}")

if __name__ == "__main__":
    main()
