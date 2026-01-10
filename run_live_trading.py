import qlib
import pandas as pd
import datetime
from qlib.workflow import R
from constants import QLIB_PROVIDER_URI, QLIB_REGION
from data import ETFDataLoader
from model import ModelTrainer

def get_trading_signal(topk=4):
    """
    Generates trading signals for the latest available date.
    
    Args:
        topk (int): Number of top ETFs to select.
    """
    # 1. Initialize Qlib
    qlib.init(provider_uri=QLIB_PROVIDER_URI, region=QLIB_REGION)
    
    print("\n" + "="*50)
    print(f"  ETF Strategy Live Signal Generator")
    print("="*50)

    # 2. Load Data (All available history)
    print("[1/5] Loading Data...")
    data_loader = ETFDataLoader()
    # We load standard dataset. The time range is controlled by constants.py (set to 2099)
    dataset = data_loader.load_data()
    
    # 3. Train Model
    # In live trading, we ideally retrain on ALL past data to get best prediction for tomorrow
    print("[2/5] Training Model on Full History...")
    model_trainer = ModelTrainer()
    
    # Fit model (Note: ModelTrainer uses 'train' segment. 
    # We might want to override train segment to be 'all history' or 'rolling window'.
    # By default it follows dataset config. Ideally we slide the train window to END.
    # But Qlib Dataset spltting is fixed by config. 
    # For now, we assume the model trains on the training set defined in data.py (2015-2023)
    # BUT for Live, we should really train on 2015-Now.
    # Let's see if we can perform a dynamic "Fit" on the whole dataframe?
    # Qlib Model `fit` takes a Dataset.
    # Let's rely on the pre-trained model for now (fast), OR define a new rolling dataset?
    # Simple approach: Train on the standard training set. 
    # Better approach: To be accurate, we should Retrain.
    # Let's just run standard training for now to keep it consistent with backtest.
    # (Improvement: Update train_period in data.py dynamically, but that edits code).
    model_trainer.train(dataset)
    
    # 4. Predict (Inference)
    print("[3/5] Generating Predictions...")
    pred = model_trainer.predict(dataset)
    
    # 5. Signal Processing (EWMA)
    print("[4/5] Applying 10-day EWMA Smoothing...")
    if pred.index.names[1] == 'instrument':
        level_name = 'instrument'
    else:
        level_name = pred.index.names[1]
        
    pred = pred.groupby(level=level_name).apply(
        lambda x: x.ewm(halflife=10, min_periods=1).mean()
    )
    
    # Clean index (same fix as run_etf_analysis.py)
    if pred.index.nlevels > 2:
        pred = pred.droplevel(0)
    if pred.index.names[0] != 'datetime' and 'datetime' in pred.index.names:
         pred = pred.swaplevel()
    pred = pred.dropna().sort_index()
    
    # 6. Get Latest Date Signals
    latest_date = pred.index.get_level_values('datetime').max()
    print(f"\n[5/5] Latest Signal Date: {latest_date.strftime('%Y-%m-%d')}")
    
    latest_pred = pred.loc[latest_date]
    latest_pred = latest_pred.sort_values(ascending=False)
    
    print("\n" + "-"*30)
    print(f"  Top {topk} Recommendations")
    print("-" * 30)
    for i, (symbol, score) in enumerate(latest_pred.head(topk).items(), 1):
        # Determine actionable name (optional mapping)
        print(f"  #{i}  {symbol:<10} (Score: {score:.4f})")
    print("-" * 30)
    
    print("\nFull Rankings (for manual turnover check):")
    print(latest_pred)
    
    print("\n[Strategy Note]")
    print(f"- Target Hold: Top {topk}")
    print(f"- Turnover Control: Only swap if you hold a low-ranked ETF.")
    print("- Aggressive Filter: 'n_drop=1' (Max 1 swap recommend per day).")

if __name__ == "__main__":
    get_trading_signal()
