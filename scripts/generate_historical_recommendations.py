import qlib
import pandas as pd
import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qlib.workflow import R
from qlib.data import D
from delorean.config import QLIB_PROVIDER_URI, QLIB_REGION, BENCHMARK, ETF_NAME_MAP, ETF_LIST
from delorean.data import ETFDataLoader
from delorean.model import ModelTrainer

def generate_historical_recommendations(num_days=7, topk=5):
    """
    Generate top recommendations for the past N trading days.
    
    Args:
        num_days (int): Number of past trading days to generate recommendations for.
        topk (int): Number of top ETFs to select per day.
    
    Returns:
        pd.DataFrame: DataFrame with columns [date, rank, symbol, name, score, volatility, price, allocation]
    """
    # 1. Initialize Qlib
    qlib.init(provider_uri=QLIB_PROVIDER_URI, region=QLIB_REGION)
    
    # 2. Load Data
    print("Loading data...")
    data_loader = ETFDataLoader(label_horizon=5)
    dataset = data_loader.load_data()
    
    # 3. Load the latest trained model from Qlib Recorder
    print("Loading latest model from experiments...")
    try:
        # Get the latest experiment
        from qlib.workflow.recorder import Experiment
        exp = Experiment.get_instance()
        
        # Load the saved prediction from the latest experiment
        # This is more reliable than re-running prediction
        pred = R.load_object("pred")
        print(f"Loaded predictions from experiment: {exp.id}")
    except Exception as e:
        print(f"Warning: Could not load from recorder ({e}), generating fresh predictions...")
        # Fallback: train a new model
        model_trainer = ModelTrainer(seed=42)
        model_trainer.train(dataset)
        pred = model_trainer.predict(dataset)
    
    # 4. Apply EWMA smoothing (matching live trading config)
    print("Applying 20-day EWMA smoothing...")
    level_name = pred.index.names[1] if pred.index.nlevels > 1 else 'instrument'
    pred = pred.groupby(level=level_name).apply(
        lambda x: x.ewm(halflife=20, min_periods=1).mean()
    )
    
    if pred.index.nlevels > 2:
        pred = pred.droplevel(0)
    
    if pred.index.names[0] != 'datetime' and 'datetime' in pred.index.names:
        pred = pred.swaplevel()
    
    pred = pred.dropna().sort_index()
    
    # 5. Get last N trading dates from TEST PERIOD (where we have actual data)
    # Filter to only dates in the test period (2023-01-01 onwards)
    test_start = pd.Timestamp('2023-01-01')
    all_dates = pred.index.get_level_values('datetime').unique().sort_values()
    test_dates = all_dates[all_dates >= test_start]
    
    # Get the LAST N dates from the test period (most recent historical data)
    last_n_dates = test_dates[-num_days:]
    
    print(f"\nGenerating recommendations for {len(last_n_dates)} trading days:")
    print(f"From: {last_n_dates[0].strftime('%Y-%m-%d')}")
    print(f"To: {last_n_dates[-1].strftime('%Y-%m-%d')}")
    
    # 6. Fetch volatility and price data for all dates
    print("\nFetching volatility and price data...")
    start_date = last_n_dates[0]
    end_date = last_n_dates[-1]
    
    feat_df = D.features(
        ETF_LIST, 
        ['$close', 'Std($close/Ref($close,1)-1, 20)'], 
        start_time=start_date, 
        end_time=end_date
    )
    feat_df.columns = ['close', 'vol20']
    
    # Check index levels and swap if needed to ensure (datetime, instrument) order
    if feat_df.index.names[0] != 'datetime':
        feat_df = feat_df.swaplevel().sort_index()
    
    # 7. Generate recommendations for each date
    results = []
    
    for date in last_n_dates:
        # Get predictions for this date
        try:
            date_pred = pred.loc[date].sort_values(ascending=False)
        except KeyError:
            print(f"Warning: No predictions for {date.strftime('%Y-%m-%d')}, skipping...")
            continue
        
        # Get top K
        top_symbols = date_pred.head(topk)
        
        # Calculate equal weights
        weight = 1.0 / topk
        
        for rank, (symbol, score) in enumerate(top_symbols.items(), 1):
            # Get volatility and price
            try:
                vol_raw = feat_df.loc[(date, symbol), 'vol20']
                close_price = feat_df.loc[(date, symbol), 'close']
            except KeyError:
                vol_raw = 0.0
                close_price = 0.0
            
            results.append({
                'date': date.strftime('%Y-%m-%d'),
                'rank': rank,
                'symbol': symbol,
                'name': ETF_NAME_MAP.get(symbol, symbol),
                'score': float(score),
                'volatility': float(vol_raw) if not pd.isna(vol_raw) else 0.0,
                'price': float(close_price) if not pd.isna(close_price) else 0.0,
                'allocation': weight
            })
    
    # 8. Create DataFrame
    df = pd.DataFrame(results)
    
    return df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate historical recommendations")
    parser.add_argument("--days", type=int, default=7, help="Number of past trading days (default: 7)")
    parser.add_argument("--topk", type=int, default=7, help="Number of top ETFs per day (default: 7)")
    parser.add_argument("--output", type=str, default="artifacts/historical_recommendations.csv", 
                        help="Output CSV file path")
    parser.add_argument("--pivot", action="store_true", help="Pivot table so dates are columns")
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Generating Historical Recommendations")
    print(f"{'='*60}")
    
    df = generate_historical_recommendations(num_days=args.days, topk=args.topk)
    
    # Pivot if requested (dates as columns)
    if args.pivot:
        print("\nPivoting table (dates as columns)...")
        # Create a combined identifier for each recommendation
        df['rank_info'] = 'Rank_' + df['rank'].astype(str)
        
        # Pivot: rows = rank, columns = date, values = symbol + name
        pivot_df = df.pivot_table(
            index='rank',
            columns='date',
            values=['symbol', 'name', 'score'],
            aggfunc='first'
        )
        
        # Flatten column names
        pivot_df.columns = [f'{col[1]}_{col[0]}' for col in pivot_df.columns]
        pivot_df = pivot_df.reset_index()
        
        # Save pivoted version
        pivot_df.to_csv(args.output, index=False)
        print(f"\nPivoted output saved to: {args.output}")
        print(f"\nPreview:")
        print(pivot_df.head(10).to_string(index=False))
    else:
        # Save to CSV (long format)
        df.to_csv(args.output, index=False)
        
        print(f"\n{'='*60}")
        print(f"Results Summary")
        print(f"{'='*60}")
        print(f"Total recommendations: {len(df)}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"Unique dates: {df['date'].nunique()}")
        print(f"Output saved to: {args.output}")
        print(f"\nPreview:")
        print(df.head(10).to_string(index=False))
