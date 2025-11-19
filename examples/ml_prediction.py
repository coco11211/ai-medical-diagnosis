"""
Example: Machine learning price prediction
"""
import sys
sys.path.append('..')

from src.engine import TradingEngine

def main():
    # Initialize trading engine
    print("Initializing trading engine...")
    engine = TradingEngine()

    symbol = 'AAPL'

    # Train ML model
    print(f"\nTraining ML model for {symbol}...")
    print("This may take a few minutes...")

    results = engine.train_ml_model(symbol)

    # Print training results
    print("\n" + "="*70)
    print("ML MODEL TRAINING RESULTS")
    print("="*70)
    print(f"Model Type: {results['model_type']}")
    print(f"Training Samples: {results['train_samples']}")
    print(f"Test Samples: {results['test_samples']}")

    metrics = results['metrics']
    print(f"\nModel Performance:")
    print(f"  R² Score: {metrics['r2']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  MAPE: {metrics['mape']:.2f}%")

    # Make predictions
    print(f"\nGenerating price predictions for {symbol}...")
    predictions = engine.predict_price(symbol, steps=5)

    print("\nPredicted prices for next 5 days:")
    for i, price in enumerate(predictions, 1):
        print(f"  Day {i}: ${price:.2f}")

    print("="*70)

if __name__ == "__main__":
    main()
