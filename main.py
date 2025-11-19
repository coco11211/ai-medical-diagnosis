"""
Autonomous Trading Bot Simulator & Audio Noise Cancellation - Main Entry Point

A comprehensive trading bot with:
- Multiple strategy algorithms (MACD, RSI, Bollinger Bands)
- Backtesting engine
- Risk management
- Paper trading
- Real-time market data
- Portfolio optimization
- Machine learning price prediction
- Multi-asset support
- Performance analytics dashboard
- Alert system

Audio Noise Cancellation Features:
- Real-time live audio processing
- Multiple noise cancellation algorithms (Spectral Subtraction, Wiener Filter)
- Low-latency processing optimized for Windows 11
- Audio file processing and batch conversion
- Real-time monitoring and visualization

Full Windows 11 compatibility
"""
import argparse
import sys
import logging
from datetime import datetime

from src.engine import TradingEngine
from src.analytics import Dashboard
from src.audio import AudioProcessor, AudioConfig
from src.audio.audio_visualizer import ConsoleVisualizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def handle_audio_mode(args):
    """Handle audio processing modes."""
    try:
        if args.mode == 'audio-devices':
            # List audio devices
            print("\n" + "="*70)
            print("AUDIO DEVICES")
            print("="*70)

            config = AudioConfig()
            processor = AudioProcessor(config)
            processor.list_devices()

            return

        elif args.mode == 'audio-file':
            # Process audio file
            if not args.audio_input:
                print("Error: --audio-input required for audio-file mode")
                sys.exit(1)

            print("\n" + "="*70)
            print("AUDIO FILE PROCESSING")
            print("="*70)
            print(f"Input: {args.audio_input}")

            # Create audio config
            config = AudioConfig(
                noise_reduction_strength=args.noise_reduction
            )

            processor = AudioProcessor(config)

            # Capture noise profile if requested
            if args.noise_profile:
                print("\nCapturing noise profile from file beginning...")
                # Use first 2 seconds as noise profile
                from src.audio.audio_utils import AudioFileHandler
                noise_audio, sr, ch = AudioFileHandler.read_wav(args.audio_input)
                noise_samples = min(len(noise_audio), int(2.0 * sr))
                processor.canceller.set_noise_profile(noise_audio[:noise_samples])

            # Process file
            output_file = processor.process_file(args.audio_input, args.audio_output)
            print(f"Output: {output_file}")
            print("="*70 + "\n")

        elif args.mode == 'audio-live':
            # Real-time audio processing
            print("\n" + "="*70)
            print("REAL-TIME AUDIO NOISE CANCELLATION")
            print("="*70)
            print("Windows 11 Optimized Real-Time Audio Processing")
            print(f"Latency Mode: {args.latency_mode.upper()}")
            print(f"Noise Reduction: {args.noise_reduction * 100:.0f}%")
            print("="*70)

            # Create audio config
            config = AudioConfig(
                noise_reduction_strength=args.noise_reduction,
                latency_mode=args.latency_mode,
                monitor_enabled=args.monitor,
                save_output=args.record
            )

            processor = AudioProcessor(config)

            # List available devices
            processor.list_devices()

            # Capture noise profile if requested
            if args.noise_profile:
                processor.capture_noise_profile()

            # Start processing
            print("\nStarting real-time audio processing...")
            print("Press Ctrl+C to stop\n")

            # Start visualization if requested
            visualizer = None
            if args.visualize:
                try:
                    from src.audio.audio_visualizer import AudioVisualizer
                    visualizer = AudioVisualizer(config)
                    processor.capture.register_callback(
                        lambda audio, time: visualizer.update_audio(audio)
                    )
                    import threading
                    viz_thread = threading.Thread(
                        target=lambda: visualizer.start(blocking=False),
                        daemon=True
                    )
                    viz_thread.start()
                except Exception as e:
                    print(f"Warning: Visualization not available: {e}")

            try:
                processor.start_processing(monitor=args.monitor, record=args.record)

                # Keep running until interrupted
                import time
                while True:
                    time.sleep(0.1)

            except KeyboardInterrupt:
                print("\n\nStopping audio processing...")
                processor.stop_processing()

                if visualizer:
                    visualizer.stop()

                # Print statistics
                stats = processor.get_statistics()
                if stats:
                    print("\n" + "="*70)
                    print("PROCESSING STATISTICS")
                    print("="*70)
                    print(f"Frames Processed: {stats['frames_processed']}")
                    print(f"Average Latency: {stats['average_latency_ms']:.2f} ms")
                    print(f"Min Latency: {stats['min_latency_ms']:.2f} ms")
                    print(f"Max Latency: {stats['max_latency_ms']:.2f} ms")
                    print(f"Real-time Factor: {stats['realtime_factor']:.2f}x")
                    print("="*70 + "\n")

    except Exception as e:
        logger.error(f"Audio processing error: {e}", exc_info=True)
        sys.exit(1)


def main():
    """Main entry point for trading bot."""
    parser = argparse.ArgumentParser(
        description='Autonomous Trading Bot Simulator',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        'mode',
        choices=['backtest', 'paper', 'optimize', 'ml', 'dashboard', 'compare',
                 'audio-live', 'audio-file', 'audio-devices'],
        help='Operating mode (trading or audio processing)'
    )

    parser.add_argument(
        '--symbol',
        type=str,
        default='AAPL',
        help='Stock symbol (default: AAPL)'
    )

    parser.add_argument(
        '--symbols',
        type=str,
        nargs='+',
        help='Multiple symbols for portfolio optimization'
    )

    parser.add_argument(
        '--strategy',
        type=str,
        choices=['macd', 'rsi', 'bollinger'],
        default='macd',
        help='Trading strategy (default: macd)'
    )

    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--end-date',
        type=str,
        help='End date (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--config',
        type=str,
        help='Path to config file'
    )

    parser.add_argument(
        '--optimize-method',
        type=str,
        choices=['max_sharpe', 'min_volatility', 'equal_weight', 'risk_parity'],
        default='max_sharpe',
        help='Portfolio optimization method'
    )

    # Audio processing arguments
    parser.add_argument(
        '--audio-input',
        type=str,
        help='Input audio file path (for audio-file mode)'
    )

    parser.add_argument(
        '--audio-output',
        type=str,
        help='Output audio file path (for audio-file mode)'
    )

    parser.add_argument(
        '--noise-profile',
        action='store_true',
        help='Capture noise profile before processing'
    )

    parser.add_argument(
        '--noise-reduction',
        type=float,
        default=0.8,
        help='Noise reduction strength (0.0 to 1.0, default: 0.8)'
    )

    parser.add_argument(
        '--latency-mode',
        type=str,
        choices=['low', 'medium', 'high'],
        default='low',
        help='Latency mode for real-time processing (default: low)'
    )

    parser.add_argument(
        '--monitor',
        action='store_true',
        help='Enable audio monitoring (playback cleaned audio)'
    )

    parser.add_argument(
        '--record',
        action='store_true',
        help='Record processed audio to file'
    )

    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Enable real-time audio visualization'
    )

    args = parser.parse_args()

    # Handle audio modes separately
    if args.mode.startswith('audio-'):
        handle_audio_mode(args)
        return

    # Initialize trading engine
    logger.info("Initializing trading engine...")
    engine = TradingEngine(config_path=args.config)

    try:
        if args.mode == 'backtest':
            # Run backtest
            logger.info(f"Running backtest for {args.symbol} with {args.strategy} strategy")

            results = engine.run_backtest(
                symbol=args.symbol,
                strategy_name=args.strategy,
                start_date=args.start_date,
                end_date=args.end_date
            )

            # Print results
            print("\n" + "="*70)
            print("BACKTEST RESULTS")
            print("="*70)
            print(f"Symbol: {args.symbol}")
            print(f"Strategy: {args.strategy.upper()}")
            print(f"Initial Capital: ${results['initial_capital']:,.2f}")
            print(f"Final Value: ${results['final_value']:,.2f}")
            print(f"Total Return: {results['total_return_pct']:.2f}%")
            print(f"Number of Trades: {results['num_trades']}")

            if 'metrics' in results:
                metrics = results['metrics']
                print(f"\nPerformance Metrics:")
                print(f"  Annualized Return: {metrics.get('annualized_return_pct', 0):.2f}%")
                print(f"  Volatility: {metrics.get('volatility_pct', 0):.2f}%")
                print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
                print(f"  Max Drawdown: {metrics.get('max_drawdown_pct', 0):.2f}%")
                print(f"  Win Rate: {metrics.get('win_rate_pct', 0):.2f}%")
            print("="*70 + "\n")

            # Generate detailed report
            report = engine.dashboard.generate_performance_report(results)
            print(report)

        elif args.mode == 'compare':
            # Compare strategies
            logger.info(f"Comparing strategies for {args.symbol}")

            comparison = engine.compare_strategies(
                symbol=args.symbol,
                start_date=args.start_date,
                end_date=args.end_date
            )

            print("\n" + "="*70)
            print("STRATEGY COMPARISON")
            print("="*70)
            print(comparison.to_string(index=False))
            print("="*70 + "\n")

        elif args.mode == 'optimize':
            # Portfolio optimization
            symbols = args.symbols or ['AAPL', 'MSFT', 'GOOGL', 'AMZN']

            logger.info(f"Optimizing portfolio for {len(symbols)} symbols")

            result = engine.optimize_portfolio(
                symbols=symbols,
                method=args.optimize_method
            )

            print("\n" + "="*70)
            print(f"PORTFOLIO OPTIMIZATION - {args.optimize_method.upper()}")
            print("="*70)
            print(f"Expected Return: {result['expected_return']:.2%}")
            print(f"Volatility: {result['volatility']:.2%}")
            print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
            print(f"\nOptimal Weights:")

            for symbol, weight in sorted(result['weights'].items(), key=lambda x: x[1], reverse=True):
                print(f"  {symbol}: {weight:.2%}")
            print("="*70 + "\n")

        elif args.mode == 'ml':
            # Machine learning prediction
            logger.info(f"Training ML model for {args.symbol}")

            # Train model
            train_results = engine.train_ml_model(args.symbol)

            print("\n" + "="*70)
            print("ML MODEL TRAINING RESULTS")
            print("="*70)
            print(f"Model Type: {train_results['model_type']}")
            print(f"Training Samples: {train_results['train_samples']}")
            print(f"Test Samples: {train_results['test_samples']}")

            metrics = train_results['metrics']
            print(f"\nModel Performance:")
            print(f"  R² Score: {metrics.get('r2', 0):.4f}")
            print(f"  RMSE: {metrics.get('rmse', 0):.4f}")
            print(f"  MAE: {metrics.get('mae', 0):.4f}")
            print(f"  MAPE: {metrics.get('mape', 0):.2f}%")

            # Make predictions
            predictions = engine.predict_price(args.symbol, steps=5)

            print(f"\nPrice Predictions (next 5 periods):")
            for i, pred in enumerate(predictions, 1):
                print(f"  Day {i}: ${pred:.2f}")
            print("="*70 + "\n")

        elif args.mode == 'paper':
            # Paper trading
            symbols = args.symbols or [args.symbol]

            logger.info(f"Starting paper trading for {symbols}")

            print("\n" + "="*70)
            print("PAPER TRADING MODE")
            print("="*70)
            print(f"Symbols: {', '.join(symbols)}")
            print(f"Strategy: {args.strategy.upper()}")
            print("Press Ctrl+C to stop trading")
            print("="*70 + "\n")

            try:
                engine.start_paper_trading(symbols, args.strategy)
            except KeyboardInterrupt:
                logger.info("Stopping paper trading...")
                engine.stop_trading()

            # Print summary
            summary = engine.get_performance_summary()

            print("\n" + "="*70)
            print("PAPER TRADING SUMMARY")
            print("="*70)
            print(f"Initial Capital: ${summary.get('initial_capital', 0):,.2f}")
            print(f"Current Value: ${summary.get('current_value', 0):,.2f}")
            print(f"Total Return: {summary.get('total_return_pct', 0):.2f}%")
            print(f"Total Trades: {summary.get('total_trades', 0)}")
            print(f"Win Rate: {summary.get('win_rate_pct', 0):.2f}%")
            print("="*70 + "\n")

        elif args.mode == 'dashboard':
            # Launch dashboard
            logger.info("Launching analytics dashboard...")

            print("\n" + "="*70)
            print("ANALYTICS DASHBOARD")
            print("="*70)
            print("Dashboard will open in your web browser")
            print("Default URL: http://127.0.0.1:8050")
            print("Press Ctrl+C to stop the dashboard")
            print("="*70 + "\n")

            engine.run_dashboard()

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║    AUTONOMOUS TRADING BOT & AUDIO NOISE CANCELLATION v2.0            ║
║                                                                      ║
║  Trading: Advanced Bot with ML, Risk Management & Portfolio Optimization║
║  Audio: Real-time Noise Cancellation with Low-Latency Processing     ║
║                  Full Windows 11 Compatibility                       ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

    main()
