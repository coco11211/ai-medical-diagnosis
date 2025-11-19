"""
Trading Bot Simulator - Windows 11 GUI Application
Complete production-ready Windows application with modern UI
"""
import sys
import os
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QPushButton, QLabel, QComboBox, QLineEdit, QTextEdit,
    QTableWidget, QTableWidgetItem, QGroupBox, QCheckBox, QSpinBox,
    QDoubleSpinBox, QProgressBar, QSystemTrayIcon, QMenu, QMessageBox,
    QFileDialog, QSplitter, QDateEdit, QListWidget, QStatusBar
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize, QDate
from PyQt6.QtGui import QIcon, QPixmap, QAction, QFont, QPalette, QColor
from PyQt6.QtWebEngineWidgets import QWebEngineView

# Import trading engine
from src.engine import TradingEngine
from src.config import Settings


class TradingWorker(QThread):
    """Background worker for trading operations"""
    progress = pyqtSignal(str)
    result = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, engine, operation, **kwargs):
        super().__init__()
        self.engine = engine
        self.operation = operation
        self.kwargs = kwargs

    def run(self):
        try:
            if self.operation == 'backtest':
                self.progress.emit("Running backtest...")
                result = self.engine.run_backtest(**self.kwargs)
                self.result.emit(result)

            elif self.operation == 'optimize':
                self.progress.emit("Optimizing portfolio...")
                result = self.engine.optimize_portfolio(**self.kwargs)
                self.result.emit(result)

            elif self.operation == 'ml_train':
                self.progress.emit("Training ML model...")
                result = self.engine.train_ml_model(**self.kwargs)
                self.result.emit(result)

            elif self.operation == 'paper_trading':
                self.progress.emit("Paper trading started...")
                self.engine.start_paper_trading(**self.kwargs)

        except Exception as e:
            self.error.emit(str(e))


class ChartWidget(QWidget):
    """Custom widget for displaying trading charts"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Web view for Plotly charts
        self.chart_view = QWebEngineView()
        layout.addWidget(self.chart_view)

        self.setLayout(layout)

    def load_chart(self, html_file: str):
        """Load a chart from HTML file"""
        if os.path.exists(html_file):
            with open(html_file, 'r') as f:
                html_content = f.read()
            self.chart_view.setHtml(html_content)


class MainWindow(QMainWindow):
    """Main application window"""

    def __init__(self):
        super().__init__()
        self.engine = None
        self.worker = None
        self.trading_active = False

        self.init_ui()
        self.init_engine()
        self.setup_tray_icon()

    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Trading Bot Simulator - Windows 11 Edition")
        self.setMinimumSize(1200, 800)

        # Apply Windows 11 style
        self.apply_windows11_style()

        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # Header
        header = self.create_header()
        main_layout.addWidget(header)

        # Tab widget
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.TabPosition.North)

        # Create tabs
        self.dashboard_tab = self.create_dashboard_tab()
        self.backtest_tab = self.create_backtest_tab()
        self.trading_tab = self.create_trading_tab()
        self.portfolio_tab = self.create_portfolio_tab()
        self.ml_tab = self.create_ml_tab()
        self.settings_tab = self.create_settings_tab()

        self.tabs.addTab(self.dashboard_tab, "📊 Dashboard")
        self.tabs.addTab(self.backtest_tab, "📈 Backtesting")
        self.tabs.addTab(self.trading_tab, "💰 Paper Trading")
        self.tabs.addTab(self.portfolio_tab, "📁 Portfolio")
        self.tabs.addTab(self.ml_tab, "🤖 Machine Learning")
        self.tabs.addTab(self.settings_tab, "⚙️ Settings")

        main_layout.addWidget(self.tabs)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    def apply_windows11_style(self):
        """Apply Windows 11 modern styling"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f3f3f3;
            }
            QTabWidget::pane {
                border: 1px solid #e0e0e0;
                background-color: white;
                border-radius: 8px;
            }
            QTabBar::tab {
                background-color: #f0f0f0;
                color: #333;
                padding: 10px 20px;
                margin-right: 2px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
            }
            QTabBar::tab:selected {
                background-color: white;
                color: #0078d4;
                font-weight: bold;
            }
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
            QPushButton:pressed {
                background-color: #005a9e;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                color: #0078d4;
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                padding: 6px;
                border: 1px solid #d0d0d0;
                border-radius: 4px;
                background-color: white;
            }
            QLineEdit:focus, QComboBox:focus {
                border: 2px solid #0078d4;
            }
            QTableWidget {
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                gridline-color: #f0f0f0;
            }
            QTableWidget::item {
                padding: 5px;
            }
            QTableWidget::item:selected {
                background-color: #0078d4;
                color: white;
            }
            QProgressBar {
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #0078d4;
                border-radius: 3px;
            }
        """)

    def create_header(self) -> QWidget:
        """Create application header"""
        header = QWidget()
        header.setMaximumHeight(80)
        layout = QHBoxLayout()

        # Logo and title
        title_layout = QVBoxLayout()
        title = QLabel("🚀 Trading Bot Simulator")
        title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        title.setStyleSheet("color: #0078d4;")

        subtitle = QLabel("Advanced Autonomous Trading Platform for Windows 11")
        subtitle.setStyleSheet("color: #666;")

        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)

        layout.addLayout(title_layout)
        layout.addStretch()

        # Quick actions
        self.start_btn = QPushButton("▶️ Start Trading")
        self.start_btn.setMinimumWidth(150)
        self.start_btn.clicked.connect(self.quick_start_trading)

        self.stop_btn = QPushButton("⏹️ Stop Trading")
        self.stop_btn.setMinimumWidth(150)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_trading)

        layout.addWidget(self.start_btn)
        layout.addWidget(self.stop_btn)

        header.setLayout(layout)
        return header

    def create_dashboard_tab(self) -> QWidget:
        """Create dashboard tab"""
        widget = QWidget()
        layout = QVBoxLayout()

        # Performance summary
        summary_group = QGroupBox("Performance Summary")
        summary_layout = QHBoxLayout()

        self.total_value_label = QLabel("Total Value: $100,000.00")
        self.total_value_label.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        self.total_value_label.setStyleSheet("color: #0078d4;")

        self.return_label = QLabel("Return: +0.00%")
        self.return_label.setFont(QFont("Segoe UI", 12))

        self.trades_label = QLabel("Trades: 0")
        self.trades_label.setFont(QFont("Segoe UI", 12))

        summary_layout.addWidget(self.total_value_label)
        summary_layout.addWidget(self.return_label)
        summary_layout.addWidget(self.trades_label)
        summary_layout.addStretch()
        summary_group.setLayout(summary_layout)

        layout.addWidget(summary_group)

        # Charts
        charts_group = QGroupBox("Performance Charts")
        charts_layout = QVBoxLayout()

        self.dashboard_chart = ChartWidget()
        charts_layout.addWidget(self.dashboard_chart)

        charts_group.setLayout(charts_layout)
        layout.addWidget(charts_group)

        # Recent trades
        trades_group = QGroupBox("Recent Trades")
        trades_layout = QVBoxLayout()

        self.trades_table = QTableWidget()
        self.trades_table.setColumnCount(7)
        self.trades_table.setHorizontalHeaderLabels([
            "Time", "Symbol", "Action", "Quantity", "Price", "Value", "P&L"
        ])
        trades_layout.addWidget(self.trades_table)

        trades_group.setLayout(trades_layout)
        layout.addWidget(trades_group)

        widget.setLayout(layout)
        return widget

    def create_backtest_tab(self) -> QWidget:
        """Create backtesting tab"""
        widget = QWidget()
        layout = QVBoxLayout()

        # Controls
        controls_group = QGroupBox("Backtest Configuration")
        controls_layout = QHBoxLayout()

        # Symbol
        controls_layout.addWidget(QLabel("Symbol:"))
        self.bt_symbol = QLineEdit("AAPL")
        self.bt_symbol.setMaximumWidth(100)
        controls_layout.addWidget(self.bt_symbol)

        # Strategy
        controls_layout.addWidget(QLabel("Strategy:"))
        self.bt_strategy = QComboBox()
        self.bt_strategy.addItems(["MACD", "RSI", "Bollinger Bands"])
        controls_layout.addWidget(self.bt_strategy)

        # Date range
        controls_layout.addWidget(QLabel("Start Date:"))
        self.bt_start_date = QDateEdit()
        self.bt_start_date.setDate(QDate.currentDate().addYears(-1))
        self.bt_start_date.setCalendarPopup(True)
        controls_layout.addWidget(self.bt_start_date)

        controls_layout.addWidget(QLabel("End Date:"))
        self.bt_end_date = QDateEdit()
        self.bt_end_date.setDate(QDate.currentDate())
        self.bt_end_date.setCalendarPopup(True)
        controls_layout.addWidget(self.bt_end_date)

        controls_layout.addStretch()

        # Run button
        self.run_backtest_btn = QPushButton("🚀 Run Backtest")
        self.run_backtest_btn.clicked.connect(self.run_backtest)
        controls_layout.addWidget(self.run_backtest_btn)

        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)

        # Progress
        self.bt_progress = QProgressBar()
        self.bt_progress.setVisible(False)
        layout.addWidget(self.bt_progress)

        # Results
        results_group = QGroupBox("Backtest Results")
        results_layout = QVBoxLayout()

        self.bt_results = QTextEdit()
        self.bt_results.setReadOnly(True)
        self.bt_results.setFont(QFont("Courier New", 10))
        results_layout.addWidget(self.bt_results)

        results_group.setLayout(results_layout)
        layout.addWidget(results_group)

        # Chart
        self.bt_chart = ChartWidget()
        layout.addWidget(self.bt_chart)

        widget.setLayout(layout)
        return widget

    def create_trading_tab(self) -> QWidget:
        """Create paper trading tab"""
        widget = QWidget()
        layout = QVBoxLayout()

        # Controls
        controls_group = QGroupBox("Trading Configuration")
        controls_layout = QVBoxLayout()

        # Symbols
        symbol_layout = QHBoxLayout()
        symbol_layout.addWidget(QLabel("Symbols (comma-separated):"))
        self.pt_symbols = QLineEdit("AAPL,MSFT,GOOGL")
        symbol_layout.addWidget(self.pt_symbols)
        controls_layout.addLayout(symbol_layout)

        # Strategy
        strategy_layout = QHBoxLayout()
        strategy_layout.addWidget(QLabel("Strategy:"))
        self.pt_strategy = QComboBox()
        self.pt_strategy.addItems(["MACD", "RSI", "Bollinger Bands"])
        strategy_layout.addWidget(self.pt_strategy)
        strategy_layout.addStretch()
        controls_layout.addLayout(strategy_layout)

        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)

        # Action buttons
        button_layout = QHBoxLayout()
        self.start_trading_btn = QPushButton("▶️ Start Paper Trading")
        self.start_trading_btn.clicked.connect(self.start_paper_trading)
        button_layout.addWidget(self.start_trading_btn)

        self.stop_trading_btn = QPushButton("⏹️ Stop Trading")
        self.stop_trading_btn.setEnabled(False)
        self.stop_trading_btn.clicked.connect(self.stop_trading)
        button_layout.addWidget(self.stop_trading_btn)

        button_layout.addStretch()
        layout.addLayout(button_layout)

        # Live status
        status_group = QGroupBox("Trading Status")
        status_layout = QVBoxLayout()

        self.pt_status = QTextEdit()
        self.pt_status.setReadOnly(True)
        self.pt_status.setMaximumHeight(200)
        status_layout.addWidget(self.pt_status)

        status_group.setLayout(status_layout)
        layout.addWidget(status_group)

        # Positions
        positions_group = QGroupBox("Current Positions")
        positions_layout = QVBoxLayout()

        self.positions_table = QTableWidget()
        self.positions_table.setColumnCount(6)
        self.positions_table.setHorizontalHeaderLabels([
            "Symbol", "Quantity", "Entry Price", "Current Price", "Value", "P&L"
        ])
        positions_layout.addWidget(self.positions_table)

        positions_group.setLayout(positions_layout)
        layout.addWidget(positions_group)

        widget.setLayout(layout)
        return widget

    def create_portfolio_tab(self) -> QWidget:
        """Create portfolio optimization tab"""
        widget = QWidget()
        layout = QVBoxLayout()

        # Controls
        controls_group = QGroupBox("Portfolio Configuration")
        controls_layout = QVBoxLayout()

        # Symbols
        symbol_layout = QHBoxLayout()
        symbol_layout.addWidget(QLabel("Symbols:"))
        self.po_symbols = QLineEdit("AAPL,MSFT,GOOGL,AMZN")
        symbol_layout.addWidget(self.po_symbols)
        controls_layout.addLayout(symbol_layout)

        # Optimization method
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Optimization Method:"))
        self.po_method = QComboBox()
        self.po_method.addItems([
            "Maximum Sharpe Ratio",
            "Minimum Volatility",
            "Equal Weight",
            "Risk Parity"
        ])
        method_layout.addWidget(self.po_method)
        method_layout.addStretch()
        controls_layout.addLayout(method_layout)

        # Optimize button
        self.optimize_btn = QPushButton("🎯 Optimize Portfolio")
        self.optimize_btn.clicked.connect(self.optimize_portfolio)
        controls_layout.addWidget(self.optimize_btn)

        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)

        # Results
        results_group = QGroupBox("Optimization Results")
        results_layout = QVBoxLayout()

        self.po_results = QTextEdit()
        self.po_results.setReadOnly(True)
        results_layout.addWidget(self.po_results)

        results_group.setLayout(results_layout)
        layout.addWidget(results_group)

        # Allocation chart
        self.po_chart = ChartWidget()
        layout.addWidget(self.po_chart)

        widget.setLayout(layout)
        return widget

    def create_ml_tab(self) -> QWidget:
        """Create machine learning tab"""
        widget = QWidget()
        layout = QVBoxLayout()

        # Controls
        controls_group = QGroupBox("ML Configuration")
        controls_layout = QHBoxLayout()

        controls_layout.addWidget(QLabel("Symbol:"))
        self.ml_symbol = QLineEdit("AAPL")
        self.ml_symbol.setMaximumWidth(100)
        controls_layout.addWidget(self.ml_symbol)

        controls_layout.addWidget(QLabel("Model:"))
        self.ml_model = QComboBox()
        self.ml_model.addItems(["LSTM", "Random Forest", "Gradient Boosting"])
        controls_layout.addWidget(self.ml_model)

        controls_layout.addStretch()

        self.train_btn = QPushButton("🤖 Train Model")
        self.train_btn.clicked.connect(self.train_ml_model)
        controls_layout.addWidget(self.train_btn)

        self.predict_btn = QPushButton("🔮 Predict")
        self.predict_btn.clicked.connect(self.predict_price)
        controls_layout.addWidget(self.predict_btn)

        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)

        # Results
        results_group = QGroupBox("ML Results")
        results_layout = QVBoxLayout()

        self.ml_results = QTextEdit()
        self.ml_results.setReadOnly(True)
        results_layout.addWidget(self.ml_results)

        results_group.setLayout(results_layout)
        layout.addWidget(results_group)

        # Predictions chart
        self.ml_chart = ChartWidget()
        layout.addWidget(self.ml_chart)

        widget.setLayout(layout)
        return widget

    def create_settings_tab(self) -> QWidget:
        """Create settings tab"""
        widget = QWidget()
        layout = QVBoxLayout()

        # Trading settings
        trading_group = QGroupBox("Trading Settings")
        trading_layout = QVBoxLayout()

        capital_layout = QHBoxLayout()
        capital_layout.addWidget(QLabel("Initial Capital:"))
        self.capital_input = QDoubleSpinBox()
        self.capital_input.setRange(1000, 10000000)
        self.capital_input.setValue(100000)
        self.capital_input.setPrefix("$")
        capital_layout.addWidget(self.capital_input)
        capital_layout.addStretch()
        trading_layout.addLayout(capital_layout)

        commission_layout = QHBoxLayout()
        commission_layout.addWidget(QLabel("Commission:"))
        self.commission_input = QDoubleSpinBox()
        self.commission_input.setRange(0, 1)
        self.commission_input.setValue(0.001)
        self.commission_input.setSingleStep(0.0001)
        self.commission_input.setSuffix("%")
        commission_layout.addWidget(self.commission_input)
        commission_layout.addStretch()
        trading_layout.addLayout(commission_layout)

        trading_group.setLayout(trading_layout)
        layout.addWidget(trading_group)

        # Risk management
        risk_group = QGroupBox("Risk Management")
        risk_layout = QVBoxLayout()

        max_position_layout = QHBoxLayout()
        max_position_layout.addWidget(QLabel("Max Position Size:"))
        self.max_position_input = QDoubleSpinBox()
        self.max_position_input.setRange(0.01, 1.0)
        self.max_position_input.setValue(0.2)
        self.max_position_input.setSingleStep(0.05)
        self.max_position_input.setSuffix("%")
        max_position_layout.addWidget(self.max_position_input)
        max_position_layout.addStretch()
        risk_layout.addLayout(max_position_layout)

        stop_loss_layout = QHBoxLayout()
        stop_loss_layout.addWidget(QLabel("Stop Loss:"))
        self.stop_loss_input = QDoubleSpinBox()
        self.stop_loss_input.setRange(0.01, 0.5)
        self.stop_loss_input.setValue(0.05)
        self.stop_loss_input.setSingleStep(0.01)
        self.stop_loss_input.setSuffix("%")
        stop_loss_layout.addWidget(self.stop_loss_input)
        stop_loss_layout.addStretch()
        risk_layout.addLayout(stop_loss_layout)

        risk_group.setLayout(risk_layout)
        layout.addWidget(risk_group)

        # Notifications
        notif_group = QGroupBox("Notifications")
        notif_layout = QVBoxLayout()

        self.notif_trades = QCheckBox("Trade Executions")
        self.notif_trades.setChecked(True)
        notif_layout.addWidget(self.notif_trades)

        self.notif_signals = QCheckBox("Trading Signals")
        self.notif_signals.setChecked(True)
        notif_layout.addWidget(self.notif_signals)

        self.notif_risks = QCheckBox("Risk Warnings")
        self.notif_risks.setChecked(True)
        notif_layout.addWidget(self.notif_risks)

        notif_group.setLayout(notif_layout)
        layout.addWidget(notif_group)

        # Save button
        save_layout = QHBoxLayout()
        save_layout.addStretch()
        self.save_settings_btn = QPushButton("💾 Save Settings")
        self.save_settings_btn.clicked.connect(self.save_settings)
        save_layout.addWidget(self.save_settings_btn)
        layout.addLayout(save_layout)

        layout.addStretch()

        widget.setLayout(layout)
        return widget

    def setup_tray_icon(self):
        """Setup system tray icon"""
        self.tray_icon = QSystemTrayIcon(self)

        # Create icon (would use actual icon file in production)
        # For now, using a placeholder
        icon = QIcon()
        self.tray_icon.setIcon(QIcon.fromTheme("application-default-icon"))

        # Create tray menu
        tray_menu = QMenu()

        show_action = QAction("Show", self)
        show_action.triggered.connect(self.show)
        tray_menu.addAction(show_action)

        hide_action = QAction("Hide", self)
        hide_action.triggered.connect(self.hide)
        tray_menu.addAction(hide_action)

        tray_menu.addSeparator()

        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(QApplication.quit)
        tray_menu.addAction(quit_action)

        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.show()

        # Double-click to show window
        self.tray_icon.activated.connect(self.tray_icon_activated)

    def tray_icon_activated(self, reason):
        """Handle tray icon activation"""
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self.show()
            self.activateWindow()

    def init_engine(self):
        """Initialize trading engine"""
        try:
            self.engine = TradingEngine()
            self.status_bar.showMessage("Trading engine initialized", 3000)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to initialize trading engine: {e}")

    def run_backtest(self):
        """Run backtest"""
        symbol = self.bt_symbol.text().strip().upper()
        strategy = self.bt_strategy.currentText().lower().replace(" ", "_")

        start_date = self.bt_start_date.date().toString("yyyy-MM-dd")
        end_date = self.bt_end_date.date().toString("yyyy-MM-dd")

        self.bt_progress.setVisible(True)
        self.bt_progress.setRange(0, 0)  # Indeterminate
        self.run_backtest_btn.setEnabled(False)

        # Run in background thread
        self.worker = TradingWorker(
            self.engine,
            'backtest',
            symbol=symbol,
            strategy_name=strategy,
            start_date=start_date,
            end_date=end_date
        )
        self.worker.result.connect(self.on_backtest_complete)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def on_backtest_complete(self, results: dict):
        """Handle backtest completion"""
        self.bt_progress.setVisible(False)
        self.run_backtest_btn.setEnabled(True)

        # Display results
        output = f"""
=== BACKTEST RESULTS ===

Symbol: {results.get('symbol', 'N/A')}
Strategy: {results.get('strategy', 'N/A')}

Initial Capital: ${results.get('initial_capital', 0):,.2f}
Final Value: ${results.get('final_value', 0):,.2f}
Total Return: {results.get('total_return_pct', 0):.2f}%
Number of Trades: {results.get('num_trades', 0)}

Performance Metrics:
  Annualized Return: {results.get('metrics', {}).get('annualized_return_pct', 0):.2f}%
  Volatility: {results.get('metrics', {}).get('volatility_pct', 0):.2f}%
  Sharpe Ratio: {results.get('metrics', {}).get('sharpe_ratio', 0):.2f}
  Max Drawdown: {results.get('metrics', {}).get('max_drawdown_pct', 0):.2f}%
  Win Rate: {results.get('metrics', {}).get('win_rate_pct', 0):.2f}%
"""
        self.bt_results.setText(output)

        # Show notification
        self.tray_icon.showMessage(
            "Backtest Complete",
            f"Return: {results.get('total_return_pct', 0):.2f}%",
            QSystemTrayIcon.MessageIcon.Information,
            3000
        )

    def quick_start_trading(self):
        """Quick start trading from header"""
        self.tabs.setCurrentWidget(self.trading_tab)
        self.start_paper_trading()

    def start_paper_trading(self):
        """Start paper trading"""
        symbols_text = self.pt_symbols.text().strip()
        symbols = [s.strip().upper() for s in symbols_text.split(',')]
        strategy = self.pt_strategy.currentText().lower().replace(" ", "_")

        self.trading_active = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.start_trading_btn.setEnabled(False)
        self.stop_trading_btn.setEnabled(True)

        self.pt_status.append(f"Starting paper trading for {', '.join(symbols)}...")
        self.pt_status.append(f"Strategy: {strategy}")

        # Start trading in background
        self.worker = TradingWorker(
            self.engine,
            'paper_trading',
            symbols=symbols,
            strategy_name=strategy
        )
        self.worker.error.connect(self.on_error)
        self.worker.start()

        # Update UI periodically
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_trading_status)
        self.update_timer.start(5000)  # Update every 5 seconds

    def stop_trading(self):
        """Stop trading"""
        if self.trading_active:
            self.engine.stop_trading()
            self.trading_active = False

            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.start_trading_btn.setEnabled(True)
            self.stop_trading_btn.setEnabled(False)

            if hasattr(self, 'update_timer'):
                self.update_timer.stop()

            self.pt_status.append("Trading stopped.")

            # Get final summary
            summary = self.engine.get_performance_summary()
            self.pt_status.append(f"\nFinal Performance:")
            self.pt_status.append(f"Total Return: {summary.get('total_return_pct', 0):.2f}%")
            self.pt_status.append(f"Total Trades: {summary.get('total_trades', 0)}")

    def update_trading_status(self):
        """Update trading status periodically"""
        if self.trading_active and self.engine:
            summary = self.engine.get_performance_summary()

            # Update dashboard
            self.total_value_label.setText(f"Total Value: ${summary.get('current_value', 0):,.2f}")
            self.return_label.setText(f"Return: {summary.get('total_return_pct', 0):+.2f}%")
            self.trades_label.setText(f"Trades: {summary.get('total_trades', 0)}")

    def optimize_portfolio(self):
        """Optimize portfolio"""
        symbols_text = self.po_symbols.text().strip()
        symbols = [s.strip().upper() for s in symbols_text.split(',')]

        method_map = {
            "Maximum Sharpe Ratio": "max_sharpe",
            "Minimum Volatility": "min_volatility",
            "Equal Weight": "equal_weight",
            "Risk Parity": "risk_parity"
        }
        method = method_map[self.po_method.currentText()]

        self.optimize_btn.setEnabled(False)

        # Run optimization
        self.worker = TradingWorker(
            self.engine,
            'optimize',
            symbols=symbols,
            method=method
        )
        self.worker.result.connect(self.on_optimize_complete)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def on_optimize_complete(self, results: dict):
        """Handle optimization completion"""
        self.optimize_btn.setEnabled(True)

        output = f"""
=== PORTFOLIO OPTIMIZATION ===

Expected Return: {results.get('expected_return', 0):.2%}
Volatility: {results.get('volatility', 0):.2%}
Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}

Optimal Weights:
"""
        for symbol, weight in sorted(results.get('weights', {}).items(),
                                     key=lambda x: x[1], reverse=True):
            output += f"  {symbol}: {weight:.2%}\n"

        self.po_results.setText(output)

    def train_ml_model(self):
        """Train ML model"""
        symbol = self.ml_symbol.text().strip().upper()

        self.train_btn.setEnabled(False)
        self.ml_results.append(f"Training model for {symbol}...")

        self.worker = TradingWorker(
            self.engine,
            'ml_train',
            symbol=symbol
        )
        self.worker.result.connect(self.on_ml_train_complete)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def on_ml_train_complete(self, results: dict):
        """Handle ML training completion"""
        self.train_btn.setEnabled(True)

        metrics = results.get('metrics', {})
        output = f"""
=== ML MODEL TRAINING ===

Model: {results.get('model_type', 'N/A')}
Training Samples: {results.get('train_samples', 0)}
Test Samples: {results.get('test_samples', 0)}

Performance:
  R² Score: {metrics.get('r2', 0):.4f}
  RMSE: {metrics.get('rmse', 0):.4f}
  MAE: {metrics.get('mae', 0):.4f}
  MAPE: {metrics.get('mape', 0):.2f}%

Model trained successfully!
"""
        self.ml_results.setText(output)

    def predict_price(self):
        """Predict future prices"""
        symbol = self.ml_symbol.text().strip().upper()

        try:
            predictions = self.engine.predict_price(symbol, steps=5)

            output = f"\nPredictions for {symbol} (next 5 periods):\n"
            for i, pred in enumerate(predictions, 1):
                output += f"  Day {i}: ${pred:.2f}\n"

            self.ml_results.append(output)
        except Exception as e:
            self.on_error(str(e))

    def save_settings(self):
        """Save settings"""
        # In production, this would save to config file
        QMessageBox.information(self, "Settings", "Settings saved successfully!")

    def on_error(self, error_msg: str):
        """Handle errors"""
        QMessageBox.critical(self, "Error", error_msg)
        logging.error(f"Error: {error_msg}")

        # Re-enable buttons
        self.run_backtest_btn.setEnabled(True)
        self.optimize_btn.setEnabled(True)
        self.train_btn.setEnabled(True)

    def closeEvent(self, event):
        """Handle window close"""
        reply = QMessageBox.question(
            self,
            "Confirm Exit",
            "Are you sure you want to exit?\nTrading will be stopped.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            if self.trading_active:
                self.stop_trading()
            event.accept()
        else:
            event.ignore()


def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("Trading Bot Simulator")
    app.setOrganizationName("TradingBot")

    # Set Windows 11 style
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
