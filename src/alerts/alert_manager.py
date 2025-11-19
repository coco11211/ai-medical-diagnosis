"""
Alert management system for trading bot.
"""
import logging
from typing import Dict, List, Optional
from datetime import datetime
import platform

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import Windows toast notification library
try:
    if platform.system() == 'Windows':
        from win10toast import ToastNotifier
        TOAST_AVAILABLE = True
    else:
        TOAST_AVAILABLE = False
except ImportError:
    TOAST_AVAILABLE = False
    logger.warning("win10toast not available. Desktop notifications disabled.")


class AlertManager:
    """
    Alert manager for trading bot notifications.
    """

    def __init__(
        self,
        enable_email: bool = False,
        enable_desktop: bool = True,
        alert_on_trade: bool = True,
        alert_on_signal: bool = True,
        alert_on_risk_breach: bool = True
    ):
        """
        Initialize alert manager.

        Args:
            enable_email: Enable email alerts
            enable_desktop: Enable desktop notifications
            alert_on_trade: Alert on trade execution
            alert_on_signal: Alert on trading signals
            alert_on_risk_breach: Alert on risk limit breaches
        """
        self.enable_email = enable_email
        self.enable_desktop = enable_desktop
        self.alert_on_trade = alert_on_trade
        self.alert_on_signal = alert_on_signal
        self.alert_on_risk_breach = alert_on_risk_breach

        self.alerts_history = []

        if TOAST_AVAILABLE and self.enable_desktop:
            self.toaster = ToastNotifier()
        else:
            self.toaster = None

    def send_alert(
        self,
        title: str,
        message: str,
        alert_type: str = "INFO",
        priority: str = "NORMAL"
    ) -> None:
        """
        Send an alert.

        Args:
            title: Alert title
            message: Alert message
            alert_type: Type of alert (INFO, WARNING, ERROR, TRADE, SIGNAL)
            priority: Priority level (LOW, NORMAL, HIGH, CRITICAL)
        """
        alert = {
            'timestamp': datetime.now(),
            'title': title,
            'message': message,
            'type': alert_type,
            'priority': priority
        }

        self.alerts_history.append(alert)

        # Log alert
        log_message = f"[{alert_type}] {title}: {message}"

        if alert_type == "ERROR" or priority == "CRITICAL":
            logger.error(log_message)
        elif alert_type == "WARNING" or priority == "HIGH":
            logger.warning(log_message)
        else:
            logger.info(log_message)

        # Desktop notification
        if self.enable_desktop:
            self._send_desktop_notification(title, message, alert_type)

        # Email notification
        if self.enable_email:
            self._send_email_notification(title, message, alert_type)

    def _send_desktop_notification(
        self,
        title: str,
        message: str,
        alert_type: str
    ) -> None:
        """
        Send desktop notification (Windows 11 compatible).

        Args:
            title: Notification title
            message: Notification message
            alert_type: Type of alert
        """
        if self.toaster is not None:
            try:
                self.toaster.show_toast(
                    title,
                    message,
                    duration=5,
                    threaded=True
                )
            except Exception as e:
                logger.error(f"Failed to send desktop notification: {e}")
        else:
            # Fallback to console output
            print(f"\n{'='*60}")
            print(f"🔔 ALERT: {title}")
            print(f"Type: {alert_type}")
            print(f"Message: {message}")
            print(f"{'='*60}\n")

    def _send_email_notification(
        self,
        title: str,
        message: str,
        alert_type: str
    ) -> None:
        """
        Send email notification (placeholder).

        Args:
            title: Email subject
            message: Email body
            alert_type: Type of alert
        """
        # Email functionality would require SMTP configuration
        # This is a placeholder for future implementation
        logger.info(f"Email alert: {title} - {message}")

    def alert_trade_executed(
        self,
        symbol: str,
        trade_type: str,
        shares: int,
        price: float,
        profit: float = None
    ) -> None:
        """
        Alert on trade execution.

        Args:
            symbol: Asset symbol
            trade_type: BUY or SELL
            shares: Number of shares
            price: Execution price
            profit: Profit/loss (for SELL trades)
        """
        if not self.alert_on_trade:
            return

        if trade_type == "SELL" and profit is not None:
            profit_str = f"Profit: ${profit:.2f}" if profit > 0 else f"Loss: ${abs(profit):.2f}"
            message = f"{trade_type} {shares} shares of {symbol} @ ${price:.2f}\n{profit_str}"
        else:
            message = f"{trade_type} {shares} shares of {symbol} @ ${price:.2f}"

        self.send_alert(
            title=f"Trade Executed: {symbol}",
            message=message,
            alert_type="TRADE",
            priority="NORMAL"
        )

    def alert_signal_generated(
        self,
        symbol: str,
        signal: str,
        strategy: str,
        strength: float = None
    ) -> None:
        """
        Alert on trading signal.

        Args:
            symbol: Asset symbol
            signal: Signal type (BUY, SELL)
            strategy: Strategy name
            strength: Signal strength (0-1)
        """
        if not self.alert_on_signal:
            return

        strength_str = f" (Strength: {strength:.2f})" if strength else ""
        message = f"{signal} signal for {symbol} from {strategy}{strength_str}"

        self.send_alert(
            title=f"Trading Signal: {symbol}",
            message=message,
            alert_type="SIGNAL",
            priority="NORMAL"
        )

    def alert_risk_breach(
        self,
        risk_type: str,
        current_value: float,
        limit_value: float,
        message: str = None
    ) -> None:
        """
        Alert on risk limit breach.

        Args:
            risk_type: Type of risk (MAX_DRAWDOWN, STOP_LOSS, etc.)
            current_value: Current value
            limit_value: Limit value
            message: Optional custom message
        """
        if not self.alert_on_risk_breach:
            return

        if message is None:
            message = f"{risk_type} breached! Current: {current_value:.2f}, Limit: {limit_value:.2f}"

        self.send_alert(
            title=f"Risk Breach: {risk_type}",
            message=message,
            alert_type="WARNING",
            priority="HIGH"
        )

    def alert_error(self, error_message: str, details: str = None) -> None:
        """
        Alert on error.

        Args:
            error_message: Error message
            details: Additional details
        """
        message = error_message
        if details:
            message += f"\nDetails: {details}"

        self.send_alert(
            title="Trading Bot Error",
            message=message,
            alert_type="ERROR",
            priority="CRITICAL"
        )

    def alert_daily_summary(self, summary: Dict) -> None:
        """
        Send daily performance summary.

        Args:
            summary: Summary dictionary
        """
        message = f"""
Daily Trading Summary:
Portfolio Value: ${summary.get('portfolio_value', 0):,.2f}
Total Return: {summary.get('total_return_pct', 0):.2f}%
Trades Today: {summary.get('trades_today', 0)}
Win Rate: {summary.get('win_rate_pct', 0):.2f}%
        """.strip()

        self.send_alert(
            title="Daily Summary",
            message=message,
            alert_type="INFO",
            priority="NORMAL"
        )

    def alert_position_update(
        self,
        symbol: str,
        action: str,
        details: Dict
    ) -> None:
        """
        Alert on position update.

        Args:
            symbol: Asset symbol
            action: Action type (OPENED, CLOSED, MODIFIED)
            details: Position details
        """
        message = f"Position {action.lower()} for {symbol}\n"

        if 'shares' in details:
            message += f"Shares: {details['shares']}\n"

        if 'price' in details:
            message += f"Price: ${details['price']:.2f}\n"

        if 'profit_pct' in details:
            message += f"Return: {details['profit_pct']:.2f}%"

        self.send_alert(
            title=f"Position {action}: {symbol}",
            message=message.strip(),
            alert_type="INFO",
            priority="NORMAL"
        )

    def get_alerts_history(
        self,
        limit: int = 100,
        alert_type: Optional[str] = None
    ) -> List[Dict]:
        """
        Get alerts history.

        Args:
            limit: Maximum number of alerts to return
            alert_type: Filter by alert type

        Returns:
            List of alert dictionaries
        """
        alerts = self.alerts_history

        if alert_type:
            alerts = [a for a in alerts if a['type'] == alert_type]

        return alerts[-limit:]

    def clear_alerts_history(self) -> None:
        """Clear alerts history."""
        self.alerts_history = []
        logger.info("Alerts history cleared")

    def get_alert_stats(self) -> Dict:
        """
        Get alert statistics.

        Returns:
            Dictionary with alert statistics
        """
        total_alerts = len(self.alerts_history)

        alert_types = {}
        for alert in self.alerts_history:
            alert_type = alert['type']
            alert_types[alert_type] = alert_types.get(alert_type, 0) + 1

        return {
            'total_alerts': total_alerts,
            'by_type': alert_types,
            'last_alert': self.alerts_history[-1] if self.alerts_history else None
        }
