"""
Alert Manager
Windows 11 compatible notification system
"""
import platform
from datetime import datetime
from typing import Dict, List, Any, Optional
from enum import Enum
import json
from pathlib import Path


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4


class AlertManager:
    """Manages alerts and notifications for the greenhouse system"""

    def __init__(self, data_dir: str = "greenhouse_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.alert_history: List[Dict[str, Any]] = []
        self.max_history = 100
        self.active_alerts: List[Dict[str, Any]] = []
        self.notification_enabled = True

        # Detect OS
        self.os_type = platform.system()
        print(f"Alert system initialized for {self.os_type}")

    def send_alert(self,
                   title: str,
                   message: str,
                   level: AlertLevel = AlertLevel.INFO,
                   category: str = "general") -> bool:
        """
        Send an alert/notification
        Returns True if notification was sent successfully
        """
        alert = {
            'timestamp': datetime.now().isoformat(),
            'title': title,
            'message': message,
            'level': level.name,
            'category': category,
            'acknowledged': False
        }

        # Add to history
        self.alert_history.append(alert)
        if len(self.alert_history) > self.max_history:
            self.alert_history = self.alert_history[-self.max_history:]

        # Add to active alerts if WARNING or higher
        if level in [AlertLevel.WARNING, AlertLevel.ERROR, AlertLevel.CRITICAL]:
            self.active_alerts.append(alert)

        # Log alert
        self._log_alert(alert)

        # Send notification
        if self.notification_enabled:
            return self._send_notification(title, message, level)

        return True

    def _send_notification(self, title: str, message: str, level: AlertLevel) -> bool:
        """Send platform-specific notification"""

        if self.os_type == "Windows":
            return self._send_windows_notification(title, message, level)
        elif self.os_type == "Darwin":  # macOS
            return self._send_macos_notification(title, message)
        elif self.os_type == "Linux":
            return self._send_linux_notification(title, message)
        else:
            print(f"[{level.name}] {title}: {message}")
            return True

    def _send_windows_notification(self, title: str, message: str, level: AlertLevel) -> bool:
        """Send Windows 11 toast notification"""
        try:
            from win10toast import ToastNotifier
            toaster = ToastNotifier()

            # Set icon based on level
            icon_path = None  # Can add custom icon path

            # Send notification
            toaster.show_toast(
                title=f"🌱 Greenhouse: {title}",
                msg=message,
                duration=10,  # seconds
                icon_path=icon_path,
                threaded=True
            )
            return True

        except ImportError:
            # Fallback to winotify for Windows 11
            try:
                from winotify import Notification, audio

                toast = Notification(
                    app_id="Smart Greenhouse System",
                    title=f"🌱 {title}",
                    msg=message,
                    duration="short"
                )

                # Set audio based on level
                if level == AlertLevel.CRITICAL:
                    toast.set_audio(audio.LoopingAlarm, loop=False)
                elif level == AlertLevel.ERROR:
                    toast.set_audio(audio.Default, loop=False)

                toast.show()
                return True

            except ImportError:
                # Final fallback - print to console
                print(f"\n{'='*60}")
                print(f"[{level.name}] {title}")
                print(f"{message}")
                print(f"{'='*60}\n")
                return False

        except Exception as e:
            print(f"Error sending Windows notification: {e}")
            return False

    def _send_macos_notification(self, title: str, message: str) -> bool:
        """Send macOS notification"""
        try:
            import subprocess
            script = f'display notification "{message}" with title "🌱 Greenhouse: {title}"'
            subprocess.run(["osascript", "-e", script], check=True)
            return True
        except Exception as e:
            print(f"Error sending macOS notification: {e}")
            return False

    def _send_linux_notification(self, title: str, message: str) -> bool:
        """Send Linux notification"""
        try:
            import subprocess
            subprocess.run([
                "notify-send",
                f"🌱 Greenhouse: {title}",
                message
            ], check=True)
            return True
        except Exception as e:
            print(f"Error sending Linux notification: {e}")
            return False

    def check_sensor_alerts(self, readings: Dict[str, Dict[str, Any]], thresholds: Dict[str, Dict[str, float]]):
        """Check sensor readings and send alerts if thresholds exceeded"""
        for sensor_id, reading in readings.items():
            if sensor_id in thresholds and 'value' in reading:
                value = reading['value']
                threshold = thresholds[sensor_id]
                sensor_name = reading.get('name', sensor_id)

                # Check minimum threshold
                if 'min' in threshold and value < threshold['min']:
                    self.send_alert(
                        title=f"{sensor_name} Too Low",
                        message=f"{sensor_name} is {value:.1f} {reading.get('unit', '')} (below {threshold['min']})",
                        level=AlertLevel.WARNING,
                        category="sensor"
                    )

                # Check maximum threshold
                if 'max' in threshold and value > threshold['max']:
                    self.send_alert(
                        title=f"{sensor_name} Too High",
                        message=f"{sensor_name} is {value:.1f} {reading.get('unit', '')} (above {threshold['max']})",
                        level=AlertLevel.WARNING,
                        category="sensor"
                    )

    def send_daily_summary(self, stats: Dict[str, Any]):
        """Send daily summary notification"""
        summary_lines = ["Daily Greenhouse Summary:"]

        for sensor_id, sensor_stats in stats.items():
            name = sensor_stats.get('name', sensor_id)
            avg = sensor_stats.get('avg', 0)
            unit = sensor_stats.get('unit', '')
            summary_lines.append(f"• {name}: {avg:.1f} {unit} avg")

        message = "\n".join(summary_lines)

        self.send_alert(
            title="Daily Summary",
            message=message,
            level=AlertLevel.INFO,
            category="summary"
        )

    def send_system_alert(self, message: str, level: AlertLevel = AlertLevel.ERROR):
        """Send system-related alert"""
        self.send_alert(
            title="System Alert",
            message=message,
            level=level,
            category="system"
        )

    def send_irrigation_alert(self, zone_name: str, action: str):
        """Send irrigation-related alert"""
        self.send_alert(
            title=f"Irrigation: {zone_name}",
            message=f"Watering {action}",
            level=AlertLevel.INFO,
            category="irrigation"
        )

    def send_growth_alert(self, message: str):
        """Send growth-related alert"""
        self.send_alert(
            title="Plant Growth Update",
            message=message,
            level=AlertLevel.INFO,
            category="growth"
        )

    def acknowledge_alert(self, alert_index: int):
        """Acknowledge an active alert"""
        if 0 <= alert_index < len(self.active_alerts):
            self.active_alerts[alert_index]['acknowledged'] = True

    def clear_acknowledged_alerts(self):
        """Remove acknowledged alerts from active list"""
        self.active_alerts = [a for a in self.active_alerts if not a.get('acknowledged', False)]

    def get_active_alerts(self, level: Optional[AlertLevel] = None) -> List[Dict[str, Any]]:
        """Get active alerts, optionally filtered by level"""
        if level:
            return [a for a in self.active_alerts if a['level'] == level.name]
        return self.active_alerts

    def get_alert_history(self, limit: int = 50, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get alert history"""
        history = self.alert_history[-limit:]

        if category:
            history = [a for a in history if a.get('category') == category]

        return history

    def _log_alert(self, alert: Dict[str, Any]):
        """Log alert to file"""
        timestamp = datetime.now().strftime("%Y%m%d")
        filename = self.data_dir / f"alerts_{timestamp}.jsonl"

        try:
            with open(filename, 'a') as f:
                json.dump(alert, f)
                f.write('\n')
        except Exception as e:
            print(f"Error logging alert: {e}")

    def enable_notifications(self):
        """Enable notifications"""
        self.notification_enabled = True
        print("Notifications enabled")

    def disable_notifications(self):
        """Disable notifications"""
        self.notification_enabled = False
        print("Notifications disabled")

    def test_notification(self):
        """Test notification system"""
        return self.send_alert(
            title="Test Notification",
            message="This is a test notification from Smart Greenhouse System",
            level=AlertLevel.INFO,
            category="test"
        )
