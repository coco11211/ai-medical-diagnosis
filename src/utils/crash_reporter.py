"""
Crash Reporter - Production error handling and reporting
"""
import sys
import os
import traceback
import logging
from datetime import datetime
from pathlib import Path
import json


class CrashReporter:
    """Production-grade crash reporter for Windows applications"""

    def __init__(self, app_name="Trading Bot Simulator", log_dir=None):
        self.app_name = app_name
        self.log_dir = log_dir or self._get_log_directory()
        self._ensure_log_directory()
        self._setup_logging()

    def _get_log_directory(self) -> Path:
        """Get application log directory (Windows AppData)"""
        if sys.platform == 'win32':
            appdata = os.getenv('APPDATA')
            log_dir = Path(appdata) / self.app_name / 'logs'
        else:
            log_dir = Path.home() / '.trading-bot-simulator' / 'logs'

        return log_dir

    def _ensure_log_directory(self):
        """Ensure log directory exists"""
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _setup_logging(self):
        """Setup application logging"""
        log_file = self.log_dir / f'app_{datetime.now().strftime("%Y%m%d")}.log'

        # Configure root logger
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )

        self.logger = logging.getLogger(self.app_name)

    def log_exception(self, exc_type, exc_value, exc_traceback):
        """Log unhandled exception"""
        if issubclass(exc_type, KeyboardInterrupt):
            # Don't log keyboard interrupts
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        # Log the exception
        self.logger.critical(
            "Unhandled exception",
            exc_info=(exc_type, exc_value, exc_traceback)
        )

        # Create crash report
        self._create_crash_report(exc_type, exc_value, exc_traceback)

    def _create_crash_report(self, exc_type, exc_value, exc_traceback):
        """Create detailed crash report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        crash_file = self.log_dir / f'crash_report_{timestamp}.json'

        # Gather system information
        crash_data = {
            'timestamp': datetime.now().isoformat(),
            'exception': {
                'type': exc_type.__name__,
                'message': str(exc_value),
                'traceback': ''.join(traceback.format_exception(
                    exc_type, exc_value, exc_traceback
                ))
            },
            'system': {
                'platform': sys.platform,
                'python_version': sys.version,
                'executable': sys.executable,
            }
        }

        # Save crash report
        with open(crash_file, 'w') as f:
            json.dump(crash_data, f, indent=2)

        self.logger.info(f"Crash report saved to: {crash_file}")

    def install_exception_handler(self):
        """Install global exception handler"""
        sys.excepthook = self.log_exception

    def log_info(self, message: str):
        """Log info message"""
        self.logger.info(message)

    def log_warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)

    def log_error(self, message: str):
        """Log error message"""
        self.logger.error(message)

    def get_recent_logs(self, lines: int = 100) -> str:
        """Get recent log entries"""
        log_file = self.log_dir / f'app_{datetime.now().strftime("%Y%m%d")}.log'

        if not log_file.exists():
            return "No log file found for today"

        with open(log_file, 'r') as f:
            all_lines = f.readlines()
            recent_lines = all_lines[-lines:]
            return ''.join(recent_lines)
