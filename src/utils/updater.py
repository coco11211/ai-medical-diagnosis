"""
Auto-Updater - Automatic application updates for Windows
"""
import os
import sys
import json
import logging
import requests
from pathlib import Path
from typing import Dict, Optional, Tuple
from packaging import version


class AutoUpdater:
    """Automatic update checker and downloader"""

    def __init__(
        self,
        current_version: str = "1.0.0",
        update_url: str = "https://api.github.com/repos/yourusername/trading-bot-simulator/releases/latest",
        app_name: str = "Trading Bot Simulator"
    ):
        self.current_version = current_version
        self.update_url = update_url
        self.app_name = app_name
        self.logger = logging.getLogger(__name__)

    def check_for_updates(self) -> Tuple[bool, Optional[Dict]]:
        """
        Check if updates are available

        Returns:
            Tuple of (update_available, update_info)
        """
        try:
            response = requests.get(self.update_url, timeout=10)
            response.raise_for_status()

            release_data = response.json()
            latest_version = release_data.get('tag_name', '').lstrip('v')

            if not latest_version:
                return False, None

            # Compare versions
            if version.parse(latest_version) > version.parse(self.current_version):
                update_info = {
                    'version': latest_version,
                    'name': release_data.get('name', ''),
                    'description': release_data.get('body', ''),
                    'published_at': release_data.get('published_at', ''),
                    'download_url': self._get_windows_download_url(release_data),
                }
                return True, update_info

            return False, None

        except Exception as e:
            self.logger.error(f"Error checking for updates: {e}")
            return False, None

    def _get_windows_download_url(self, release_data: Dict) -> Optional[str]:
        """Extract Windows installer download URL from release data"""
        assets = release_data.get('assets', [])

        for asset in assets:
            name = asset.get('name', '').lower()
            if name.endswith('.exe') or name.endswith('-setup.exe'):
                return asset.get('browser_download_url')

        return None

    def download_update(self, download_url: str, save_path: Optional[Path] = None) -> Optional[Path]:
        """
        Download update installer

        Args:
            download_url: URL to download from
            save_path: Where to save the installer

        Returns:
            Path to downloaded installer or None
        """
        if not save_path:
            temp_dir = Path(os.getenv('TEMP', '/tmp'))
            save_path = temp_dir / 'TradingBotSimulator-Update.exe'

        try:
            self.logger.info(f"Downloading update from {download_url}")

            response = requests.get(download_url, stream=True, timeout=30)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0

            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

                        # Log progress
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            if downloaded % (1024 * 1024) == 0:  # Log every MB
                                self.logger.info(f"Download progress: {progress:.1f}%")

            self.logger.info(f"Update downloaded to {save_path}")
            return save_path

        except Exception as e:
            self.logger.error(f"Error downloading update: {e}")
            if save_path.exists():
                save_path.unlink()
            return None

    def install_update(self, installer_path: Path) -> bool:
        """
        Launch installer and exit application

        Args:
            installer_path: Path to installer executable

        Returns:
            True if installer launched successfully
        """
        try:
            import subprocess

            # Launch installer
            if sys.platform == 'win32':
                # Use start to launch installer and exit
                subprocess.Popen(['start', '/wait', str(installer_path)], shell=True)
            else:
                subprocess.Popen([str(installer_path)])

            self.logger.info("Installer launched, application will exit")
            return True

        except Exception as e:
            self.logger.error(f"Error launching installer: {e}")
            return False
