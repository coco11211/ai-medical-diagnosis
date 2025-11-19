"""Plugin system for voice assistant"""

from .base_plugin import BasePlugin, PluginManager
from .builtin_plugins import (
    TimePlugin,
    WeatherPlugin,
    TimerPlugin,
    ReminderPlugin,
    NewsPlugin,
    TradingPlugin
)

__all__ = [
    "BasePlugin",
    "PluginManager",
    "TimePlugin",
    "WeatherPlugin",
    "TimerPlugin",
    "ReminderPlugin",
    "NewsPlugin",
    "TradingPlugin"
]
