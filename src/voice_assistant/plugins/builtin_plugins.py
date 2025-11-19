"""
Built-in Plugins

Collection of built-in plugins providing common functionality:
- Time/Date
- Weather
- Timers
- Reminders
- News
- Trading (integrates with trading bot)
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any
import time
import threading

from .base_plugin import BasePlugin

logger = logging.getLogger(__name__)


class TimePlugin(BasePlugin):
    """Handles time and date queries"""

    def get_intents(self) -> List[str]:
        return ["time"]

    def handle(self, intent_name: str, entities: Dict[str, Any], context: Dict[str, Any]) -> str:
        now = datetime.now()

        if intent_name == "time":
            time_str = now.strftime("%I:%M %p")
            return f"The current time is {time_str}"

        return "I'm not sure what you're asking about."


class WeatherPlugin(BasePlugin):
    """Handles weather queries (requires API key)"""

    def get_intents(self) -> List[str]:
        return ["weather"]

    def handle(self, intent_name: str, entities: Dict[str, Any], context: Dict[str, Any]) -> str:
        location = entities.get("location", "your location")
        date = entities.get("date", "today")

        # In a real implementation, you'd call a weather API here
        # For now, return a placeholder response
        return f"I would check the weather for {location} for {date}, but I need a weather API key configured."

    def get_help(self) -> str:
        return "Weather Plugin - Ask about weather. Requires API key in config."


class TimerPlugin(BasePlugin):
    """Handles timer creation and management"""

    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.timers: Dict[str, threading.Timer] = {}
        self.timer_count = 0

    def get_intents(self) -> List[str]:
        return ["timer"]

    def handle(self, intent_name: str, entities: Dict[str, Any], context: Dict[str, Any]) -> str:
        duration_text = entities.get("duration", "")

        if not duration_text:
            return "How long should I set the timer for?"

        # Parse duration
        duration_seconds = self._parse_duration(duration_text)

        if duration_seconds is None:
            return f"I couldn't understand the duration '{duration_text}'"

        # Create timer
        timer_id = f"timer_{self.timer_count}"
        self.timer_count += 1

        def timer_callback():
            logger.info(f"Timer {timer_id} completed")
            # In a real implementation, this would trigger a notification
            tts = context.get("tts")
            if tts:
                tts.speak("Timer complete!")

        timer = threading.Timer(duration_seconds, timer_callback)
        timer.start()
        self.timers[timer_id] = timer

        minutes = duration_seconds // 60
        seconds = duration_seconds % 60

        if minutes > 0:
            time_str = f"{minutes} minute{'s' if minutes != 1 else ''}"
            if seconds > 0:
                time_str += f" and {seconds} second{'s' if seconds != 1 else ''}"
        else:
            time_str = f"{seconds} second{'s' if seconds != 1 else ''}"

        return f"Timer set for {time_str}"

    def _parse_duration(self, duration_text: str) -> int:
        """Parse duration text to seconds"""
        import re

        # Extract number and unit
        match = re.search(r"(\d+)\s*(second|minute|hour|sec|min|hr)s?", duration_text.lower())

        if not match:
            return None

        amount = int(match.group(1))
        unit = match.group(2)

        if unit.startswith("sec"):
            return amount
        elif unit.startswith("min"):
            return amount * 60
        elif unit.startswith("hour") or unit.startswith("hr"):
            return amount * 3600

        return None

    def shutdown(self):
        """Cancel all timers on shutdown"""
        for timer in self.timers.values():
            timer.cancel()
        self.timers.clear()
        super().shutdown()


class ReminderPlugin(BasePlugin):
    """Handles reminder creation"""

    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.reminders: List[Dict] = []

    def get_intents(self) -> List[str]:
        return ["reminder"]

    def handle(self, intent_name: str, entities: Dict[str, Any], context: Dict[str, Any]) -> str:
        task = entities.get("task", "")
        time_text = entities.get("time", "")

        if not task:
            return "What would you like me to remind you about?"

        if not time_text:
            return f"When should I remind you to {task}?"

        # Store reminder (in real implementation, this would schedule a notification)
        reminder = {
            "task": task,
            "time": time_text,
            "created_at": datetime.now()
        }
        self.reminders.append(reminder)

        return f"I'll remind you to {task} {time_text}"


class NewsPlugin(BasePlugin):
    """Handles news queries (requires news API)"""

    def get_intents(self) -> List[str]:
        return ["news"]

    def handle(self, intent_name: str, entities: Dict[str, Any], context: Dict[str, Any]) -> str:
        category = entities.get("category", "general")

        # In a real implementation, you'd call a news API here
        return f"I would fetch {category} news, but I need a news API key configured."

    def get_help(self) -> str:
        return "News Plugin - Get latest news. Requires API key in config."


class TradingPlugin(BasePlugin):
    """Handles trading queries (integrates with trading bot)"""

    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.trading_engine = None

    def get_intents(self) -> List[str]:
        return ["trading"]

    def handle(self, intent_name: str, entities: Dict[str, Any], context: Dict[str, Any]) -> str:
        symbol = entities.get("symbol", "")
        action = entities.get("action", "")
        quantity = entities.get("quantity", "")

        if not symbol:
            return "Which stock are you interested in?"

        # In a real implementation, integrate with the trading engine
        if action == "buy":
            return f"I would buy {quantity or '1'} share{'s' if quantity != '1' else ''} of {symbol}"
        elif action == "sell":
            return f"I would sell {quantity or '1'} share{'s' if quantity != '1' else ''} of {symbol}"
        else:
            # Get stock price
            return f"The current price of {symbol} would be shown here with trading engine integration"

    def get_help(self) -> str:
        return "Trading Plugin - Query stock prices and execute trades (requires trading engine)"


class GreetingPlugin(BasePlugin):
    """Handles greetings and goodbyes"""

    def get_intents(self) -> List[str]:
        return ["greeting", "goodbye"]

    def handle(self, intent_name: str, entities: Dict[str, Any], context: Dict[str, Any]) -> str:
        if intent_name == "greeting":
            greetings = [
                "Hello! How can I help you?",
                "Hi there! What can I do for you?",
                "Greetings! How may I assist you?",
            ]
            import random
            return random.choice(greetings)

        elif intent_name == "goodbye":
            farewells = [
                "Goodbye! Have a great day!",
                "See you later!",
                "Farewell! Let me know if you need anything.",
            ]
            import random
            return random.choice(farewells)

        return "Hello!"


class HelpPlugin(BasePlugin):
    """Provides help and lists available commands"""

    def get_intents(self) -> List[str]:
        return ["help"]

    def handle(self, intent_name: str, entities: Dict[str, Any], context: Dict[str, Any]) -> str:
        plugin_manager = context.get("plugin_manager")

        if not plugin_manager:
            return "I can help you with various tasks. Try asking me about the time, weather, or to set a timer."

        plugins = plugin_manager.list_plugins()
        intents = plugin_manager.get_all_intents()

        help_text = f"I have {len(plugins)} plugins loaded that can handle the following intents: "
        help_text += ", ".join(sorted(intents))

        return help_text


class SearchPlugin(BasePlugin):
    """Handles web search queries"""

    def get_intents(self) -> List[str]:
        return ["search"]

    def handle(self, intent_name: str, entities: Dict[str, Any], context: Dict[str, Any]) -> str:
        query = entities.get("query", "")

        if not query:
            return "What would you like me to search for?"

        # In a real implementation, you'd perform a web search here
        return f"I would search for '{query}', but I need a search API configured."

    def get_help(self) -> str:
        return "Search Plugin - Search the web. Requires search API."
