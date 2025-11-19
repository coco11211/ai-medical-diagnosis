"""
Base Plugin System

Provides a framework for creating extensible voice assistant plugins.
Each plugin can handle specific intents and provide custom functionality.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import importlib
import os
import sys

logger = logging.getLogger(__name__)


class BasePlugin(ABC):
    """Base class for all voice assistant plugins"""

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize plugin

        Args:
            config: Plugin configuration dictionary
        """
        self.config = config or {}
        self.name = self.__class__.__name__
        self.enabled = True

    @abstractmethod
    def get_intents(self) -> List[str]:
        """
        Get list of intents this plugin handles

        Returns:
            List of intent names
        """
        pass

    @abstractmethod
    def handle(self, intent_name: str, entities: Dict[str, Any], context: Dict[str, Any]) -> str:
        """
        Handle an intent

        Args:
            intent_name: Name of the intent
            entities: Extracted entities from NLU
            context: Context dictionary with assistant state

        Returns:
            Response text to be spoken
        """
        pass

    def can_handle(self, intent_name: str) -> bool:
        """
        Check if this plugin can handle the given intent

        Args:
            intent_name: Intent name to check

        Returns:
            True if plugin can handle this intent
        """
        return intent_name in self.get_intents()

    def initialize(self):
        """Initialize plugin (called when plugin is loaded)"""
        logger.info(f"Plugin {self.name} initialized")

    def shutdown(self):
        """Shutdown plugin (called when assistant is shutting down)"""
        logger.info(f"Plugin {self.name} shutdown")

    def get_help(self) -> str:
        """Get help text for this plugin"""
        return f"Plugin: {self.name}"


class PluginManager:
    """Manages voice assistant plugins"""

    def __init__(self):
        self.plugins: List[BasePlugin] = []
        self.intent_map: Dict[str, BasePlugin] = {}

    def register(self, plugin: BasePlugin):
        """
        Register a plugin

        Args:
            plugin: Plugin instance to register
        """
        if not isinstance(plugin, BasePlugin):
            raise TypeError(f"Plugin must inherit from BasePlugin, got {type(plugin)}")

        self.plugins.append(plugin)

        # Map intents to plugin
        for intent in plugin.get_intents():
            if intent in self.intent_map:
                logger.warning(
                    f"Intent '{intent}' already handled by {self.intent_map[intent].name}, "
                    f"overriding with {plugin.name}"
                )
            self.intent_map[intent] = plugin

        plugin.initialize()
        logger.info(f"Registered plugin: {plugin.name} (handles: {', '.join(plugin.get_intents())})")

    def unregister(self, plugin: BasePlugin):
        """
        Unregister a plugin

        Args:
            plugin: Plugin instance to unregister
        """
        if plugin in self.plugins:
            plugin.shutdown()
            self.plugins.remove(plugin)

            # Remove from intent map
            for intent, p in list(self.intent_map.items()):
                if p == plugin:
                    del self.intent_map[intent]

            logger.info(f"Unregistered plugin: {plugin.name}")

    def handle_intent(self, intent_name: str, entities: Dict[str, Any], context: Dict[str, Any]) -> Optional[str]:
        """
        Handle an intent using the appropriate plugin

        Args:
            intent_name: Intent name
            entities: Extracted entities
            context: Context dictionary

        Returns:
            Response text or None if no plugin can handle
        """
        plugin = self.intent_map.get(intent_name)

        if plugin and plugin.enabled:
            try:
                response = plugin.handle(intent_name, entities, context)
                return response
            except Exception as e:
                logger.error(f"Error in plugin {plugin.name} handling {intent_name}: {e}")
                return f"Sorry, I encountered an error while processing that request."

        logger.warning(f"No plugin registered for intent: {intent_name}")
        return None

    def get_plugin(self, name: str) -> Optional[BasePlugin]:
        """Get plugin by name"""
        for plugin in self.plugins:
            if plugin.name == name:
                return plugin
        return None

    def list_plugins(self) -> List[str]:
        """Get list of registered plugin names"""
        return [p.name for p in self.plugins]

    def get_all_intents(self) -> List[str]:
        """Get all intents handled by registered plugins"""
        return list(self.intent_map.keys())

    def load_from_directory(self, directory: str):
        """
        Load plugins from a directory

        Args:
            directory: Directory containing plugin files
        """
        if not os.path.isdir(directory):
            logger.warning(f"Plugin directory not found: {directory}")
            return

        # Add directory to Python path
        if directory not in sys.path:
            sys.path.insert(0, directory)

        # Find all Python files
        for filename in os.listdir(directory):
            if filename.endswith('.py') and not filename.startswith('_'):
                module_name = filename[:-3]

                try:
                    # Import module
                    module = importlib.import_module(module_name)

                    # Find plugin classes
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)

                        # Check if it's a plugin class
                        if (isinstance(attr, type) and
                            issubclass(attr, BasePlugin) and
                            attr is not BasePlugin):

                            # Instantiate and register
                            plugin = attr()
                            self.register(plugin)

                except Exception as e:
                    logger.error(f"Error loading plugin from {filename}: {e}")

    def shutdown_all(self):
        """Shutdown all plugins"""
        for plugin in self.plugins:
            try:
                plugin.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down plugin {plugin.name}: {e}")

        self.plugins.clear()
        self.intent_map.clear()
        logger.info("All plugins shut down")
