#!/usr/bin/env python3
"""
AI Music Composer - Main Application Entry Point
Run this file to start the GUI application
"""

import sys
import os

# Add music_composer to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'music_composer'))

from music_composer.gui.main_window import main

if __name__ == "__main__":
    print("Starting AI Music Composer...")
    print("Loading models and initializing GUI...")
    main()
