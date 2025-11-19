#!/usr/bin/env python3
"""
Neural Style Transfer Suite - Main Entry Point
Comprehensive neural style transfer with real-time video processing, GPU acceleration, and advanced features
"""

import sys
import os

# Add neural_style_transfer to path
sys.path.insert(0, os.path.dirname(__file__))

from neural_style_transfer.cli import main

if __name__ == '__main__':
    main()
