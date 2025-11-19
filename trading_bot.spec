# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Trading Bot Simulator Windows Application
This file is used to build a standalone Windows executable
"""

import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Collect all necessary data files
datas = [
    ('config.yaml', '.'),
    ('src', 'src'),
]

# Collect hidden imports
hiddenimports = [
    'PyQt6',
    'PyQt6.QtCore',
    'PyQt6.QtGui',
    'PyQt6.QtWidgets',
    'PyQt6.QtWebEngineWidgets',
    'sklearn.utils._weight_vector',
    'tensorflow',
    'keras',
    'yfinance',
    'pandas',
    'numpy',
    'plotly',
    'dash',
    'backtrader',
    'ta',
    'pandas_ta',
]

# Collect all submodules from src
hiddenimports += collect_submodules('src')
hiddenimports += collect_submodules('sklearn')
hiddenimports += collect_submodules('tensorflow')

# Analysis
a = Analysis(
    ['windows_app.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib'],  # Exclude if not directly needed
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Remove duplicate files
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='TradingBotSimulator',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # No console window for GUI app
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='resources/icon.ico',  # App icon (to be created)
    version='version_info.txt',  # Version info (to be created)
)

# Collect all dependencies
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='TradingBotSimulator',
)
