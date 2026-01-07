# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Transformer Design Optimizer
Builds time-limited executable for macOS and Windows
"""

import sys
import platform

block_cipher = None

# Determine platform
is_windows = platform.system() == 'Windows'
is_macos = platform.system() == 'Darwin'

# Data files to include
datas = []  # Add any data files here if needed

# Hidden imports that PyInstaller might miss
hidden_imports = [
    'dearpygui',
    'dearpygui.dearpygui',
    'numpy',
    'scipy',
    'scipy.optimize',
    'scipy.special',
    'torch',
    'torch.mps',
    'numba',
    'numba.core',
    'concurrent.futures',
    'multiprocessing',
]

# Try to add MLX for Apple Silicon
try:
    import mlx
    hidden_imports.append('mlx')
    hidden_imports.append('mlx.core')
except ImportError:
    pass

a = Analysis(
    ['launcher.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'matplotlib',
        'PIL',
        'IPython',
        'jupyter',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

if is_macos:
    # macOS: Create .app bundle
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name='TransformerOptimizer',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,  # UPX can cause issues on macOS
        console=False,  # No console window
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
    )

    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip=False,
        upx=False,
        upx_exclude=[],
        name='TransformerOptimizer',
    )

    app = BUNDLE(
        coll,
        name='TransformerOptimizer.app',
        icon=None,  # Add icon path here if you have one
        bundle_identifier='com.trafodes.transformeroptimizer',
        info_plist={
            'CFBundleName': 'Transformer Optimizer',
            'CFBundleDisplayName': 'Transformer Design Optimizer',
            'CFBundleVersion': '1.0.0',
            'CFBundleShortVersionString': '1.0.0',
            'NSHighResolutionCapable': True,
        },
    )

else:
    # Windows: Create single .exe file
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.zipfiles,
        a.datas,
        [],
        name='TransformerOptimizer',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        upx_exclude=[],
        runtime_tmpdir=None,
        console=False,  # No console window
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon=None,  # Add .ico path here if you have one
    )
