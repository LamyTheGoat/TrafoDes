# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Transformer Design Optimizer
Builds time-limited executable for macOS and Windows
"""

import sys
import platform
from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

block_cipher = None

# Collect PyTorch with CUDA support (includes all DLLs and binaries)
try:
    torch_datas, torch_binaries, torch_hiddenimports = collect_all('torch')
    print(f"Collected {len(torch_binaries)} PyTorch binaries (including CUDA if available)")
except Exception as e:
    print(f"Warning: Could not collect PyTorch: {e}")
    torch_datas, torch_binaries, torch_hiddenimports = [], [], []

# Determine platform
is_windows = platform.system() == 'Windows'
is_macos = platform.system() == 'Darwin'

# Collect pandas and its dependencies
pandas_datas, pandas_binaries, pandas_hiddenimports = collect_all('pandas')

# Collect openpyxl (Excel export)
try:
    openpyxl_datas, openpyxl_binaries, openpyxl_hiddenimports = collect_all('openpyxl')
except Exception:
    openpyxl_datas, openpyxl_binaries, openpyxl_hiddenimports = [], [], []

# Collect reportlab (PDF export)
try:
    reportlab_datas, reportlab_binaries, reportlab_hiddenimports = collect_all('reportlab')
except Exception:
    reportlab_datas, reportlab_binaries, reportlab_hiddenimports = [], [], []

# Data files to include
datas = [
    ('quack.wav', '.'),  # Include quack.wav in the bundle root
]
datas += pandas_datas
datas += openpyxl_datas
datas += reportlab_datas
datas += torch_datas

# Hidden imports that PyInstaller might miss
hidden_imports = [
    # Application modules (imported in try/except blocks, not traced automatically)
    'mainRect',
    'tank_oil',
    'batch_optimizer',
    'export_utils',
    # GUI
    'dearpygui',
    'dearpygui.dearpygui',
    # Scientific computing
    'numpy',
    'scipy',
    'scipy.optimize',
    'scipy.special',
    'torch',
    'torch.cuda',
    'torch.cuda.amp',
    'torch.backends',
    'torch.backends.cuda',
    'torch.backends.cudnn',
    'torch.mps',
    'torch._C',
    'torch._C._cuda',
    # Sound
    'playsound3',
    # Performance
    'numba',
    'numba.core',
    'concurrent.futures',
    'multiprocessing',
    # Data export
    'pandas',
    'openpyxl',
    'reportlab',
    'reportlab.lib',
    'reportlab.lib.colors',
    'reportlab.lib.pagesizes',
    'reportlab.platypus',
    'reportlab.lib.styles',
    'reportlab.lib.units',
]

# Try to add MLX for Apple Silicon
try:
    import mlx
    hidden_imports.append('mlx')
    hidden_imports.append('mlx.core')
except ImportError:
    pass

# Combine hidden imports
hidden_imports += pandas_hiddenimports
hidden_imports += openpyxl_hiddenimports
hidden_imports += reportlab_hiddenimports
hidden_imports += torch_hiddenimports

# Combine binaries (torch_binaries includes CUDA DLLs when built with CUDA)
all_binaries = torch_binaries + pandas_binaries + openpyxl_binaries + reportlab_binaries

a = Analysis(
    ['launcher.py'],
    pathex=[],
    binaries=all_binaries,
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
        upx=False,  # Disabled - UPX can cause issues with some libraries
        upx_exclude=[],
        runtime_tmpdir=None,
        console=True,  # No console window
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon='program.ico',  # Add .ico path here if you have one
    )
