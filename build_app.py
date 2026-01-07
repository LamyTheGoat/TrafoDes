#!/usr/bin/env python3
"""
Build script for Transformer Design Optimizer
Creates time-limited executable for distribution
"""

import os
import sys
import subprocess
import datetime
import shutil
import argparse

def update_expiration_date(days_from_now=7):
    """Update the expiration date in launcher.py"""
    expiration_date = datetime.date.today() + datetime.timedelta(days=days_from_now)

    launcher_path = os.path.join(os.path.dirname(__file__), 'launcher.py')

    with open(launcher_path, 'r') as f:
        content = f.read()

    # Find and replace the expiration date line
    import re
    pattern = r'EXPIRATION_DATE = datetime\.date\(\d+, \d+, \d+\)'
    replacement = f'EXPIRATION_DATE = datetime.date({expiration_date.year}, {expiration_date.month}, {expiration_date.day})'

    new_content = re.sub(pattern, replacement, content)

    with open(launcher_path, 'w') as f:
        f.write(new_content)

    print(f"Updated expiration date to: {expiration_date}")
    return expiration_date

def check_pyinstaller():
    """Check if PyInstaller is installed"""
    try:
        import PyInstaller
        return True
    except ImportError:
        return False

def install_pyinstaller():
    """Install PyInstaller"""
    print("Installing PyInstaller...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pyinstaller'])

def build_executable():
    """Build the executable using PyInstaller"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    spec_file = os.path.join(script_dir, 'TransformerOptimizer.spec')

    if not os.path.exists(spec_file):
        print(f"Error: Spec file not found: {spec_file}")
        return False

    print("\nBuilding executable...")
    print("This may take several minutes...\n")

    try:
        result = subprocess.run(
            [sys.executable, '-m', 'PyInstaller', spec_file, '--clean'],
            cwd=script_dir,
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Build failed with error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Build Transformer Optimizer executable')
    parser.add_argument('--days', type=int, default=7,
                       help='Number of days until expiration (default: 7)')
    parser.add_argument('--no-update-date', action='store_true',
                       help='Skip updating the expiration date')

    args = parser.parse_args()

    print("="*60)
    print("  Transformer Design Optimizer - Build Script")
    print("="*60)

    # Check/install PyInstaller
    if not check_pyinstaller():
        print("\nPyInstaller not found.")
        install_pyinstaller()

    # Update expiration date
    if not args.no_update_date:
        expiration = update_expiration_date(args.days)
        print(f"\nThe executable will expire on: {expiration}")
        print(f"({args.days} days from today)")
    else:
        print("\nKeeping existing expiration date")

    # Build
    print("\n" + "-"*60)
    if build_executable():
        print("\n" + "="*60)
        print("  BUILD SUCCESSFUL!")
        print("="*60)

        import platform
        if platform.system() == 'Darwin':
            print("\nmacOS build location:")
            print("  ./dist/TransformerOptimizer.app")
            print("\nTo distribute, zip the .app bundle:")
            print("  cd dist && zip -r TransformerOptimizer.zip TransformerOptimizer.app")
        else:
            print("\nWindows build location:")
            print("  ./dist/TransformerOptimizer.exe")

        print("\n" + "-"*60)
        print("IMPORTANT: This build will expire and self-delete after the")
        print(f"expiration date. Users won't be able to use it after that.")
        print("-"*60)
    else:
        print("\n" + "="*60)
        print("  BUILD FAILED")
        print("="*60)
        print("\nCheck the error messages above for details.")
        sys.exit(1)

if __name__ == '__main__':
    main()
