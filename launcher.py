"""
Time-Limited Launcher for Transformer Design Optimizer
Expires after 1 week and self-deletes
"""

import os
import sys
import time
import datetime
import platform
import subprocess
import shutil

# ============================================
# EXPIRATION CONFIGURATION
# Set the expiration date (YYYY, MM, DD)
# ============================================
EXPIRATION_DATE = datetime.date(2026, 1, 22)  # 1 week from Jan 15, 2026

def get_executable_path():
    """Get the path to the current executable."""
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        return sys.executable
    else:
        # Running as script
        return os.path.abspath(__file__)

def self_delete_windows(exe_path):
    """Self-delete on Windows using a batch script."""
    batch_script = f'''@echo off
:loop
del "{exe_path}" 2>nul
if exist "{exe_path}" (
    timeout /t 1 /nobreak >nul
    goto loop
)
del "%~f0"
'''
    batch_path = os.path.join(os.environ.get('TEMP', '.'), 'cleanup.bat')
    with open(batch_path, 'w') as f:
        f.write(batch_script)

    # Run batch script detached
    subprocess.Popen(
        ['cmd', '/c', batch_path],
        creationflags=subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS,
        close_fds=True
    )

def self_delete_unix(exe_path):
    """Self-delete on macOS/Linux."""
    # For .app bundles, delete the entire bundle
    if '.app' in exe_path:
        # Find the .app bundle root
        app_path = exe_path
        while app_path and not app_path.endswith('.app'):
            app_path = os.path.dirname(app_path)
        if app_path and os.path.exists(app_path):
            try:
                shutil.rmtree(app_path)
                return
            except:
                pass

    # For standalone executables
    try:
        os.remove(exe_path)
    except:
        pass

def show_expired_message():
    """Show expiration message to user."""
    try:
        import dearpygui.dearpygui as dpg

        dpg.create_context()
        dpg.create_viewport(title="License Expired", width=400, height=200)

        with dpg.window(label="Expired", tag="expired_window", no_title_bar=True,
                       no_move=True, no_resize=True, no_collapse=True):
            dpg.add_spacer(height=30)
            dpg.add_text("This software license has expired.", color=(255, 100, 100))
            dpg.add_spacer(height=10)
            dpg.add_text("Please contact the developer for a new version.")
            dpg.add_spacer(height=30)
            dpg.add_button(label="OK", callback=lambda: dpg.stop_dearpygui(), width=100)

        dpg.set_primary_window("expired_window", True)
        dpg.setup_dearpygui()
        dpg.show_viewport()

        while dpg.is_dearpygui_running():
            dpg.render_dearpygui_frame()

        dpg.destroy_context()
    except:
        # Fallback to console message
        print("\n" + "="*50)
        print("  SOFTWARE LICENSE EXPIRED")
        print("="*50)
        print("\nThis software license has expired.")
        print("Please contact the developer for a new version.")
        print("\n" + "="*50)
        input("\nPress Enter to exit...")

def check_expiration():
    """Check if the software has expired."""
    today = datetime.date.today()
    return today > EXPIRATION_DATE

def main():
    """Main entry point with expiration check."""
    exe_path = get_executable_path()

    # Check expiration
    if check_expiration():
        print(f"Software expired on {EXPIRATION_DATE}")
        show_expired_message()

        # Attempt self-deletion
        system = platform.system()
        if system == 'Windows':
            self_delete_windows(exe_path)
        else:  # macOS, Linux
            self_delete_unix(exe_path)

        sys.exit(1)

    # Not expired - run the main application
    # Add the script directory to path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    # For PyInstaller bundles, also add the _MEIPASS directory
    # This is where data files (including .py modules) are extracted
    if getattr(sys, 'frozen', False):
        meipass = getattr(sys, '_MEIPASS', None)
        if meipass and meipass not in sys.path:
            sys.path.insert(0, meipass)

    # Import and run the main application
    try:
        from transformer_ui import TransformerOptimizerApp
        app = TransformerOptimizerApp()
        app.run()
    except Exception as e:
        print(f"Error starting application: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")
        sys.exit(1)

if __name__ == "__main__":
    main()
