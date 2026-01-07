# Building Transformer Design Optimizer

## Time-Limited Executable

This build system creates executables that:
- Expire after a specified number of days (default: 7)
- Self-delete when opened after expiration
- Show an "expired" message to users

## Files Created

- `launcher.py` - Entry point with expiration logic
- `TransformerOptimizer.spec` - PyInstaller configuration
- `build_app.py` - Automated build script

---

## macOS Build (Already Complete)

The macOS build has been created:

**Location:** `dist/TransformerOptimizer.app`
**Zip file:** `dist/TransformerOptimizer_macOS.zip` (524 MB)
**Expires:** January 14, 2026

### To Distribute
Send the zip file to users. They can:
1. Unzip the file
2. Double-click `TransformerOptimizer.app` to run
3. If macOS blocks it: Right-click > Open > Open

---

## Windows Build Instructions

Since you're on macOS, you need a Windows machine to build the `.exe`. Here are your options:

### Option 1: Build on a Windows Machine

1. Copy these files to a Windows computer:
   - `launcher.py`
   - `transformer_ui.py`
   - `mainRect.py`
   - `TransformerOptimizer.spec`
   - `build_app.py`
   - `.venv/` (or recreate virtualenv)

2. Install Python 3.8+ and dependencies:
   ```cmd
   pip install dearpygui numpy scipy torch numba pyinstaller
   ```

3. Run the build:
   ```cmd
   python build_app.py --days 7
   ```

4. The `.exe` will be in `dist/TransformerOptimizer.exe`

### Option 2: Use GitHub Actions (Automated)

Create `.github/workflows/build.yml`:

```yaml
name: Build Windows Executable

on:
  workflow_dispatch:
    inputs:
      expiration_days:
        description: 'Days until expiration'
        default: '7'

jobs:
  build-windows:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install dearpygui numpy scipy torch numba pyinstaller

      - name: Build executable
        run: |
          python build_app.py --days ${{ github.event.inputs.expiration_days }}

      - name: Upload artifact
        uses: actions/upload-artifact@v4
        with:
          name: TransformerOptimizer-Windows
          path: dist/TransformerOptimizer.exe
```

### Option 3: Use a Windows VM or Remote Desktop

Use a Windows virtual machine (Parallels, VMware, VirtualBox) or cloud instance (Azure, AWS) to build.

---

## Changing the Expiration Date

### Method 1: Use build script
```bash
python build_app.py --days 14  # Expires in 14 days
```

### Method 2: Edit manually
Edit `launcher.py` line ~17:
```python
EXPIRATION_DATE = datetime.date(2026, 1, 21)  # Set your date
```
Then rebuild.

---

## Security Notes

- The expiration date is embedded in the executable
- Determined users could potentially bypass this by:
  - Setting their system clock back
  - Reverse engineering the binary
- For stronger protection, consider:
  - Online license validation
  - Hardware-locked licenses
  - Code obfuscation tools (pyarmor, etc.)

---

## Troubleshooting

### "App is damaged" on macOS
Right-click the app > Open > Open, or run:
```bash
xattr -cr dist/TransformerOptimizer.app
```

### Large file size
The executable includes PyTorch, NumPy, SciPy, etc. To reduce size:
- Use `--exclude-module` flags for unused packages
- Use PyTorch CPU-only builds

### Missing DLLs on Windows
Install Visual C++ Redistributable 2015-2022.
