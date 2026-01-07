# Build Instructions - Transformer Design Optimizer

Complete guide for setting up the development environment and building distributable executables.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Project Structure](#project-structure)
3. [Development Setup](#development-setup)
4. [Dependencies](#dependencies)
5. [Running from Source](#running-from-source)
6. [Building Executables](#building-executables)
7. [Distribution](#distribution)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software

| Software | Version | Notes |
|----------|---------|-------|
| Python | 3.8 - 3.11 | Python 3.8.10 recommended |
| Git | Latest | For version control |
| pip | Latest | Python package manager |

### Platform-Specific Requirements

**macOS:**
- Xcode Command Line Tools: `xcode-select --install`
- For Apple Silicon (M1/M2/M3): MLX support is automatically enabled

**Windows:**
- Visual C++ Redistributable 2015-2022
- Windows 10 or later recommended

---

## Project Structure

```
TrafoDes/
├── launcher.py              # Entry point with expiration logic
├── transformer_ui.py        # Main GUI application (Dear PyGui)
├── mainRect.py              # Core optimization engine (rectangular cores)
├── mainRound.py             # Alternative round core design
├── turboOpt.py              # Optimized version of optimizer
├── main.py                  # Legacy implementation
├── app.py                   # Alternative Streamlit web interface
├── build_app.py             # Automated build script
├── TransformerOptimizer.spec # PyInstaller configuration
├── program.ico              # Application icon (Windows)
├── quack.wav                # Sound effect
├── .github/workflows/
│   └── build-windows.yml    # GitHub Actions CI/CD
└── .venv/                   # Virtual environment (not in repo)
```

---

## Development Setup

### 1. Clone the Repository

```bash
git clone https://github.com/LamyTheGoat/TrafoDes.git
cd TrafoDes
```

### 2. Create Virtual Environment

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows:**
```cmd
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If `requirements.txt` doesn't exist, install manually:

```bash
# Core dependencies
pip install dearpygui>=1.11
pip install numpy>=1.24
pip install scipy>=1.10
pip install numba>=0.57

# GPU acceleration (choose based on your platform)
# For NVIDIA GPUs:
pip install torch

# For Apple Silicon (M1/M2/M3) - optional, enhances performance:
pip install mlx

# Audio
pip install playsound3

# CAD export
pip install ezdxf

# Build tools
pip install pyinstaller
```

---

## Dependencies

### Core Dependencies

| Package | Purpose | Required |
|---------|---------|----------|
| `dearpygui` | GUI framework | Yes |
| `numpy` | Numerical computations | Yes |
| `scipy` | Optimization algorithms | Yes |
| `numba` | JIT compilation & parallelization | Yes |

### GPU Acceleration (Optional)

| Package | Purpose | Platform |
|---------|---------|----------|
| `torch` | PyTorch with MPS/CUDA support | All |
| `mlx` | Apple Silicon GPU optimization | macOS (Apple Silicon) |

### Other Dependencies

| Package | Purpose | Required |
|---------|---------|----------|
| `playsound3` | Audio playback | Yes |
| `ezdxf` | DXF file export | Optional |
| `pyinstaller` | Building executables | For builds only |

### Alternative Web Interface (Optional)

| Package | Purpose |
|---------|---------|
| `streamlit` | Web UI framework |
| `plotly` | Interactive visualization |
| `pandas` | Data manipulation |

---

## Running from Source

### Main GUI Application

```bash
# Activate virtual environment first
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate     # Windows

# Run the application
python launcher.py
```

### Alternative: Direct UI Launch (bypasses expiration)

```bash
python transformer_ui.py
```

### Alternative: Streamlit Web Interface

```bash
streamlit run app.py
```

---

## Building Executables

### Quick Build

The easiest way to build is using the automated build script:

```bash
python build_app.py --days 7
```

Options:
- `--days N`: Set expiration to N days from build date (default: 7)
- `--no-update-date`: Keep existing expiration date

### macOS Build

1. **Activate virtual environment:**
   ```bash
   source .venv/bin/activate
   ```

2. **Run build script:**
   ```bash
   python build_app.py --days 7
   ```

3. **Output location:**
   ```
   dist/TransformerOptimizer.app
   ```

4. **Create distributable zip:**
   ```bash
   cd dist
   zip -r TransformerOptimizer_macOS.zip TransformerOptimizer.app
   ```

### Windows Build

#### Option 1: Build on Windows Machine

1. Copy the project to a Windows computer
2. Create virtual environment and install dependencies
3. Run:
   ```cmd
   python build_app.py --days 7
   ```
4. Output: `dist/TransformerOptimizer.exe`

#### Option 2: GitHub Actions (Automated)

The repository includes a GitHub Actions workflow for building Windows executables automatically.

**Step-by-step instructions:**

1. **Push your code to GitHub** (if not already done):
   ```bash
   git add .
   git commit -m "Prepare for Windows build"
   git push origin main
   ```

2. **Go to your GitHub repository**:
   ```
   https://github.com/LamyTheGoat/TrafoDes
   ```

3. **Navigate to Actions tab**:
   - Click the **"Actions"** tab at the top of the repository

4. **Select the workflow**:
   - In the left sidebar, click **"Build Windows Executable"**

5. **Run the workflow**:
   - Click the **"Run workflow"** dropdown button (right side)
   - Enter the number of days until expiration (default: 7)
   - Click the green **"Run workflow"** button

6. **Wait for the build** (typically 3-5 minutes):
   - The workflow will show a yellow spinning indicator while running
   - Green checkmark = success, Red X = failure

7. **Download the executable**:
   - Click on the completed workflow run
   - Scroll down to **"Artifacts"** section
   - Click **"TransformerOptimizer-Windows"** to download
   - Extract the zip file to get `TransformerOptimizer.exe`

**Workflow file location**: `.github/workflows/build-windows.yml`

**What the workflow does**:
- Uses Windows Server (windows-latest)
- Installs Python 3.10
- Installs all dependencies (dearpygui, numpy, scipy, numba, torch CPU-only)
- Updates expiration date in launcher.py
- Builds single-file executable with PyInstaller
- Uploads artifact (available for 30 days)

**Triggering builds programmatically** (via GitHub CLI):
```bash
gh workflow run build-windows.yml -f expiration_days=14
```

#### Option 3: Windows VM

Use Parallels, VMware, VirtualBox, or cloud VM (Azure/AWS) to build on Windows.

### Manual PyInstaller Build

If you need more control:

```bash
# Clean previous builds
rm -rf build dist

# Build using spec file
pyinstaller TransformerOptimizer.spec --clean
```

---

## Distribution

### Time-Limited Executables

This build system creates executables that:
- Expire after a specified number of days
- Show an expiration message when opened after the date
- Self-delete when expired (optional behavior in launcher.py)

### Changing Expiration Date

**Method 1: Build Script**
```bash
python build_app.py --days 14  # 14 days from now
python build_app.py --days 30  # 30 days from now
```

**Method 2: Manual Edit**

Edit `launcher.py` line ~17:
```python
EXPIRATION_DATE = datetime.date(2026, 2, 1)  # Set your date
```

Then rebuild.

### Distributing macOS Builds

1. Zip the `.app` bundle:
   ```bash
   cd dist && zip -r TransformerOptimizer_macOS.zip TransformerOptimizer.app
   ```

2. Send the zip file to users

3. Instructions for users:
   - Unzip the file
   - Double-click `TransformerOptimizer.app`
   - If macOS blocks it: Right-click > Open > Open

### Distributing Windows Builds

1. The `.exe` file is self-contained
2. Users may need Visual C++ Redistributable 2015-2022

---

## Troubleshooting

### GitHub Actions Issues

**Workflow not visible:**
- Ensure the workflow file exists at `.github/workflows/build-windows.yml`
- Push the file to GitHub: `git push origin main`

**Build fails with import errors:**
- Check the workflow logs for missing packages
- Add missing packages to the `pip install` step in the workflow

**Artifact expired:**
- Artifacts are kept for 30 days by default
- Re-run the workflow to generate a new build

**Workflow won't trigger:**
- Ensure you have write access to the repository
- Check that Actions are enabled in repository settings

### Build Issues

**PyInstaller not found:**
```bash
pip install pyinstaller
```

**Missing hidden imports:**
Add the missing module to `TransformerOptimizer.spec` in `hidden_imports` list.

**Build takes too long:**
- Ensure you're using a clean build (`--clean` flag)
- Close other applications to free memory

### macOS Issues

**"App is damaged and can't be opened":**
```bash
xattr -cr dist/TransformerOptimizer.app
```

**"Apple cannot check it for malicious software":**
Right-click the app > Open > Open

**Code signing for distribution:**
```bash
codesign --deep --force --sign - dist/TransformerOptimizer.app
```

### Windows Issues

**Missing DLLs:**
Install Visual C++ Redistributable 2015-2022 from Microsoft.

**Antivirus blocking:**
Some antivirus software flags PyInstaller executables. Add an exception or use code signing.

**Windows SmartScreen warning:**
Users can click "More info" > "Run anyway"

### Runtime Issues

**GPU not detected:**
- Ensure PyTorch is installed with CUDA/MPS support
- For Apple Silicon: Install `mlx` package
- Check GPU availability in Python:
  ```python
  import torch
  print(torch.backends.mps.is_available())  # macOS
  print(torch.cuda.is_available())          # NVIDIA
  ```

**Slow performance:**
- Ensure numba is properly installed
- Check that multi-threading is enabled
- Use GPU acceleration if available

### Large File Size

The executable includes PyTorch, NumPy, SciPy, etc. To reduce size:

1. Use PyTorch CPU-only builds:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```

2. Add more excludes to `TransformerOptimizer.spec`:
   ```python
   excludes=[
       'tkinter',
       'matplotlib',
       'PIL',
       'IPython',
       'jupyter',
       # Add more unused packages
   ],
   ```

---

## Security Notes

The expiration mechanism is embedded in the executable. Determined users could bypass it by:
- Setting system clock back
- Reverse engineering the binary

For stronger protection, consider:
- Online license validation
- Hardware-locked licenses
- Code obfuscation (PyArmor)
- Commercial licensing solutions

---

## Version History

- **v1.1** - Current version
- **v1.0** - Initial release

---

## Support

For issues and feature requests, visit:
https://github.com/LamyTheGoat/TrafoDes/issues
