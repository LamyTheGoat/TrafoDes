# TrafoDes - Transformer Design Optimizer

<div align="center">

**Professional-grade transformer design and optimization software**

[![Version](https://img.shields.io/badge/version-1.1-blue.svg)](https://github.com/LamyTheGoat/TrafoDes)
[![Python](https://img.shields.io/badge/python-3.8--3.11-green.svg)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS-lightgrey.svg)]()
[![License](https://img.shields.io/badge/license-Proprietary-red.svg)]()

[Features](#features) • [Installation](#installation) • [Usage](#usage) • [Documentation](#documentation)

</div>

---

## Overview

TrafoDes is an advanced transformer design optimization tool that automatically calculates optimal transformer dimensions and specifications to meet design constraints while minimizing manufacturing costs. Built for electrical engineers and transformer manufacturers, it leverages modern optimization algorithms and GPU acceleration to rapidly evaluate thousands of design configurations.

### Key Capabilities

- **Automated Optimization** — Multi-variable optimization of transformer parameters using advanced algorithms
- **GPU Acceleration** — Support for Apple Silicon (MPS/MLX) and NVIDIA CUDA for faster computations
- **Real-time Analysis** — Instant loss calculations, impedance analysis, and cost estimation
- **Thermal Design** — Integrated cooling duct calculations and thermal gradient analysis

---

## Features

### Design Parameters
- Power ratings from 100 kVA to 100,000 kVA
- HV/LV voltage configurations with Delta/Star connections
- Rectangular and round (obround) core geometries
- Customizable core diameter, foil dimensions, turns, and wire specifications

### Constraint Management
- No-Load Loss (NLL) targets with tolerance control
- Load Loss (LL) constraints and optimization
- Short-circuit impedance (Ucc) with configurable tolerance
- Penalty-based constraint handling for robust solutions

### Material Support
- **Conductors:** Copper and Aluminum with custom pricing
- **Core Materials:** M5, M4, M3, Hi-B steel grades
- **Insulation:** Configurable thickness parameters

### Optimization Engines
| Method | Description |
|--------|-------------|
| Smart Grid Search | Two-phase coarse-to-fine parameter sweep |
| Differential Evolution | Population-based stochastic optimization |
| GPU Accelerated | Parallel evaluation on Apple MPS or NVIDIA CUDA |
| Hybrid | GPU screening with CPU refinement |
| Multi-Seed DE | Parallel independent optimization runs |

### Analysis Output
- Optimal design parameters (turns, dimensions, core specifications)
- Performance metrics (actual vs. guaranteed losses)
- Detailed cost breakdown by material
- Cooling duct configuration recommendations
- Complete calculation logs

---

## Screenshots

<div align="center">

| Main Interface | Optimization Progress |
|:--------------:|:---------------------:|
| *Design input and parameter configuration* | *Real-time optimization with progress tracking* |

| Results View | Material Properties |
|:------------:|:-------------------:|
| *Detailed results and cost breakdown* | *Advanced material customization* |

</div>

> Screenshots coming soon

---

## System Requirements

### Minimum
- **OS:** Windows 10+ or macOS 11+
- **Python:** 3.8 - 3.11
- **RAM:** 4 GB
- **CPU:** Any modern multi-core processor

### Recommended
- **RAM:** 8 GB+
- **GPU:** Apple Silicon (M1/M2/M3) or NVIDIA with CUDA support
- **CPU:** 4+ cores for parallel optimization

---

## Installation

### Quick Start

```bash
# Clone the repository
git clone https://github.com/LamyTheGoat/TrafoDes.git
cd TrafoDes

# Create virtual environment
python -m venv .venv

# Activate (macOS/Linux)
source .venv/bin/activate
# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch the application
python launcher.py
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `dearpygui` | Modern GPU-accelerated GUI framework |
| `numpy` | Numerical computations |
| `scipy` | Optimization algorithms |
| `numba` | JIT compilation for performance |
| `torch` | GPU acceleration (MPS/CUDA) |
| `playsound3` | Audio notifications |

**Optional:**
- `mlx` — Apple Silicon native acceleration (macOS)
- `ezdxf` — CAD/DXF export support
- `streamlit` — Alternative web interface

---

## Usage

### Desktop Application

```bash
python launcher.py
```

The main interface provides:
1. **Input Panel** — Enter transformer specifications (power, voltages, frequency)
2. **Material Properties** — Configure conductor and core material parameters
3. **Optimization Settings** — Select algorithm and constraint tolerances
4. **Results Display** — View optimized parameters, losses, and costs

### Alternative Web Interface

```bash
pip install streamlit plotly pandas
streamlit run app.py
```

### Direct Module Access

```bash
python transformer_ui.py   # Bypass launcher
python mainRect.py         # CLI optimization engine
```

---

## Project Structure

```
TrafoDes/
├── launcher.py              # Application entry point
├── transformer_ui.py        # Main GUI (Dear PyGui)
├── mainRect.py              # Rectangular core optimizer
├── mainRound.py             # Round core optimizer
├── turboOpt.py              # Performance-optimized variant
├── app.py                   # Streamlit web interface
├── build_app.py             # Build automation script
├── TransformerOptimizer.spec # PyInstaller configuration
├── requirements.txt         # Python dependencies
├── BUILD_INSTRUCTIONS.md    # Detailed build guide
└── .github/workflows/       # CI/CD automation
```

---

## Building Executables

### Quick Build

```bash
python build_app.py --days 30
```

### Platform Builds

| Platform | Output | Command |
|----------|--------|---------|
| macOS | `TransformerOptimizer.app` | `python build_app.py --days 30` |
| Windows | `TransformerOptimizer.exe` | Use GitHub Actions or build on Windows |

For detailed build instructions, CI/CD setup, and distribution guidelines, see [BUILD_INSTRUCTIONS.md](BUILD_INSTRUCTIONS.md).

---

## Technical Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface Layer                      │
│  ┌─────────────────────┐    ┌─────────────────────────────┐ │
│  │  Dear PyGui (GUI)   │    │  Streamlit (Web)            │ │
│  └──────────┬──────────┘    └──────────────┬──────────────┘ │
└─────────────┼──────────────────────────────┼────────────────┘
              │                              │
              ▼                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Optimization Engine                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Grid Search │  │ Diff. Evol. │  │ GPU Accelerated     │  │
│  │  (Numba)    │  │  (SciPy)    │  │  (PyTorch/MLX)      │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Calculation Core                           │
│  • Core geometry & magnetic induction                        │
│  • Winding resistance & losses (LV/HV)                       │
│  • Short-circuit impedance                                   │
│  • Thermal gradients & cooling requirements                  │
│  • Cost estimation                                           │
└─────────────────────────────────────────────────────────────┘
```

---

## Performance

TrafoDes achieves high performance through:

- **Numba JIT Compilation** — 100+ calculation functions compiled to machine code
- **Parallel Processing** — Multi-core CPU utilization via Numba parallel loops
- **GPU Acceleration** — Apple MPS and NVIDIA CUDA support via PyTorch
- **Smart Algorithms** — Coarse-to-fine search reduces evaluation count by 90%+

Typical optimization times:
| Complexity | CPU Only | With GPU |
|------------|----------|----------|
| Simple | 5-15 sec | 2-5 sec |
| Standard | 30-60 sec | 10-20 sec |
| Complex | 2-5 min | 30-90 sec |

---

## Documentation

- [Build Instructions](BUILD_INSTRUCTIONS.md) — Complete guide for development setup and building executables
- [Requirements](requirements.txt) — Python package dependencies

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.1 | Jan 2026 | Build system improvements, UI updates |
| 1.0 | Dec 2025 | Initial release |

---

## Support

For issues, feature requests, or questions:

- **GitHub Issues:** [github.com/LamyTheGoat/TrafoDes/issues](https://github.com/LamyTheGoat/TrafoDes/issues)

---

## License

This software is proprietary. All rights reserved.

---

<div align="center">

**TrafoDes** — Precision Transformer Design

*Engineered for efficiency*

</div>
