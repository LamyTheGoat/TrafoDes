"""
Transformer Design Optimization
Fast optimization using Numba parallel processing and scipy optimization algorithms.
"""

import math
import time
import numpy as np

# Optional Numba imports for parallel CPU processing
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Create dummy decorators
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator
    prange = range
    print("Warning: Numba not available. CPU parallel processing disabled.")

# Optional scipy imports for smart optimization
try:
    from scipy.optimize import differential_evolution, minimize
    from scipy.stats import qmc
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Smart optimization disabled.")

# Optional CUDA imports for GPU acceleration
try:
    from numba import cuda
    import numba.cuda
    CUDA_AVAILABLE = cuda.is_available()
    if CUDA_AVAILABLE:
        print(f"CUDA GPU detected: {cuda.get_current_device().name}")
except ImportError:
    CUDA_AVAILABLE = False
except Exception:
    CUDA_AVAILABLE = False

if not CUDA_AVAILABLE:
    print("Warning: CUDA not available.")

# Optional PyTorch MPS imports for Apple Silicon GPU acceleration
try:
    import torch
    MPS_AVAILABLE = torch.backends.mps.is_available() and torch.backends.mps.is_built()
    if MPS_AVAILABLE:
        print(f"Apple MPS GPU detected (PyTorch {torch.__version__})")
except ImportError:
    MPS_AVAILABLE = False
    torch = None
except Exception:
    MPS_AVAILABLE = False
    torch = None

if not MPS_AVAILABLE:
    print("Warning: PyTorch MPS not available.")

# Optional MLX imports for Apple Silicon GPU acceleration
try:
    import mlx.core as mx
    MLX_AVAILABLE = True
    print("MLX available for Apple Silicon GPU acceleration")
except ImportError:
    MLX_AVAILABLE = False
    mx = None
except Exception:
    MLX_AVAILABLE = False
    mx = None

if not MLX_AVAILABLE:
    print("Warning: MLX not available.")

# Summary of GPU options
GPU_OPTIONS = []
if CUDA_AVAILABLE:
    GPU_OPTIONS.append("cuda")
if MPS_AVAILABLE:
    GPU_OPTIONS.append("mps")
if MLX_AVAILABLE:
    GPU_OPTIONS.append("mlx")

if GPU_OPTIONS:
    print(f"GPU acceleration available: {', '.join(GPU_OPTIONS)}")
else:
    print("No GPU acceleration available. Using CPU parallel processing.")


# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# Connection Types
ConnectionType = {
    "D": 1,
    "Y": 1 / math.sqrt(3),
}

# Parameter Bounds
CORELENGTH_MINIMUM = 0
CORE_MINIMUM = 80
CORE_MAXIMUM = 500
FOILHEIGHT_MINIMUM = 200
FOILHEIGHT_MAXIMUM = 1200
FOILTHICKNESS_MINIMUM = 0.3
FOILTHICKNESS_MAXIMUM = 4
FOILTURNS_MINIMUM = 5
FOILTURNS_MAXIMUM = 100
HVTHICK_MINIMUM = 1
HVTHICK_MAXIMUM = 5
HV_RATIO_MAXIMUM = 6
HV_RATION_MINIMUM = 1
HV_LEN_MAXIMUM = 20
HV_LEN_MINIMUM = 3.5

# Transformer Design Specifications
HVCONNECTION = ConnectionType["D"]
LVCONNECTION = ConnectionType["Y"]
POWERRATING = 1000
HVRATE = 20000 * HVCONNECTION
LVRATE = 400 * LVCONNECTION
FREQUENCY = 50

# Guaranteed Loss Limits
GUARANTEED_NO_LOAD_LOSS = 900 * 0.98 #* 1.15
GUARANTEED_LOAD_LOSS = 7500 * 0.98 #* 1.15
GUARANTEED_UCC = 6
UCC_TOLERANCE_PERCENT = 3
UCC_TOLERANCE = GUARANTEED_UCC * (UCC_TOLERANCE_PERCENT / 100)

# Penalty Factors
PENALTY_NLL_FACTOR = 60
PENALTY_LL_FACTOR = 10
PENALTY_UCC_FACTOR = 10000
MAX_GRADIENT = 21

# Insulation and Gap Constants
INSULATION_THICKNESS_WIRE = 0.12
LVInsulationThickness = 0.125
HVInsulationThickness = 0.5
MainGap = 12
DistanceCoreLV = 2
PhaseGap = 12
CoreFillingFactorRound = 0.84
CoreFillingFactorRectangular = 0.97

AdditionalLossFactorLV = 1.12  # DEPRECATED: Now calculated via IEC 60076 Dowell method
AdditionalLossFactorHV = 1.12  # DEPRECATED: Now calculated via IEC 60076 Dowell method

# Physical constant for IEC 60076 calculations
MU_0 = 4 * math.pi * 1e-7  # Permeability of free space (H/m)


@njit(fastmath=True)
def calculate_skin_depth(resistivity, frequency):
    """
    Calculate skin depth for conductor material (IEC 60076).

    Args:
        resistivity: Material resistivity in Ω·mm²/m (e.g., 0.021 for copper)
        frequency: Operating frequency in Hz

    Returns:
        Skin depth in mm
    """
    # Guard against invalid inputs
    if frequency <= 0 or resistivity <= 0:
        return 1000.0  # Return large value (effectively no skin effect)

    # Convert resistivity from Ω·mm²/m to Ω·m (multiply by 1e-6)
    resistivity_ohm_m = resistivity * 1e-6
    # δ = sqrt(ρ / (π × f × μ₀))
    delta_m = math.sqrt(resistivity_ohm_m / (math.pi * frequency * MU_0))
    return delta_m * 1000  # Convert m to mm


@njit(fastmath=True)
def calculate_dowell_factor(conductor_thickness_mm, n_layers, frequency,
                            resistivity, circular=False):
    """
    Calculate Dowell's AC resistance factor (F_R = R_ac / R_dc) per IEC 60076.

    This factor accounts for:
    - Skin effect (current crowding at conductor surface)
    - Proximity effect (induced currents from adjacent layers)

    Args:
        conductor_thickness_mm: Conductor dimension perpendicular to leakage flux (mm)
        n_layers: Number of winding layers
        frequency: Operating frequency (Hz)
        resistivity: Conductor resistivity in Ω·mm²/m
        circular: True if circular wire (applies sqrt(π)/2 correction)

    Returns:
        F_R: AC/DC resistance ratio (typically 1.0 to 3.0 for power transformers)
    """
    # Guard against invalid inputs - use 12% minimum safeguard
    if conductor_thickness_mm <= 0 or n_layers <= 0:
        return 1.12  # Minimum 12% safeguard for test compliance

    # Calculate skin depth
    delta = calculate_skin_depth(resistivity, frequency)

    # For circular wire, use equivalent rectangular dimension
    h = conductor_thickness_mm
    if circular:
        h = conductor_thickness_mm * math.sqrt(math.pi) / 2

    # Penetration ratio
    xi = h / delta

    # Avoid numerical issues for very small xi - still apply 12% safeguard
    if xi < 0.01:
        return 1.12  # Minimum 12% safeguard for test compliance

    # Dowell's M and D functions
    sinh_2xi = math.sinh(2 * xi)
    sin_2xi = math.sin(2 * xi)
    cosh_2xi = math.cosh(2 * xi)
    cos_2xi = math.cos(2 * xi)

    sinh_xi = math.sinh(xi)
    sin_xi = math.sin(xi)
    cosh_xi = math.cosh(xi)
    cos_xi = math.cos(xi)

    # M function (skin effect term)
    denom_M = cosh_2xi - cos_2xi
    if abs(denom_M) < 1e-10:
        denom_M = 1e-10
    M = xi * (sinh_2xi + sin_2xi) / denom_M

    # D function (proximity effect term)
    denom_D = cosh_xi + cos_xi
    if abs(denom_D) < 1e-10:
        denom_D = 1e-10
    D = 2 * xi * (sinh_xi - sin_xi) / denom_D

    # Dowell's formula: F_R = M + ((m² - 1) / 3) × D
    m = max(1.0, float(n_layers))
    F_R = M + ((m * m - 1) / 3.0) * D

    # Clamp to reasonable range (1.12 to 5.0) - 12% minimum safeguard for test compliance
    return max(1.12, min(5.0, F_R))


@njit(fastmath=True)
def calculate_hv_layer_count(lv_foil_height, lv_turns, hv_wire_thickness,
                              hv_wire_length, hv_rate, lv_rate,
                              insulation_thickness, circular=False):
    """
    Calculate number of HV winding layers.

    Args:
        lv_foil_height: LV foil height in mm
        lv_turns: Number of LV turns
        hv_wire_thickness: HV wire thickness (radial dimension) in mm
        hv_wire_length: HV wire length (axial dimension) in mm
        hv_rate: HV voltage rating
        lv_rate: LV voltage rating
        insulation_thickness: Insulation thickness between turns in mm
        circular: True if circular wire

    Returns:
        Number of HV layers
    """
    hv_turns = lv_turns * (hv_rate / lv_rate)
    hv_layer_height = lv_foil_height - 50

    wire_axial_size = hv_wire_thickness if circular else hv_wire_length
    turns_per_layer = (hv_layer_height / (wire_axial_size + insulation_thickness)) - 1

    if turns_per_layer <= 0:
        return 1

    return int(math.ceil(hv_turns / turns_per_layer))


# Material Properties
CoreDensity = 7.65
CorePricePerKg = 3.6
AlDensity = 2.7
CuDensity = 8.9
AlRestivity = 0.0336

# Cooling Duct Constants
COOLING_DUCT_THICKNESS = 4
COOLING_DUCT_CENTER_TO_CENTER_DISTANCE = 25
COOLING_DUCT_WIDTH = 7

# Material Selection (Copper)
materialToBeUsedFoil_Density = CuDensity
materialToBeUsedFoil_Price = 11.55  # CU_PRICE_FOIL
materialToBeUsedFoil_Resistivity = 0.021  # CU_RESISTIVITY
materialToBeUsedWire_Density = CuDensity
materialToBeUsedWire_Price = 11.05  # CU_PRICE_WIRE
materialToBeUsedWire_Resistivity = 0.021  # CU_RESISTIVITY
materialCore_Density = CoreDensity
materialCore_Price = CorePricePerKg
oilToBeUsed_Factor = 1  # OIL_MINERAL_FACTOR

# HV Wire Shape Option
# When True, HV wire is circular (radius = HVWireThickness / 2)
# When False, HV wire is rectangular with rounded corners (default)
HV_WIRE_CIRCULAR = False


# =============================================================================
# CORE CALCULATION FUNCTIONS
# =============================================================================

@njit(fastmath=True)
def Clamp(Number, Minimum, Maximum):
    if Number < Minimum:
        return Minimum
    elif Number > Maximum:
        return Maximum
    return Number


@njit(fastmath=True)
def CalculateVoltsPerTurns(LVRate, LVTurns):
    if LVTurns <= 0:
        return 1e18
    return LVRate / LVTurns


@njit(fastmath=True)
def CalculateCoreSection(CoreDiameter, CoreLength):
    return ((CoreDiameter**2 * math.pi) / (4 * 100)) * CoreFillingFactorRound + \
           (CoreLength * CoreDiameter / 100) * CoreFillingFactorRectangular


@njit(fastmath=True)
def CalculateInduction(VolsPerTurn, CoreDiameter, CoreLength):
    core_section = CalculateCoreSection(CoreDiameter, CoreLength)
    if core_section <= 0:
        return 1e18
    denom = math.sqrt(2) * math.pi * FREQUENCY * core_section
    if denom <= 0:
        return 1e18
    return (VolsPerTurn * 10000) / denom


@njit(fastmath=True)
def CalculateWattsPerKG(Induction):
    return (1.3498 * Induction**6) + (-8.1737 * Induction**5) + \
           (19.884 * Induction**4) + (-24.708 * Induction**3) + \
           (16.689 * Induction**2) + (-5.5386 * Induction) + 0.7462


@njit(fastmath=True)
def CalculateRadialThicknessLV(LVNumberOfTurns, LVFoilThickness, NumberOfDucts=0, ThicknessOfDucts=COOLING_DUCT_THICKNESS):
    return LVNumberOfTurns * LVFoilThickness + ((LVNumberOfTurns - 1) * LVInsulationThickness) + \
           (NumberOfDucts * (ThicknessOfDucts + 0.5))


@njit(fastmath=True)
def CalculateAverageDiameterLV(LVNumberOfTurns, LVFoilThickness, CoreDiameter, CoreLength, NumberOfDucts):
    return CoreDiameter + (2 * DistanceCoreLV) + CalculateRadialThicknessLV(LVNumberOfTurns, LVFoilThickness, NumberOfDucts) + \
           (2 * CoreLength / math.pi)


@njit(fastmath=True)
def CalculateTotalLengthCoilLV(LVNumberOfTurns, LVFoilThickness, CoreDiameter, CoreLength, NumberOfDucts):
    return CalculateAverageDiameterLV(LVNumberOfTurns, LVFoilThickness, CoreDiameter, CoreLength, NumberOfDucts) * \
           math.pi * LVNumberOfTurns


@njit(fastmath=True)
def CalculateRadialThicknessHV(LVFoilHeight, LVNumberOfTurns, HVWireThickness, HVWireLength, NumberOfDucts=0, ThicknessOfDucts=COOLING_DUCT_THICKNESS, circular=False):
    HVNumberOfTurns = LVNumberOfTurns * (HVRATE / LVRATE)
    HVLayerHeight = LVFoilHeight - 50
    # For circular wire, axial space per turn = diameter (thickness)
    wire_axial_size = HVWireThickness if circular else HVWireLength
    HVTurnsPerLayer = (HVLayerHeight / (wire_axial_size + INSULATION_THICKNESS_WIRE)) - 1
    if HVTurnsPerLayer <= 0:
        return 1e18
    HVLayerNumber = math.ceil(HVNumberOfTurns / HVTurnsPerLayer)
    return HVLayerNumber * HVWireThickness + (HVLayerNumber - 1) * HVInsulationThickness + \
           (NumberOfDucts * (ThicknessOfDucts + 0.5))


@njit(fastmath=True)
def CalculateAverageDiameterHV(LVNumberOfTurns, LVFoilThickness, CoreDiameter, LVFoilHeight, HVWireThickness, HVWireLength, CoreLength, NumberOfDucts, circular=False):
    return CoreDiameter + 2 * DistanceCoreLV + 2 * CalculateRadialThicknessLV(LVNumberOfTurns, LVFoilThickness, NumberOfDucts) + \
           2 * MainGap + CalculateRadialThicknessHV(LVFoilHeight, LVNumberOfTurns, HVWireThickness, HVWireLength, NumberOfDucts, circular=circular) + \
           (2 * CoreLength / math.pi)


@njit(fastmath=True)
def CalculateTotalLengthCoilHV(LVNumberOfTurns, LVFoilThickness, CoreDiameter, LVFoilHeight, HVWireThickness, HVWireLength, CoreLength, NumberOfDucts, circular=False):
    HVNumberOfTurns = LVNumberOfTurns * HVRATE / LVRATE
    return CalculateAverageDiameterHV(LVNumberOfTurns, LVFoilThickness, CoreDiameter, LVFoilHeight, HVWireThickness, HVWireLength, CoreLength, NumberOfDucts, circular) * \
           math.pi * HVNumberOfTurns


@njit(fastmath=True)
def CalculateCoreWeight(LVNumberOfTurns, LVFoilHeight, LVFoilThickness, HVWireThickness, HVWireLength, CoreDiameter, CoreLength, NumberOfDuctsLv, NumberOfDuctsHv, circular=False):
    FoilHeight = LVFoilHeight
    WindowHeight = FoilHeight + 40
    RadialThickness = CalculateRadialThicknessLV(LVNumberOfTurns, LVFoilThickness, NumberOfDuctsLv) + \
                      CalculateRadialThicknessHV(LVFoilHeight, LVNumberOfTurns, HVWireThickness, HVWireLength, NumberOfDuctsHv, circular=circular) + \
                      MainGap + DistanceCoreLV
    CenterBetweenLegs = (CoreDiameter + RadialThickness * 2) + PhaseGap
    rectWeight = (((3 * WindowHeight) + 2 * (2 * CenterBetweenLegs + CoreDiameter)) * (CoreDiameter * CoreLength / 100) * CoreDensity * CoreFillingFactorRectangular) / 1e6
    radius = CoreDiameter / 2
    squareEdge = radius * math.sqrt(math.pi)
    roundWeight = (((3 * (WindowHeight + 10)) + 2 * (2 * CenterBetweenLegs + CoreDiameter)) * (squareEdge * squareEdge / 100) * CoreDensity * CoreFillingFactorRound) / 1e6
    return (rectWeight + roundWeight) * 100


@njit(fastmath=True)
def CalculateVolumeLV(LVNumberOfTurns, LVFoilThickness, CoreDiameter, LVFoilHeight, CoreLength, NumberOfDucts):
    length = CalculateTotalLengthCoilLV(LVNumberOfTurns, LVFoilThickness, CoreDiameter, CoreLength, NumberOfDucts)
    return (length * LVFoilHeight * LVFoilThickness) / 1000000


@njit(fastmath=True)
def CalculateSectionHV(Thickness, Length, circular=False):
    """Calculate HV wire cross-section area.

    Args:
        Thickness: Wire thickness (diameter for circular)
        Length: Wire length (ignored for circular)
        circular: If True, calculate circular wire section (pi * r^2)
                  If False, calculate rectangular with rounded corners (default)
    """
    if circular:
        # Circular wire: area = pi * (diameter/2)^2
        radius = Thickness / 2.0
        return math.pi * radius * radius
    else:
        # Rectangular wire with rounded corners
        return (Thickness * Length) - (Thickness**2) + (((Thickness / 2)**2) * math.pi)


@njit(fastmath=True)
def CalculateSectionLV(HeightLV, ThicknessLV):
    return HeightLV * ThicknessLV


@njit(fastmath=True)
def CalculateVolumeHV(LVNumberOfTurns, LVFoilThickness, CoreDiameter, LVFoilHeight, HVWireThickness, HVWireLength, CoreLength, NumberOfDucts, circular=False):
    length = CalculateTotalLengthCoilHV(LVNumberOfTurns, LVFoilThickness, CoreDiameter, LVFoilHeight, HVWireThickness, HVWireLength, CoreLength, NumberOfDucts, circular)
    return (length * CalculateSectionHV(HVWireThickness, HVWireLength, circular)) / 1000000


@njit(fastmath=True)
def CalculateWeightOfVolume(Volume, MaterialDensity):
    return Volume * MaterialDensity


@njit(fastmath=True)
def CalculatePriceOfWeight(Weight, MaterialPrice):
    return Weight * MaterialPrice


@njit(fastmath=True)
def CalculateResistanceLV(MaterialResistivity, Length, Section):
    if Section <= 0:
        return 1e18
    return (Length / 1000) * MaterialResistivity / Section


@njit(fastmath=True)
def CalculateResistanceHV(MaterialResistivity, Length, Section):
    if Section <= 0:
        return 1e18
    return (Length / 1000) * MaterialResistivity / Section


@njit(fastmath=True)
def CalculateCurrent(Power, Voltage):
    if Voltage <= 0:
        return 1e18
    return (Power * 1000) / (Voltage * 3)


@njit(fastmath=True)
def CalculateCurrentLV(Power, VoltageLV):
    return CalculateCurrent(Power, VoltageLV)


@njit(fastmath=True)
def CalculateCurrentHV(Power, VoltageHV):
    return CalculateCurrent(Power, VoltageHV)


@njit(fastmath=True)
def CalculateLoadLosses(LVNumberOfTurns, LVFoilThickness, CoreDiameter, LVFoilHeight, HVWireThickness, HVWireLength, MaterialResistivity, Power, HVRating, LVRating, CoreLength, NumberOfDuctsLv, NumberOfDuctsHv, circular=False, frequency=FREQUENCY):
    lvLength = CalculateTotalLengthCoilLV(LVNumberOfTurns, LVFoilThickness, CoreDiameter, CoreLength, NumberOfDuctsLv)
    hvLength = CalculateTotalLengthCoilHV(LVNumberOfTurns, LVFoilThickness, CoreDiameter, LVFoilHeight, HVWireThickness, HVWireLength, CoreLength, NumberOfDuctsHv, circular)
    lvSection = CalculateSectionLV(LVFoilHeight, LVFoilThickness)
    hvSection = CalculateSectionHV(HVWireThickness, HVWireLength, circular)
    lvResistance = CalculateResistanceLV(MaterialResistivity, lvLength, lvSection)
    hvResistance = CalculateResistanceHV(MaterialResistivity, hvLength, hvSection)
    hvCurrent = CalculateCurrentHV(Power, HVRating)
    lvCurrent = CalculateCurrentLV(Power, LVRating)

    # Calculate IEC 60076 Dowell factors (replaces fixed 1.12)
    # LV: foil winding, each turn = 1 layer
    n_layers_lv = LVNumberOfTurns
    F_R_LV = calculate_dowell_factor(LVFoilThickness, n_layers_lv, frequency,
                                      MaterialResistivity, circular=False)

    # HV: wire winding, multiple turns per layer
    n_layers_hv = calculate_hv_layer_count(LVFoilHeight, LVNumberOfTurns,
                                            HVWireThickness, HVWireLength,
                                            HVRating, LVRating, INSULATION_THICKNESS_WIRE,
                                            circular)
    F_R_HV = calculate_dowell_factor(HVWireThickness, n_layers_hv, frequency,
                                      MaterialResistivity, circular=circular)

    lvLosses = lvResistance * (lvCurrent**2) * 3 * F_R_LV
    hvLosses = hvResistance * (hvCurrent**2) * 3 * F_R_HV
    return lvLosses + hvLosses, lvLosses, hvLosses


@njit(fastmath=True)
def CalculateNoLoadLosses(LVNumberOfTurns, LVFoilHeight, LVFoilThickness, HVWireThickness, HVWireLength, CoreDiameter, LVRate, CoreLength, NumberOfDuctsLv, NumberOfDuctsHv, circular=False):
    CoreWeight = CalculateCoreWeight(LVNumberOfTurns, LVFoilHeight, LVFoilThickness, HVWireThickness, HVWireLength, CoreDiameter, CoreLength, NumberOfDuctsLv, NumberOfDuctsHv, circular)
    Induction = CalculateInduction(CalculateVoltsPerTurns(LVRate, LVNumberOfTurns), CoreDiameter, CoreLength)
    WattsPerKG = CalculateWattsPerKG(Induction)
    return WattsPerKG * CoreWeight * 1.2


@njit(fastmath=True)
def CalculateStrayDiameter(LVNumberOfTurns, LVFoilThickness, LVFoilHeight, HVWireThickness, HVWireLength, DiameterOfCore, CoreLength, NumberOfDuctsLV, NumberOfDuctsHV, circular=False):
    MainGapDiameter = DiameterOfCore + DistanceCoreLV * 2 + CalculateRadialThicknessLV(LVNumberOfTurns, LVFoilThickness, NumberOfDuctsLV) * 2 + \
                      (2 * CoreLength / math.pi) + MainGap
    HVRadialThickness = CalculateRadialThicknessHV(LVFoilHeight, LVNumberOfTurns, HVWireThickness, HVWireLength, NumberOfDuctsHV, circular=circular)
    ReducedWidthHV = HVRadialThickness / 3
    LVRadialThickness = CalculateRadialThicknessLV(LVNumberOfTurns, LVFoilThickness, NumberOfDuctsLV)
    ReducedWidthLV = LVRadialThickness / 3
    denom = ReducedWidthLV + ReducedWidthHV + MainGap
    if denom <= 0:
        denom = 0.001  # Safety
    SD = MainGapDiameter + ReducedWidthHV - ReducedWidthLV + ((ReducedWidthHV**2 - ReducedWidthLV**2) / denom)
    return SD


@njit(fastmath=True)
def CalculateUr(LoadLosses, Power):
    if Power <= 0:
        return 1e18
    return LoadLosses / (10 * Power)


@njit(fastmath=True)
def CalculateUx(Power, StrayDiameter, LVNumberOfTurns, LVFoilThickness, LVFoilHeight, HVWireThickness, HVWireLength, Frequency, LVRate, NumberOfDuctsLV, NumberOfDuctsHV, circular=False):
    if LVFoilHeight <= 0:
        return 1e18
    HVRadialThickness = CalculateRadialThicknessHV(LVFoilHeight, LVNumberOfTurns, HVWireThickness, HVWireLength, NumberOfDuctsHV, circular=circular)
    if HVRadialThickness >= 1e17:
        return 1e18
    ReducedWidthHV = HVRadialThickness / 3
    LVRadialThickness = CalculateRadialThicknessLV(LVNumberOfTurns, LVFoilThickness, NumberOfDuctsLV)
    ReducedWidthLV = LVRadialThickness / 3
    volts_per_turn = CalculateVoltsPerTurns(LVRate, LVNumberOfTurns)
    if volts_per_turn >= 1e17:
        return 1e18
    denom = 1210 * (volts_per_turn**2) * LVFoilHeight
    if denom <= 0:
        return 1e18
    return (Power * StrayDiameter * Frequency * (ReducedWidthLV + ReducedWidthHV + MainGap)) / denom


@njit(fastmath=True)
def CalculateImpedance(Ux, Ur):
    return math.sqrt(Ux**2 + Ur**2)


@njit(fastmath=True)
def CalculatePrice(LVNumberOfTurns, LVFoilHeight, LVFoilThickness, HVWireThickness, HVWireLength, CoreDiameter, CoreLength, NumberOfDuctsLV, NumberOfDuctsHV, printValues=False, circular=False):
    wc = CalculateCoreWeight(LVNumberOfTurns, LVFoilHeight, LVFoilThickness, HVWireThickness, HVWireLength, CoreDiameter, CoreLength, NumberOfDuctsLV, NumberOfDuctsHV, circular)
    pc = CalculatePriceOfWeight(wc, materialCore_Price)
    volumeHV = CalculateVolumeHV(LVNumberOfTurns, LVFoilThickness, CoreDiameter, LVFoilHeight, HVWireThickness, HVWireLength, CoreLength, NumberOfDuctsHV, circular) * 3
    whv = CalculateWeightOfVolume(volumeHV, materialToBeUsedWire_Density)
    phv = CalculatePriceOfWeight(whv, materialToBeUsedWire_Price)
    volumeLV = CalculateVolumeLV(LVNumberOfTurns, LVFoilThickness, CoreDiameter, LVFoilHeight, CoreLength, NumberOfDuctsLV) * 3
    wlv = CalculateWeightOfVolume(volumeLV, materialToBeUsedFoil_Density)
    plv = CalculatePriceOfWeight(wlv, materialToBeUsedFoil_Price)
    if printValues:
        print("CORE WEIGHT = ", wc)
        print("FOIL WEIGHT = ", wlv)
        print("WIRE WEIGHT = ", whv)
    return pc + phv + plv


# =============================================================================
# COOLING DUCT CALCULATIONS
# =============================================================================

@njit(fastmath=True)
def CalculateEfficencyOfCoolingDuct(FoilHeight, CoolingDuctThickness=COOLING_DUCT_THICKNESS):
    if FoilHeight <= 0:
        return 1e18
    return min((CoolingDuctThickness / (0.949 * (FoilHeight**0.25))), 1)


@njit(fastmath=True)
def CalculateEfficencyOfMainGap(FoilHeight, DistanceBetweenCoils=MainGap):
    if FoilHeight <= 0:
        return 1e18
    return min(((DistanceBetweenCoils - 0.5) / (0.949 * (FoilHeight**0.25))), 1)


@njit(fastmath=True)
def CalculateTotalInsulationThicknessLV(LVTurns, insulationThicknessLV=LVInsulationThickness):
    return (LVTurns - 1) * insulationThicknessLV


@njit(fastmath=True)
def CalculateTotalInsulationThicknessHV(NumLayerHV, insulationThicknessHV=HVInsulationThickness):
    return (NumLayerHV + 1) * insulationThicknessHV


@njit(fastmath=True)
def CalculateHeatFluxLV(LoadLossesLV, FoilHeight, LVNumberOfTurns, LVFoilThickness, CoreDiameter, CoreLength=0):
    if LVNumberOfTurns <= 0 or FoilHeight <= 0:
        return 1e18
    AverageLengthLV = CalculateTotalLengthCoilLV(LVNumberOfTurns, LVFoilThickness, CoreDiameter, CoreLength, 0) / LVNumberOfTurns
    if AverageLengthLV <= 0:
        return 1e18
    val = (((LoadLossesLV / 3) * 1.03) / (AverageLengthLV * FoilHeight))
    return val * 10**4


@njit(fastmath=True)
def CalculateHeatFluxHV(LoadLossesHV, HVHeight, FoilHeight, LVNumberOfTurns, LVFoilThickness, CoreDiameter, HVWireThickness, HVWireLength, CoreLength=0, circular=False):
    if HVHeight <= 0:
        return 1e18
    AverageLengthHV = CalculateAverageDiameterHV(LVNumberOfTurns, LVFoilThickness, CoreDiameter, FoilHeight, HVWireThickness, HVWireLength, CoreLength, 0, circular) * math.pi
    if AverageLengthHV <= 0:
        return 1e18
    val = (((LoadLossesHV / 3) * 1.03) / (AverageLengthHV * HVHeight))
    return val * 10**4


@njit(fastmath=True)
def AidingFormulaHVOpenDucts(HeatFluxHV, MainGapEfficiency, EfficiencyDuct, NumberDucts, DistanceBetWeenCoreLV=DistanceCoreLV, CCDucts=COOLING_DUCT_CENTER_TO_CENTER_DISTANCE, WidthDuct=COOLING_DUCT_WIDTH):
    if CCDucts <= 0:
        return 1e18
    denom = 2 * (0.5 + (((CCDucts - WidthDuct) / CCDucts) * ((0.5 * MainGapEfficiency) + (NumberDucts * EfficiencyDuct))))
    if denom <= 0:
        return 1e18
    return HeatFluxHV / denom


@njit(fastmath=True)
def AidingFormulaHVCloseDucts(HeatFluxHV, MainGapEfficiency, EfficiencyDuct, NumberDucts, DistanceBetWeenCoreLV=DistanceCoreLV, CCDucts=COOLING_DUCT_CENTER_TO_CENTER_DISTANCE, WidthDuct=COOLING_DUCT_WIDTH):
    if CCDucts <= 0:
        return 1e18
    denom = 2 * (0.5 + (((CCDucts - WidthDuct) / CCDucts) * ((0.5 * MainGapEfficiency) + ((NumberDucts - 1) * EfficiencyDuct))))
    if denom <= 0:
        return 1e18
    return HeatFluxHV / denom


@njit(fastmath=True)
def AidingFormulaLVOpenDucts(HeatFluxLV, MainGapEfficiency, EfficiencyDuct, NumberDucts, DistanceBetWeenCoreLV=DistanceCoreLV, CCDucts=COOLING_DUCT_CENTER_TO_CENTER_DISTANCE, WidthDuct=COOLING_DUCT_WIDTH):
    if CCDucts <= 0:
        return 1e18
    coefCLV = 0.5 if DistanceCoreLV > 3 else 0.318
    denom = 2 * (coefCLV + (((CCDucts - WidthDuct) / CCDucts) * ((0.5 * MainGapEfficiency) + (NumberDucts * EfficiencyDuct))))
    if denom <= 0:
        return 1e18
    return HeatFluxLV / denom


@njit(fastmath=True)
def AidingFormulaLVCloseDuct(HeatFluxLV, MainGapEfficiency, EfficiencyDuct, NumberDucts, DistanceBetWeenCoreLV=DistanceCoreLV, CCDucts=COOLING_DUCT_CENTER_TO_CENTER_DISTANCE, WidthDuct=COOLING_DUCT_WIDTH):
    if CCDucts <= 0:
        return 1e18
    coefCLV = 0.5 if DistanceCoreLV > 3 else 0.318
    denom = 2 * (coefCLV + (((CCDucts - WidthDuct) / CCDucts) * ((0.5 * MainGapEfficiency) + ((NumberDucts - 1) * EfficiencyDuct))))
    if denom <= 0:
        return 1e18
    return HeatFluxLV / denom


@njit(fastmath=True)
def CalculateGradientHeatLV(TotalTurnsLV, HeatFluxLV, MainGapEfficiency, EfficiencyDuct, NumberDucts):
    openDuctVal = AidingFormulaLVOpenDucts(HeatFluxLV, MainGapEfficiency, EfficiencyDuct, NumberDucts)
    if openDuctVal >= 1e17:
        return 1e18
    closedDuctVal = openDuctVal if NumberDucts == 0 else AidingFormulaLVCloseDuct(HeatFluxLV, MainGapEfficiency, EfficiencyDuct, NumberDucts)
    if closedDuctVal >= 1e17:
        return 1e18
    denom = 6 * 1.16 * (1 + NumberDucts)
    if denom <= 0:
        return 1e18
    val0_1 = (1.754 * openDuctVal**0.8 + (openDuctVal * CalculateTotalInsulationThicknessLV(TotalTurnsLV)) / denom)
    val0_2 = 0.3 * (1.754 * closedDuctVal**0.8 + (closedDuctVal * CalculateTotalInsulationThicknessLV(TotalTurnsLV)) / denom)
    return (val0_1 + val0_2) * oilToBeUsed_Factor * (50 / 47)


@njit(fastmath=True)
def CalculateGradientHeatHV(TotalTurnsHV, HeatFluxHV, MainGapEfficiency, EfficiencyDuct, NumberDucts):
    openDuctVal = AidingFormulaHVOpenDucts(HeatFluxHV, MainGapEfficiency, EfficiencyDuct, NumberDucts)
    if openDuctVal >= 1e17:
        return 1e18
    closedDuctVal = openDuctVal if NumberDucts == 0 else AidingFormulaHVCloseDucts(HeatFluxHV, MainGapEfficiency, EfficiencyDuct, NumberDucts)
    if closedDuctVal >= 1e17:
        return 1e18
    denom = 6 * 1.5 * (1 + NumberDucts)
    if denom <= 0:
        return 1e18
    val0_1 = (0.75 * (1.754 * openDuctVal**0.8 + (openDuctVal * CalculateTotalInsulationThicknessHV(TotalTurnsHV)) / denom))
    val0_2 = (0.30 * (1.754 * closedDuctVal**0.8 + (closedDuctVal * CalculateTotalInsulationThicknessHV(TotalTurnsHV)) / denom))
    return (val0_1 + val0_2) * oilToBeUsed_Factor


@njit(fastmath=True)
def CalculateNumberOfCoolingDucts(LVNumberOfTurns, LVFoilHeight, LVFoilThickness, HVWireThickness, HVWireLength, CoreDiameter, CoreLength, LVRATE=LVRATE, HVRATE=HVRATE, POWER=POWERRATING, FREQ=FREQUENCY, MaterialResistivity=materialToBeUsedWire_Resistivity, GUARANTEEDNLL=GUARANTEED_NO_LOAD_LOSS, GUARANTEEDLL=GUARANTEED_LOAD_LOSS, GUARANTEEDUCC=GUARANTEED_UCC, tolerance=1, isFinal=False, circular=False):
    Ll, LlLv, LlHv = CalculateLoadLosses(LVNumberOfTurns, LVFoilThickness, CoreDiameter, LVFoilHeight, HVWireThickness, HVWireLength, MaterialResistivity, POWER, HVRATE, LVRATE, CoreLength, 0, 0, circular)
    heatFluxLV = CalculateHeatFluxLV(LlLv, LVFoilHeight, LVNumberOfTurns, LVFoilThickness, CoreDiameter, CoreLength)
    MainGapEff = CalculateEfficencyOfMainGap(LVFoilHeight)
    DuctEff = CalculateEfficencyOfCoolingDuct(LVFoilHeight)
    gradientLV = CalculateGradientHeatLV(LVNumberOfTurns, heatFluxLV, MainGapEff, DuctEff, 0)
    heatFluxHV = CalculateHeatFluxHV(LlHv, LVFoilHeight - 50, LVFoilHeight, LVNumberOfTurns, LVFoilThickness, CoreDiameter, HVWireThickness, HVWireLength, CoreLength, circular)
    HVNumberOfTurns = LVNumberOfTurns * HVRATE / LVRATE
    HVLayerHeight = LVFoilHeight - 50
    # For circular wire, axial space per turn = diameter (thickness)
    wire_axial_size = HVWireThickness if circular else HVWireLength
    wire_denom = wire_axial_size + INSULATION_THICKNESS_WIRE
    if wire_denom <= 0:
        return 0, 0
    HVTurnsPerLayer = (HVLayerHeight / wire_denom) - 1
    if HVTurnsPerLayer <= 0:
        return 0, 0
    HVLayerNumber = math.ceil(HVNumberOfTurns / HVTurnsPerLayer)
    gradientHV = CalculateGradientHeatHV(HVLayerNumber, heatFluxLV, MainGapEff, DuctEff, 0)
    numberOfDuctsLV = 0
    numberOfDuctsHV = 0
    if isFinal:
        print("Gradient LV : ", gradientLV)
        print("Gradient HV : ", gradientHV)
    if gradientLV > MAX_GRADIENT:
        numberOfDuctsLV = min((math.ceil(2 * (gradientLV / MAX_GRADIENT) / 1.5)) / 2, 6)
    if gradientHV > MAX_GRADIENT:
        numberOfDuctsHV = min((math.ceil(2 * (gradientHV / MAX_GRADIENT) / 1.5)) / 2, 6)
    return numberOfDuctsLV, numberOfDuctsHV


@njit(fastmath=True)
def CalculateNumberOfCoolingDucts_WithLosses(LVNumberOfTurns, LVFoilHeight, LVFoilThickness, HVWireThickness, HVWireLength, CoreDiameter, CoreLength, MaterialResistivity, POWER, HVRATE_VAL, LVRATE_VAL, isFinal=False, circular=False):
    """Returns (numberOfDuctsLV, numberOfDuctsHV, Ll_zero, LlLv_zero, LlHv_zero)"""
    Ll, LlLv, LlHv = CalculateLoadLosses(LVNumberOfTurns, LVFoilThickness, CoreDiameter, LVFoilHeight, HVWireThickness, HVWireLength, MaterialResistivity, POWER, HVRATE_VAL, LVRATE_VAL, CoreLength, 0, 0, circular)
    heatFluxLV = CalculateHeatFluxLV(LlLv, LVFoilHeight, LVNumberOfTurns, LVFoilThickness, CoreDiameter, CoreLength)
    MainGapEff = CalculateEfficencyOfMainGap(LVFoilHeight)
    DuctEff = CalculateEfficencyOfCoolingDuct(LVFoilHeight)
    gradientLV = CalculateGradientHeatLV(LVNumberOfTurns, heatFluxLV, MainGapEff, DuctEff, 0)
    heatFluxHV = CalculateHeatFluxHV(LlHv, LVFoilHeight - 50, LVFoilHeight, LVNumberOfTurns, LVFoilThickness, CoreDiameter, HVWireThickness, HVWireLength, CoreLength, circular)
    HVNumberOfTurns = LVNumberOfTurns * HVRATE_VAL / LVRATE_VAL
    HVLayerHeight = LVFoilHeight - 50
    # For circular wire, axial space per turn = diameter (thickness)
    wire_axial_size = HVWireThickness if circular else HVWireLength
    wire_denom = wire_axial_size + INSULATION_THICKNESS_WIRE
    if wire_denom <= 0:
        return 0, 0, Ll, LlLv, LlHv
    HVTurnsPerLayer = (HVLayerHeight / wire_denom) - 1
    if HVTurnsPerLayer <= 0:
        return 0, 0, Ll, LlLv, LlHv
    HVLayerNumber = math.ceil(HVNumberOfTurns / HVTurnsPerLayer)
    gradientHV = CalculateGradientHeatHV(HVLayerNumber, heatFluxHV, MainGapEff, DuctEff, 0)
    numberOfDuctsLV = 0.0
    numberOfDuctsHV = 0.0
    if isFinal:
        print("Gradient LV : ", gradientLV)
        print("Gradient HV : ", gradientHV)
    if gradientLV > MAX_GRADIENT:
        numberOfDuctsLV = min((math.ceil(2 * (gradientLV / MAX_GRADIENT) / 1.5)) / 2, 6)
    if gradientHV > MAX_GRADIENT:
        numberOfDuctsHV = min((math.ceil(2 * (gradientHV / MAX_GRADIENT) / 1.5)) / 2, 6)
    return numberOfDuctsLV, numberOfDuctsHV, Ll, LlLv, LlHv


# =============================================================================
# PRICE CALCULATION FUNCTIONS
# =============================================================================

@njit(fastmath=True)
def CalculateFinalizedPrice(LVNumberOfTurns, LVFoilHeight, LVFoilThickness, HVWireThickness, HVWireLength, CoreDiameter, CoreLength=0, LVRATE=LVRATE, HVRATE=HVRATE, POWER=POWERRATING, FREQ=FREQUENCY, MaterialResistivity=materialToBeUsedWire_Resistivity, GUARANTEEDNLL=GUARANTEED_NO_LOAD_LOSS, GUARANTEEDLL=GUARANTEED_LOAD_LOSS, GUARANTEEDUCC=GUARANTEED_UCC, isFinal=False, PutCoolingDucts=True, circular=False):
    LvCD = 0
    HvCD = 0
    if PutCoolingDucts:
        LvCD, HvCD = CalculateNumberOfCoolingDucts(LVNumberOfTurns, LVFoilHeight, LVFoilThickness, HVWireThickness, HVWireLength, CoreDiameter, CoreLength, LVRATE=LVRATE, HVRATE=HVRATE, POWER=POWERRATING, FREQ=FREQUENCY, MaterialResistivity=materialToBeUsedWire_Resistivity, GUARANTEEDNLL=GUARANTEED_NO_LOAD_LOSS, GUARANTEEDLL=GUARANTEED_LOAD_LOSS, GUARANTEEDUCC=GUARANTEED_UCC, tolerance=1, isFinal=isFinal, circular=circular)
    Nll = CalculateNoLoadLosses(LVNumberOfTurns, LVFoilHeight, LVFoilThickness, HVWireThickness, HVWireLength, CoreDiameter, LVRATE, CoreLength, LvCD, HvCD, circular)
    Ll, LlHv, LlLv = CalculateLoadLosses(LVNumberOfTurns, LVFoilThickness, CoreDiameter, LVFoilHeight, HVWireThickness, HVWireLength, MaterialResistivity, POWER, HVRATE, LVRATE, CoreLength, LvCD, HvCD, circular)
    strayDia = CalculateStrayDiameter(LVNumberOfTurns, LVFoilThickness, LVFoilHeight, HVWireThickness, HVWireLength, CoreDiameter, CoreLength, LvCD, HvCD, circular)
    Ux = CalculateUx(POWER, strayDia, LVNumberOfTurns, LVFoilThickness, LVFoilHeight, HVWireThickness, HVWireLength, FREQ, LVRATE, LvCD, HvCD, circular)
    Ur = CalculateUr(Ll, POWER)
    Ucc = CalculateImpedance(Ux, Ur)
    price = CalculatePrice(LVNumberOfTurns, LVFoilHeight, LVFoilThickness, HVWireThickness, HVWireLength, CoreDiameter, CoreLength, LvCD, HvCD, isFinal, circular)
    penaltyForNll = max(0, Nll - GUARANTEEDNLL) * PENALTY_NLL_FACTOR
    penaltyForLL = max(0, Ll - GUARANTEEDLL) * PENALTY_LL_FACTOR
    penaltyforUcc = max(0, abs(Ucc - GUARANTEED_UCC) - abs(UCC_TOLERANCE)) * PENALTY_UCC_FACTOR
    if isFinal:
        print("#######################################################")
        print("BARE PRICE = ", price)
        print("NLL penalty = ", penaltyForNll)
        print("LL penalty = ", penaltyForLL)
        print("UCC Penalty = ", penaltyforUcc)
        print("No load losses ", Nll)
        print("Load losses = ", Ll)
        print("Ucc  = ", Ucc)
        print("Cooling Ducts LV = ", LvCD)
        print("Cooling Ducts HV = ", HvCD)
        print("Price Is = ", price + penaltyForNll + penaltyForLL + penaltyforUcc)
    return price + penaltyForNll + penaltyForLL + penaltyforUcc


@njit(fastmath=True)
def CalculateFinalizedPriceIntolerant(LVNumberOfTurns, LVFoilHeight, LVFoilThickness, HVWireThickness, HVWireLength, CoreDiameter, CoreLength, LVRATE=LVRATE, HVRATE=HVRATE, POWER=POWERRATING, FREQ=FREQUENCY, MaterialResistivity=materialToBeUsedWire_Resistivity, GUARANTEEDNLL=GUARANTEED_NO_LOAD_LOSS, GUARANTEEDLL=GUARANTEED_LOAD_LOSS, GUARANTEEDUCC=GUARANTEED_UCC, tolerance=1, isFinal=False, PutCoolingDucts=True, circular=False):
    LvCD = 0
    HvCD = 0
    if PutCoolingDucts:
        LvCD, HvCD = CalculateNumberOfCoolingDucts(LVNumberOfTurns, LVFoilHeight, LVFoilThickness, HVWireThickness, HVWireLength, CoreDiameter, CoreLength, LVRATE=LVRATE, HVRATE=HVRATE, POWER=POWERRATING, FREQ=FREQUENCY, MaterialResistivity=materialToBeUsedWire_Resistivity, GUARANTEEDNLL=GUARANTEED_NO_LOAD_LOSS, GUARANTEEDLL=GUARANTEED_LOAD_LOSS, GUARANTEEDUCC=GUARANTEED_UCC, tolerance=1, isFinal=False, circular=circular)
    Nll = CalculateNoLoadLosses(LVNumberOfTurns, LVFoilHeight, LVFoilThickness, HVWireThickness, HVWireLength, CoreDiameter, LVRATE, CoreLength, LvCD, HvCD, circular)
    Ll, LlHv, LlLv = CalculateLoadLosses(LVNumberOfTurns, LVFoilThickness, CoreDiameter, LVFoilHeight, HVWireThickness, HVWireLength, MaterialResistivity, POWER, HVRATE, LVRATE, CoreLength, LvCD, HvCD, circular)
    strayDia = CalculateStrayDiameter(LVNumberOfTurns, LVFoilThickness, LVFoilHeight, HVWireThickness, HVWireLength, CoreDiameter, CoreLength, LvCD, HvCD, circular)
    Ux = CalculateUx(POWER, strayDia, LVNumberOfTurns, LVFoilThickness, LVFoilHeight, HVWireThickness, HVWireLength, FREQ, LVRATE, LvCD, HvCD, circular)
    Ur = CalculateUr(Ll, POWER)
    Ucc = CalculateImpedance(Ux, Ur)
    price = CalculatePrice(LVNumberOfTurns, LVFoilHeight, LVFoilThickness, HVWireThickness, HVWireLength, CoreDiameter, CoreLength, LvCD, HvCD, printValues=isFinal, circular=circular)
    NllExtraLoss = max(0, Nll - GUARANTEEDNLL)
    LlExtraLoss = max(0, Ll - GUARANTEEDLL)
    UccExtraLoss = max(0, abs(Ucc - GUARANTEEDUCC) - abs(UCC_TOLERANCE))
    penaltyForNll = NllExtraLoss * PENALTY_NLL_FACTOR
    penaltyForLL = LlExtraLoss * PENALTY_LL_FACTOR
    penaltyforUcc = UccExtraLoss * PENALTY_UCC_FACTOR
    total_price = price + penaltyForNll + penaltyForLL + penaltyforUcc
    # Mark as invalid (1e18) if exceeds tolerance - consistent with GPU implementation
    if (NllExtraLoss > (GUARANTEEDNLL * tolerance / 100)) or (LlExtraLoss > (GUARANTEEDLL * tolerance / 100)) or (UccExtraLoss > (GUARANTEEDUCC * tolerance / 100)):
        total_price = 1e18
    if isFinal:
        print("#######################################################")
        print("BARE PRICE = ", price)
        print("NLL penalty = ", penaltyForNll)
        print("LL penalty = ", penaltyForLL)
        print("UCC Penalty = ", penaltyforUcc)
        print("No load losses ", Nll)
        print("Load losses = ", Ll)
        print("Ucc  = ", Ucc)
        print("Price Is = ", price + penaltyForNll + penaltyForLL + penaltyforUcc)
    return total_price


@njit(fastmath=True)
def CalculateFinalizedPriceIntolerant_Optimized(LVNumberOfTurns, LVFoilHeight, LVFoilThickness, HVWireThickness, HVWireLength, CoreDiameter, CoreLength, LVRATE_VAL=LVRATE, HVRATE_VAL=HVRATE, POWER=POWERRATING, FREQ=FREQUENCY, MaterialResistivity=materialToBeUsedWire_Resistivity, GUARANTEEDNLL=GUARANTEED_NO_LOAD_LOSS, GUARANTEEDLL=GUARANTEED_LOAD_LOSS, GUARANTEEDUCC=GUARANTEED_UCC, tolerance=1, isFinal=False, PutCoolingDucts=True, circular=False):
    """Optimized version that calculates load losses only once when possible."""
    LvCD = 0.0
    HvCD = 0.0
    if PutCoolingDucts:
        LvCD, HvCD, Ll_zero, LlLv_zero, LlHv_zero = CalculateNumberOfCoolingDucts_WithLosses(
            LVNumberOfTurns, LVFoilHeight, LVFoilThickness, HVWireThickness, HVWireLength,
            CoreDiameter, CoreLength, MaterialResistivity, POWER, HVRATE_VAL, LVRATE_VAL, isFinal, circular
        )
        if LvCD == 0 and HvCD == 0:
            Ll, LlHv, LlLv = Ll_zero, LlHv_zero, LlLv_zero
        else:
            Ll, LlHv, LlLv = CalculateLoadLosses(LVNumberOfTurns, LVFoilThickness, CoreDiameter, LVFoilHeight, HVWireThickness, HVWireLength, MaterialResistivity, POWER, HVRATE_VAL, LVRATE_VAL, CoreLength, LvCD, HvCD, circular)
    else:
        Ll, LlHv, LlLv = CalculateLoadLosses(LVNumberOfTurns, LVFoilThickness, CoreDiameter, LVFoilHeight, HVWireThickness, HVWireLength, MaterialResistivity, POWER, HVRATE_VAL, LVRATE_VAL, CoreLength, 0, 0, circular)
    Nll = CalculateNoLoadLosses(LVNumberOfTurns, LVFoilHeight, LVFoilThickness, HVWireThickness, HVWireLength, CoreDiameter, LVRATE_VAL, CoreLength, LvCD, HvCD, circular)
    strayDia = CalculateStrayDiameter(LVNumberOfTurns, LVFoilThickness, LVFoilHeight, HVWireThickness, HVWireLength, CoreDiameter, CoreLength, LvCD, HvCD, circular)
    Ux = CalculateUx(POWER, strayDia, LVNumberOfTurns, LVFoilThickness, LVFoilHeight, HVWireThickness, HVWireLength, FREQ, LVRATE_VAL, LvCD, HvCD, circular)
    Ur = CalculateUr(Ll, POWER)
    Ucc = CalculateImpedance(Ux, Ur)
    price = CalculatePrice(LVNumberOfTurns, LVFoilHeight, LVFoilThickness, HVWireThickness, HVWireLength, CoreDiameter, CoreLength, LvCD, HvCD, isFinal, circular)
    NllExtraLoss = max(0, Nll - GUARANTEEDNLL)
    LlExtraLoss = max(0, Ll - GUARANTEEDLL)
    UccExtraLoss = max(0, abs(Ucc - GUARANTEEDUCC) - abs(UCC_TOLERANCE))
    penaltyForNll = NllExtraLoss * PENALTY_NLL_FACTOR
    penaltyForLL = LlExtraLoss * PENALTY_LL_FACTOR
    penaltyforUcc = UccExtraLoss * PENALTY_UCC_FACTOR
    total_price = price + penaltyForNll + penaltyForLL + penaltyforUcc
    # Mark as invalid (1e18) if exceeds tolerance - consistent with GPU implementation
    if (NllExtraLoss > (GUARANTEEDNLL * tolerance / 100)) or (LlExtraLoss > (GUARANTEEDLL * tolerance / 100)) or (UccExtraLoss > (GUARANTEEDUCC * tolerance / 100)):
        total_price = 1e18
    if isFinal:
        print("#######################################################")
        print("BARE PRICE = ", price)
        print("NLL penalty = ", penaltyForNll)
        print("LL penalty = ", penaltyForLL)
        print("UCC Penalty = ", penaltyforUcc)
        print("No load losses ", Nll)
        print("Load losses = ", Ll)
        print("Ucc  = ", Ucc)
        print("Cooling Ducts LV = ", LvCD)
        print("Cooling Ducts HV = ", HvCD)
        print("Price Is = ", (price + penaltyForNll + penaltyForLL + penaltyforUcc))
    return total_price


# =============================================================================
# GRID SEARCH OPTIMIZATION
# =============================================================================

@njit(fastmath=True)
def BucketFillingSmart(TurnsStep=1, ThicknessStep=0.2, HeightStep=50, CoreDiaStep=30, CoreLengthStep=1, HVWireThicknessStep=0.2, HVLengthStep=10, TurnsStepMinimum=1, FoilHeightStepMinimum=5, FoilThicknessStepMinimum=0.05, CoreDiameterStepMinimum=1, CoreLengthStepMinimum=1, HVThicknessStepMinimum=0.05, HVLengthMinimumStep=0.1, BrakeDistance=5, tolerance=1, printValuesProc=False, printValuesFinal=False, obround=True, PutCoolingDuct=True):
    """Original grid search optimization algorithm."""
    if obround:
        CoreDiameterStepMinimum = 10

    core_length_t = CORELENGTH_MINIMUM
    lv_turns_t = FOILTURNS_MINIMUM
    lv_thickness_t = FOILTHICKNESS_MINIMUM
    lv_foilheight_t = FOILHEIGHT_MINIMUM
    core_diameter_t = CORE_MINIMUM
    hv_wire_thickness_t = HVTHICK_MINIMUM
    hv_wire_length_t = hv_wire_thickness_t
    value = -1
    iteration = 0
    loopEnded = False

    # Phase 1: Coarse search
    while (core_diameter_t < CORE_MAXIMUM) and (not loopEnded):
        if printValuesProc:
            print("Core Diameter is:", int(core_diameter_t))
        lv_thickness_t = FOILTHICKNESS_MINIMUM
        lv_foilheight_t = FOILHEIGHT_MINIMUM
        lv_turns_t = FOILTURNS_MINIMUM
        hv_wire_thickness_t = HVTHICK_MINIMUM
        core_length_t = CORELENGTH_MINIMUM
        CoreLengthStep = core_diameter_t / 5
        obroundStackMax = core_diameter_t if obround else 0

        while (core_length_t <= obroundStackMax) and (not loopEnded):
            if printValuesProc:
                print("Obround Stack is:", int(core_length_t))
            lv_thickness_t = FOILTHICKNESS_MINIMUM
            lv_foilheight_t = FOILHEIGHT_MINIMUM
            lv_turns_t = FOILTURNS_MINIMUM
            hv_wire_thickness_t = HVTHICK_MINIMUM

            while (lv_turns_t < FOILTURNS_MAXIMUM) and (not loopEnded):
                lv_foilheight_t = FOILHEIGHT_MINIMUM
                lv_thickness_t = FOILTHICKNESS_MINIMUM
                hv_wire_thickness_t = HVTHICK_MINIMUM

                while (lv_foilheight_t < FOILHEIGHT_MAXIMUM) and (not loopEnded):
                    lv_thickness_t = FOILTHICKNESS_MINIMUM
                    hv_wire_thickness_t = HVTHICK_MINIMUM
                    while (lv_thickness_t < FOILTHICKNESS_MAXIMUM) and (not loopEnded):
                        hv_wire_thickness_t = HVTHICK_MINIMUM
                        while (hv_wire_thickness_t < HVTHICK_MAXIMUM) and (not loopEnded):
                            hv_wire_length_t = hv_wire_thickness_t
                            while (hv_wire_length_t <= HV_LEN_MAXIMUM) and (not loopEnded):
                                value = CalculateFinalizedPriceIntolerant(lv_turns_t, lv_foilheight_t, lv_thickness_t, hv_wire_thickness_t, hv_wire_length_t, core_diameter_t, tolerance=tolerance, CoreLength=core_length_t, PutCoolingDucts=PutCoolingDuct)
                                iteration += 1
                                if value > 0:
                                    if printValuesProc:
                                        print("################################################")
                                        print("First Loop Is Completed With Values")
                                        print("PRICE = ", value)
                                        print("LV TURNS IS = ", lv_turns_t)
                                        print("LV THICKNESS IS = ", lv_thickness_t)
                                        print("LV HEIGHT IS = ", lv_foilheight_t)
                                        print("CORE DIAMETER IS = ", core_diameter_t)
                                        print("HV THICKNESS IS = ", hv_wire_thickness_t)
                                        print("################################################")
                                    loopEnded = True
                                hv_wire_length_t += HVLengthStep
                            hv_wire_thickness_t += HVWireThicknessStep
                        lv_thickness_t += ThicknessStep
                    lv_foilheight_t += HeightStep
                lv_turns_t += TurnsStep
            core_length_t += CoreLengthStep
        core_diameter_t += CoreDiaStep

    # Phase 2: Fine search
    coreLengthNewStartPoint = CORELENGTH_MINIMUM
    coreLengthNewEndPoint = 0
    coreDiameterNewStartPoint = max(core_diameter_t - CoreDiaStep * 2, CORE_MINIMUM)
    foilTurnsNewStartPoint = max(lv_turns_t - 30 * TurnsStep, FOILTURNS_MINIMUM)
    foilHeightNewStartPoint = max(lv_foilheight_t - 10 * HeightStep, FOILHEIGHT_MINIMUM)
    foilThicknessNewStartPoint = max(lv_thickness_t - 6 * ThicknessStep, FOILTHICKNESS_MINIMUM)
    hvWireThicknessNewStartPoint = max(hv_wire_thickness_t - 6 * HVWireThicknessStep, HVTHICK_MINIMUM)
    coreDiameterNewEndPoint = CORE_MAXIMUM
    foilTurnsNewEndPoint = min(lv_turns_t + TurnsStep, FOILTURNS_MAXIMUM)
    foilHeightNewEndPoint = min(lv_foilheight_t + HeightStep, FOILHEIGHT_MAXIMUM)
    foilThicknessNewEndPoint = min(lv_thickness_t + ThicknessStep, FOILTHICKNESS_MAXIMUM)
    hvWireThicknessNewEndPoint = min(hv_wire_thickness_t + HVWireThicknessStep, HVTHICK_MAXIMUM)
    loopEnded = False
    CountDown = BrakeDistance + 1
    HVLengthMinimumStep = hvWireThicknessNewStartPoint / 2
    hvWireLengthNewEndPoint = hvWireThicknessNewEndPoint * 5

    foundPrice = 999999999999999999
    foundCoreDiameter = 0
    foundHVThickness = 0
    foundLVTurns = 0
    foundLVHeight = 0
    foundLVThickness = 0
    foundCoreLength = 0
    foundHVLength = 0

    core_diameter_t = coreDiameterNewStartPoint
    while (core_diameter_t < coreDiameterNewEndPoint) and (CountDown > 0):
        if printValuesProc:
            print("Core Diameter is:", int(core_diameter_t))
        lv_thickness_t = foilThicknessNewStartPoint
        lv_foilheight_t = foilHeightNewStartPoint
        lv_turns_t = foilTurnsNewStartPoint
        hv_wire_thickness_t = hvWireThicknessNewStartPoint
        core_length_t = coreLengthNewStartPoint
        CoreLengthStepMinimum = core_diameter_t / 10
        if obround:
            coreLengthNewEndPoint = core_diameter_t

        while core_length_t <= coreLengthNewEndPoint:
            lv_thickness_t = foilThicknessNewStartPoint
            lv_foilheight_t = foilHeightNewStartPoint
            lv_turns_t = foilTurnsNewStartPoint
            hv_wire_thickness_t = hvWireThicknessNewStartPoint

            while lv_turns_t < foilTurnsNewEndPoint:
                lv_foilheight_t = foilHeightNewStartPoint
                lv_thickness_t = foilThicknessNewStartPoint
                hv_wire_thickness_t = hvWireThicknessNewStartPoint

                while lv_foilheight_t < foilHeightNewEndPoint:
                    lv_thickness_t = foilThicknessNewStartPoint
                    hv_wire_thickness_t = hvWireThicknessNewStartPoint

                    while lv_thickness_t < foilThicknessNewEndPoint:
                        hv_wire_thickness_t = hvWireThicknessNewStartPoint

                        while hv_wire_thickness_t < hvWireThicknessNewEndPoint:
                            hv_wire_length_t = 2 * hv_wire_thickness_t
                            HVLengthMinimumStep = (hv_wire_thickness_t / 2) - ((hv_wire_thickness_t / 2) % 0.1)
                            hvWireLengthNewEndPoint = min(hv_wire_thickness_t * 6, HV_LEN_MAXIMUM)
                            while hv_wire_length_t <= hvWireLengthNewEndPoint:
                                value = CalculateFinalizedPriceIntolerant(lv_turns_t, lv_foilheight_t, lv_thickness_t, hv_wire_thickness_t, hv_wire_length_t, core_diameter_t, CoreLength=core_length_t, tolerance=tolerance, PutCoolingDucts=PutCoolingDuct)
                                iteration += 1
                                if value > 0:
                                    if not loopEnded:
                                        if printValuesProc:
                                            print("FIRST TRANSFORMER FOUND SOON THE PROCESS WILL FINISH")
                                    if value < foundPrice:
                                        foundPrice = value
                                        foundCoreDiameter = core_diameter_t
                                        foundHVThickness = hv_wire_thickness_t
                                        foundLVHeight = lv_foilheight_t
                                        foundLVThickness = lv_thickness_t
                                        foundLVTurns = lv_turns_t
                                        foundCoreLength = core_length_t
                                        foundHVLength = hv_wire_length_t
                                    loopEnded = True
                                hv_wire_length_t += HVLengthMinimumStep
                            hv_wire_thickness_t += HVThicknessStepMinimum
                        lv_thickness_t += FoilThicknessStepMinimum
                    lv_foilheight_t += FoilHeightStepMinimum
                lv_turns_t += TurnsStepMinimum
            core_length_t += CoreLengthStepMinimum
        if loopEnded:
            CountDown = CountDown - 1
            if printValuesProc:
                print(CountDown)
        core_diameter_t += CoreDiameterStepMinimum

    if printValuesFinal:
        CalculateFinalizedPrice(foundLVTurns, foundLVHeight, foundLVThickness, foundHVThickness, foundHVLength, foundCoreDiameter, foundCoreLength, LVRATE, HVRATE, POWERRATING, FREQUENCY, materialToBeUsedWire_Resistivity, GUARANTEED_NO_LOAD_LOSS, GUARANTEED_LOAD_LOSS, GUARANTEED_UCC, isFinal=True, PutCoolingDucts=PutCoolingDuct)
        print("####################################################")
        print("Core Diameter = ", foundCoreDiameter)
        print("Core Length = ", foundCoreLength)
        print("HV Diameter = ", foundHVThickness)
        print("Foil Height = ", foundLVHeight)
        print("Foil Turns = ", foundLVTurns)
        print("Foil Thickness = ", foundLVThickness)
        print("THE PRICE IS =", foundPrice)
        print("THE LENGTH HV IS = ", foundHVLength)
        print("####################################################")

    return (foundPrice, foundCoreLength, foundHVThickness, foundLVHeight, foundLVTurns, foundLVThickness)


# =============================================================================
# PARALLEL GRID SEARCH (Numba Optimized)
# =============================================================================

@njit(fastmath=True, parallel=True)
def parallel_grid_search_kernel(core_dia_arr, core_len_arr, turns_arr, height_arr, thick_arr, hvthick_arr, hvlen_arr,
                                 tolerance, obround, put_cooling_ducts, circular,
                                 LVRATE_VAL, HVRATE_VAL, POWER, FREQ, MaterialResistivity,
                                 GUARANTEEDNLL, GUARANTEEDLL, GUARANTEEDUCC,
                                 CoreFillRound, CoreFillRect, InsulationThickness):
    """Parallel grid search kernel using Numba prange.

    Uses CalculateFinalizedPriceIntolerant_Optimized for accurate penalty calculations
    including cooling ducts.

    All module-level parameters are passed explicitly to avoid JIT compilation capture issues.
    """
    n_core = len(core_dia_arr)
    n_turns = len(turns_arr)
    n_height = len(height_arr)
    n_thick = len(thick_arr)
    n_hvthick = len(hvthick_arr)
    n_hvlen = len(hvlen_arr)

    results = np.full((n_core, 9), 1e18, dtype=np.float64)

    for i in prange(n_core):
        core_dia = core_dia_arr[i]
        local_best_price = 1e18
        best_turns = 0.0
        best_height = 0.0
        best_thick = 0.0
        best_hvthick = 0.0
        best_hvlen = 0.0
        best_core_len = 0.0
        found_valid = 0.0

        core_len_max = core_dia if obround else 0.0
        core_len_step = max(core_dia / 10.0, 1.0) if obround else 1.0
        core_len = 0.0

        while core_len <= core_len_max:
            # Quick induction pre-filter - use passed parameters
            section_round = ((core_dia**2) * math.pi / 400.0) * CoreFillRound
            section_rect = (core_len * core_dia / 100.0) * CoreFillRect
            core_section = section_round + section_rect

            for j in range(n_turns):
                turns = turns_arr[j]
                volts_per_turn = LVRATE_VAL / turns
                induction = (volts_per_turn * 10000) / (math.sqrt(2) * math.pi * FREQ * core_section)

                # Skip invalid induction ranges
                if induction > 1.95 or induction < 0.8:
                    continue

                for k in range(n_height):
                    height = height_arr[k]

                    for l in range(n_thick):
                        thick = thick_arr[l]

                        for m in range(n_hvthick):
                            hvthick = hvthick_arr[m]

                            for n in range(n_hvlen):
                                hvlen = hvlen_arr[n]

                                # Basic validity checks (skip for circular wire)
                                if not circular and hvlen < hvthick:
                                    continue

                                hv_layer_height = height - 50
                                # For circular wire, axial space per turn = diameter (hvthick)
                                wire_axial_size = hvthick if circular else hvlen
                                hv_turns_per_layer = (hv_layer_height / (wire_axial_size + InsulationThickness)) - 1
                                if hv_turns_per_layer <= 0:
                                    continue

                                # Use full calculation with cooling ducts and penalties
                                price = CalculateFinalizedPriceIntolerant_Optimized(
                                    turns, height, thick, hvthick, hvlen,
                                    core_dia, core_len,
                                    LVRATE_VAL, HVRATE_VAL, POWER, FREQ,
                                    MaterialResistivity,
                                    GUARANTEEDNLL, GUARANTEEDLL, GUARANTEEDUCC,
                                    tolerance, False, put_cooling_ducts, circular
                                )

                                # Positive price = valid design within tolerance
                                if price > 0 and price < local_best_price:
                                    local_best_price = price
                                    best_turns = turns
                                    best_height = height
                                    best_thick = thick
                                    best_hvthick = hvthick
                                    best_hvlen = hvlen
                                    best_core_len = core_len
                                    found_valid = 1.0

            if obround:
                core_len += core_len_step
            else:
                break

        results[i, 0] = local_best_price
        results[i, 1] = best_turns
        results[i, 2] = best_height
        results[i, 3] = best_thick
        results[i, 4] = best_hvthick
        results[i, 5] = best_hvlen
        results[i, 6] = core_dia
        results[i, 7] = best_core_len
        results[i, 8] = found_valid

    return results


# =============================================================================
# PARALLEL LOCAL REFINEMENT KERNEL (for Stage 3)
# =============================================================================

@njit(fastmath=True, parallel=True)
def parallel_local_refinement_kernel(
    turns_arr, height_arr, thick_arr, hvthick_arr, hvlen_arr,
    core_dia_arr, core_len_arr,
    LVRATE_VAL, HVRATE_VAL, POWER, FREQ, MaterialResistivity,
    GUARANTEEDNLL, GUARANTEEDLL, GUARANTEEDUCC,
    tolerance, put_cooling_ducts, circular
):
    """
    Parallel kernel for Stage 3 local refinement.

    Takes flattened arrays of all parameter combinations and evaluates them in parallel.
    Returns array of [price, turns, height, thick, hvthick, hvlen, core_dia, core_len] for each combo.
    """
    n_combinations = len(turns_arr)

    # Results: each row = [price, turns, height, thick, hvthick, hvlen, core_dia, core_len]
    results = np.empty((n_combinations, 8), dtype=np.float64)

    for i in prange(n_combinations):
        turns = turns_arr[i]
        height = height_arr[i]
        thick = thick_arr[i]
        hvthick = hvthick_arr[i]
        hvlen = hvlen_arr[i]
        core_dia = core_dia_arr[i]
        core_len = core_len_arr[i]

        # Skip invalid combinations (for rectangular wire, hvlen must be >= hvthick)
        # For circular wire, this check is skipped
        if not circular and hvlen < hvthick:
            results[i, 0] = 1e18
            results[i, 1] = turns
            results[i, 2] = height
            results[i, 3] = thick
            results[i, 4] = hvthick
            results[i, 5] = hvlen
            results[i, 6] = core_dia
            results[i, 7] = core_len
            continue

        # Core length must not exceed core diameter
        if core_len > core_dia:
            results[i, 0] = 1e18
            results[i, 1] = turns
            results[i, 2] = height
            results[i, 3] = thick
            results[i, 4] = hvthick
            results[i, 5] = hvlen
            results[i, 6] = core_dia
            results[i, 7] = core_len
            continue

        # Full evaluation WITH cooling ducts
        price = CalculateFinalizedPriceIntolerant_Optimized(
            turns, height, thick, hvthick, hvlen,
            core_dia, core_len,
            LVRATE_VAL, HVRATE_VAL, POWER, FREQ,
            MaterialResistivity,
            GUARANTEEDNLL, GUARANTEEDLL, GUARANTEEDUCC,
            tolerance, False, put_cooling_ducts, circular
        )

        results[i, 0] = price
        results[i, 1] = turns
        results[i, 2] = height
        results[i, 3] = thick
        results[i, 4] = hvthick
        results[i, 5] = hvlen
        results[i, 6] = core_dia
        results[i, 7] = core_len

    return results


def parallel_local_refinement_with_progress(
    turns_arr, height_arr, thick_arr, hvthick_arr, hvlen_arr,
    core_dia_arr, core_len_arr,
    LVRATE_VAL, HVRATE_VAL, POWER, FREQ, MaterialResistivity,
    GUARANTEEDNLL, GUARANTEEDLL, GUARANTEEDUCC,
    tolerance, put_cooling_ducts, circular,
    progress_callback=None, batch_size=2000
):
    """
    Wrapper for parallel_local_refinement_kernel with progress reporting.

    Processes in batches to allow progress updates between batches.
    Uses smaller batch_size (2000) for more frequent progress updates.
    """
    import time
    n_total = len(turns_arr)

    if n_total == 0:
        return np.empty((0, 8), dtype=np.float64)

    # Always process in batches for progress updates (even for small datasets)
    start_time = time.time()

    # Use smaller batches for small datasets to get at least a few progress updates
    effective_batch_size = min(batch_size, max(500, n_total // 5))
    n_batches = (n_total + effective_batch_size - 1) // effective_batch_size
    all_results = []

    # Initial progress report with estimated ETA based on typical rate
    # Estimate ~50,000-100,000 evaluations per second on CPU with parallel processing
    estimated_rate = 75000  # evaluations per second (conservative estimate)
    initial_eta = n_total / estimated_rate if n_total > 0 else None
    if progress_callback:
        try:
            progress_callback(3, 0.68, f"Stage 3: Starting {n_total:,} evaluations...", initial_eta)
        except:
            pass

    for batch_idx in range(n_batches):
        start_idx = batch_idx * effective_batch_size
        end_idx = min(start_idx + effective_batch_size, n_total)

        # Extract batch
        batch_results = parallel_local_refinement_kernel(
            turns_arr[start_idx:end_idx],
            height_arr[start_idx:end_idx],
            thick_arr[start_idx:end_idx],
            hvthick_arr[start_idx:end_idx],
            hvlen_arr[start_idx:end_idx],
            core_dia_arr[start_idx:end_idx],
            core_len_arr[start_idx:end_idx],
            LVRATE_VAL, HVRATE_VAL, POWER, FREQ, MaterialResistivity,
            GUARANTEEDNLL, GUARANTEEDLL, GUARANTEEDUCC,
            tolerance, put_cooling_ducts, circular
        )
        all_results.append(batch_results)

        # Report progress with actual ETA based on elapsed time
        if progress_callback:
            completed = end_idx
            progress_pct = 0.68 + (0.30 * completed / n_total)
            elapsed = time.time() - start_time
            if completed > 0 and completed < n_total:
                eta = (elapsed / completed) * (n_total - completed)
            else:
                eta = None
            try:
                progress_callback(3, progress_pct,
                                f"Stage 3: {completed:,}/{n_total:,} ({100*completed/n_total:.1f}%)", eta)
            except:
                pass

    return np.vstack(all_results)


def generate_local_combinations(turns_range, height_range, thick_range,
                                 hvthick_range, hvlen_range,
                                 core_dia_range, core_len_range):
    """
    Generate flattened arrays of all parameter combinations for parallel processing.

    Returns tuple of (turns_arr, height_arr, thick_arr, hvthick_arr, hvlen_arr,
                      core_dia_arr, core_len_arr) as numpy arrays.

    WARNING: For large combinations (>10M), use generate_local_combinations_chunked instead.
    """
    # Calculate total combinations
    n_total = (len(turns_range) * len(height_range) * len(thick_range) *
               len(hvthick_range) * len(hvlen_range) *
               len(core_dia_range) * len(core_len_range))

    # Pre-allocate arrays
    turns_arr = np.empty(n_total, dtype=np.float64)
    height_arr = np.empty(n_total, dtype=np.float64)
    thick_arr = np.empty(n_total, dtype=np.float64)
    hvthick_arr = np.empty(n_total, dtype=np.float64)
    hvlen_arr = np.empty(n_total, dtype=np.float64)
    core_dia_arr = np.empty(n_total, dtype=np.float64)
    core_len_arr = np.empty(n_total, dtype=np.float64)

    # Fill arrays with all combinations
    idx = 0
    for cd in core_dia_range:
        for cl in core_len_range:
            for t in turns_range:
                for h in height_range:
                    for th in thick_range:
                        for hvth in hvthick_range:
                            for hvl in hvlen_range:
                                turns_arr[idx] = t
                                height_arr[idx] = h
                                thick_arr[idx] = th
                                hvthick_arr[idx] = hvth
                                hvlen_arr[idx] = hvl
                                core_dia_arr[idx] = cd
                                core_len_arr[idx] = cl
                                idx += 1

    return turns_arr, height_arr, thick_arr, hvthick_arr, hvlen_arr, core_dia_arr, core_len_arr


def streaming_local_refinement(
    turns_range, height_range, thick_range, hvthick_range, hvlen_range,
    core_dia_range, core_len_range,
    LVRATE_VAL, HVRATE_VAL, POWER, FREQ, MaterialResistivity,
    GUARANTEEDNLL, GUARANTEEDLL, GUARANTEEDUCC,
    tolerance, put_cooling_ducts, circular,
    progress_callback=None, chunk_size=100000
):
    """
    Streaming local refinement that generates and processes combinations in chunks.

    This avoids memory issues for large search spaces (>10M combinations).
    Generates chunks on-the-fly, processes them, and tracks the best result.
    """
    import time

    # Convert ranges to lists for indexing
    turns_list = list(turns_range)
    height_list = list(height_range)
    thick_list = list(thick_range)
    hvthick_list = list(hvthick_range)
    hvlen_list = list(hvlen_range)
    core_dia_list = list(core_dia_range)
    core_len_list = list(core_len_range)

    # Calculate total combinations
    n_turns = len(turns_list)
    n_height = len(height_list)
    n_thick = len(thick_list)
    n_hvthick = len(hvthick_list)
    n_hvlen = len(hvlen_list)
    n_core_dia = len(core_dia_list)
    n_core_len = len(core_len_list)

    n_total = n_turns * n_height * n_thick * n_hvthick * n_hvlen * n_core_dia * n_core_len

    if n_total == 0:
        return None

    # Initialize best result tracking
    best_price = 1e18
    best_params = None

    start_time = time.time()
    processed = 0

    # Initial progress report
    if progress_callback:
        estimated_rate = 50000  # Conservative estimate
        initial_eta = n_total / estimated_rate
        try:
            progress_callback(3, 0.68, f"Stage 3: Starting {n_total:,} evaluations...", initial_eta)
        except:
            pass

    # Pre-allocate chunk arrays (reused for each chunk)
    actual_chunk_size = min(chunk_size, n_total)
    turns_arr = np.empty(actual_chunk_size, dtype=np.float64)
    height_arr = np.empty(actual_chunk_size, dtype=np.float64)
    thick_arr = np.empty(actual_chunk_size, dtype=np.float64)
    hvthick_arr = np.empty(actual_chunk_size, dtype=np.float64)
    hvlen_arr = np.empty(actual_chunk_size, dtype=np.float64)
    core_dia_arr = np.empty(actual_chunk_size, dtype=np.float64)
    core_len_arr = np.empty(actual_chunk_size, dtype=np.float64)

    # Process combinations in chunks using nested iteration
    chunk_idx = 0
    for cd_idx, cd in enumerate(core_dia_list):
        for cl_idx, cl in enumerate(core_len_list):
            for t_idx, t in enumerate(turns_list):
                for h_idx, h in enumerate(height_list):
                    for th_idx, th in enumerate(thick_list):
                        for hvth_idx, hvth in enumerate(hvthick_list):
                            for hvl_idx, hvl in enumerate(hvlen_list):
                                turns_arr[chunk_idx] = t
                                height_arr[chunk_idx] = h
                                thick_arr[chunk_idx] = th
                                hvthick_arr[chunk_idx] = hvth
                                hvlen_arr[chunk_idx] = hvl
                                core_dia_arr[chunk_idx] = cd
                                core_len_arr[chunk_idx] = cl
                                chunk_idx += 1

                                # Process chunk when full
                                if chunk_idx >= actual_chunk_size:
                                    # Run parallel kernel on this chunk
                                    results = parallel_local_refinement_kernel(
                                        turns_arr[:chunk_idx], height_arr[:chunk_idx],
                                        thick_arr[:chunk_idx], hvthick_arr[:chunk_idx],
                                        hvlen_arr[:chunk_idx], core_dia_arr[:chunk_idx],
                                        core_len_arr[:chunk_idx],
                                        LVRATE_VAL, HVRATE_VAL, POWER, FREQ, MaterialResistivity,
                                        GUARANTEEDNLL, GUARANTEEDLL, GUARANTEEDUCC,
                                        tolerance, put_cooling_ducts, circular
                                    )

                                    # Update best result
                                    valid_mask = (results[:, 0] > 0) & (results[:, 0] < 1e17)
                                    if np.any(valid_mask):
                                        valid_results = results[valid_mask]
                                        chunk_best_idx = np.argmin(valid_results[:, 0])
                                        chunk_best_price = valid_results[chunk_best_idx, 0]
                                        if chunk_best_price < best_price:
                                            best_price = chunk_best_price
                                            best_params = valid_results[chunk_best_idx].copy()

                                    # Update progress
                                    processed += chunk_idx
                                    chunk_idx = 0

                                    if progress_callback:
                                        elapsed = time.time() - start_time
                                        progress_pct = 0.68 + (0.30 * processed / n_total)
                                        if processed > 0 and processed < n_total:
                                            eta = (elapsed / processed) * (n_total - processed)
                                        else:
                                            eta = None
                                        try:
                                            progress_callback(3, progress_pct,
                                                f"Stage 3: {processed:,}/{n_total:,} ({100*processed/n_total:.1f}%) Best: {best_price:.2f}",
                                                eta)
                                        except:
                                            pass

    # Process remaining combinations in final partial chunk
    if chunk_idx > 0:
        results = parallel_local_refinement_kernel(
            turns_arr[:chunk_idx], height_arr[:chunk_idx],
            thick_arr[:chunk_idx], hvthick_arr[:chunk_idx],
            hvlen_arr[:chunk_idx], core_dia_arr[:chunk_idx],
            core_len_arr[:chunk_idx],
            LVRATE_VAL, HVRATE_VAL, POWER, FREQ, MaterialResistivity,
            GUARANTEEDNLL, GUARANTEEDLL, GUARANTEEDUCC,
            tolerance, put_cooling_ducts, circular
        )

        valid_mask = (results[:, 0] > 0) & (results[:, 0] < 1e17)
        if np.any(valid_mask):
            valid_results = results[valid_mask]
            chunk_best_idx = np.argmin(valid_results[:, 0])
            chunk_best_price = valid_results[chunk_best_idx, 0]
            if chunk_best_price < best_price:
                best_price = chunk_best_price
                best_params = valid_results[chunk_best_idx].copy()

        processed += chunk_idx

    # Final progress report
    if progress_callback:
        try:
            progress_callback(3, 0.98, f"Stage 3: Completed {processed:,} evaluations. Best: {best_price:.2f}", None)
        except:
            pass

    return best_params


def local_optimizer_polish(
    initial_params,
    tolerance, put_cooling_ducts,
    progress_callback=None
):
    """
    Polish Stage 3 result using L-BFGS-B local optimizer.

    Refines the coarse grid result to find true local minimum
    with high precision. Uses scipy's bounded L-BFGS-B method.

    Args:
        initial_params: tuple/list of (turns, height, thick, hvthick, hvlen, core_dia, core_len)
        tolerance: Tolerance percentage for constraints
        put_cooling_ducts: Whether to include cooling duct calculations
        progress_callback: Optional progress callback function

    Returns:
        tuple: (optimized_params_array, final_price)
    """
    if not SCIPY_AVAILABLE:
        print("  scipy not available, skipping local optimizer polish")
        return initial_params, 1e18

    from scipy.optimize import minimize
    import time

    start_time = time.time()
    eval_count = [0]  # Use list for mutable in closure

    def objective(x):
        eval_count[0] += 1
        price = CalculateFinalizedPriceIntolerant_Optimized(
            int(round(x[0])), x[1], x[2], x[3], x[4], x[5], x[6],
            LVRATE, HVRATE, POWERRATING, FREQUENCY,
            materialToBeUsedWire_Resistivity,
            GUARANTEED_NO_LOAD_LOSS, GUARANTEED_LOAD_LOSS, GUARANTEED_UCC,
            tolerance, False, put_cooling_ducts, circular=HV_WIRE_CIRCULAR
        )
        # Handle invalid prices
        if price < 0:
            return abs(price) + 1e6  # Penalize invalid but don't return inf
        return price

    # Convert to list if needed
    x0 = list(initial_params)

    # Bounds around initial point (reasonable local search range)
    bounds = [
        (max(FOILTURNS_MINIMUM, x0[0] - 3), min(FOILTURNS_MAXIMUM, x0[0] + 3)),
        (max(FOILHEIGHT_MINIMUM, x0[1] - 30), min(FOILHEIGHT_MAXIMUM, x0[1] + 30)),
        (max(FOILTHICKNESS_MINIMUM, x0[2] - 0.15), min(FOILTHICKNESS_MAXIMUM, x0[2] + 0.15)),
        (max(HVTHICK_MINIMUM, x0[3] - 0.15), min(HVTHICK_MAXIMUM, x0[3] + 0.15)),
        (max(HV_LEN_MINIMUM, x0[4] - 0.5), min(HV_LEN_MAXIMUM, x0[4] + 0.5)),
        (max(CORE_MINIMUM, x0[5] - 15), min(CORE_MAXIMUM, x0[5] + 15)),
        (max(0, x0[6] - 15), min(x0[5] + 15, x0[6] + 15)),  # core_len bounded by core_dia
    ]

    try:
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': 200,
                'ftol': 1e-8,
                'gtol': 1e-6,
                'disp': False
            }
        )

        elapsed = time.time() - start_time
        print(f"  Local optimizer: {eval_count[0]} evaluations in {elapsed:.2f}s")

        if result.success or result.fun < objective(x0):
            return result.x, result.fun
        else:
            print(f"  Optimizer did not improve (status: {result.message})")
            return np.array(x0), objective(x0)

    except Exception as e:
        print(f"  Local optimizer failed: {e}")
        return np.array(x0), objective(x0)


def basin_hopping_polish(
    initial_params,
    tolerance, put_cooling_ducts,
    progress_callback=None
):
    """
    Global search with local refinement using basin hopping.

    Escapes local minima by combining random jumps with L-BFGS-B refinement.
    Good for finding better basins than pure local search.

    Args:
        initial_params: tuple/list of (turns, height, thick, hvthick, hvlen, core_dia, core_len)
        tolerance: Tolerance percentage for constraints
        put_cooling_ducts: Whether to include cooling duct calculations
        progress_callback: Optional progress callback function

    Returns:
        tuple: (optimized_params_array, final_price)
    """
    if not SCIPY_AVAILABLE:
        print("  scipy not available, skipping basin hopping")
        return np.array(initial_params), 1e18

    from scipy.optimize import basinhopping
    import time

    start_time = time.time()
    eval_count = [0]

    def objective(x):
        eval_count[0] += 1
        price = CalculateFinalizedPriceIntolerant_Optimized(
            int(round(x[0])), x[1], x[2], x[3], x[4], x[5], x[6],
            LVRATE, HVRATE, POWERRATING, FREQUENCY,
            materialToBeUsedWire_Resistivity,
            GUARANTEED_NO_LOAD_LOSS, GUARANTEED_LOAD_LOSS, GUARANTEED_UCC,
            tolerance, False, put_cooling_ducts, circular=HV_WIRE_CIRCULAR
        )
        if price < 0:
            return abs(price) + 1e6
        return price

    x0 = list(initial_params)

    # Bounds around initial point
    bounds = [
        (max(FOILTURNS_MINIMUM, x0[0] - 5), min(FOILTURNS_MAXIMUM, x0[0] + 5)),
        (max(FOILHEIGHT_MINIMUM, x0[1] - 50), min(FOILHEIGHT_MAXIMUM, x0[1] + 50)),
        (max(FOILTHICKNESS_MINIMUM, x0[2] - 0.3), min(FOILTHICKNESS_MAXIMUM, x0[2] + 0.3)),
        (max(HVTHICK_MINIMUM, x0[3] - 0.3), min(HVTHICK_MAXIMUM, x0[3] + 0.3)),
        (max(HV_LEN_MINIMUM, x0[4] - 1.0), min(HV_LEN_MAXIMUM, x0[4] + 1.0)),
        (max(CORE_MINIMUM, x0[5] - 25), min(CORE_MAXIMUM, x0[5] + 25)),
        (max(0, x0[6] - 25), min(x0[5] + 25, x0[6] + 25)),
    ]

    # Custom step taker that respects bounds
    class BoundedStep:
        def __init__(self, stepsize=0.5):
            self.stepsize = stepsize

        def __call__(self, x):
            # Random step scaled by parameter ranges
            x_new = x.copy()
            for i, (lo, hi) in enumerate(bounds):
                step = np.random.uniform(-1, 1) * self.stepsize * (hi - lo) * 0.1
                x_new[i] = np.clip(x[i] + step, lo, hi)
            return x_new

    # Local minimizer settings
    minimizer_kwargs = {
        'method': 'L-BFGS-B',
        'bounds': bounds,
        'options': {'maxiter': 50, 'ftol': 1e-6}
    }

    try:
        result = basinhopping(
            objective,
            x0,
            minimizer_kwargs=minimizer_kwargs,
            niter=15,  # Number of basin hops
            T=50.0,    # Temperature for acceptance
            stepsize=0.5,
            take_step=BoundedStep(0.5),
            seed=42
        )

        elapsed = time.time() - start_time
        print(f"  Basin hopping: {eval_count[0]} evaluations in {elapsed:.2f}s, best={result.fun:.2f}")

        return result.x, result.fun

    except Exception as e:
        print(f"  Basin hopping failed: {e}")
        return np.array(x0), objective(x0)


def dual_annealing_polish(
    initial_params,
    tolerance, put_cooling_ducts,
    progress_callback=None
):
    """
    Handle discontinuities with dual annealing.

    Combines simulated annealing (global) with local search.
    Excellent for functions with discontinuities like cooling duct count changes.

    Args:
        initial_params: tuple/list of (turns, height, thick, hvthick, hvlen, core_dia, core_len)
        tolerance: Tolerance percentage for constraints
        put_cooling_ducts: Whether to include cooling duct calculations
        progress_callback: Optional progress callback function

    Returns:
        tuple: (optimized_params_array, final_price)
    """
    if not SCIPY_AVAILABLE:
        print("  scipy not available, skipping dual annealing")
        return np.array(initial_params), 1e18

    from scipy.optimize import dual_annealing
    import time

    start_time = time.time()
    eval_count = [0]

    def objective(x):
        eval_count[0] += 1
        price = CalculateFinalizedPriceIntolerant_Optimized(
            int(round(x[0])), x[1], x[2], x[3], x[4], x[5], x[6],
            LVRATE, HVRATE, POWERRATING, FREQUENCY,
            materialToBeUsedWire_Resistivity,
            GUARANTEED_NO_LOAD_LOSS, GUARANTEED_LOAD_LOSS, GUARANTEED_UCC,
            tolerance, False, put_cooling_ducts, circular=HV_WIRE_CIRCULAR
        )
        if price < 0:
            return abs(price) + 1e6
        return price

    x0 = list(initial_params)

    # Bounds around initial point (wider for global exploration)
    bounds = [
        (max(FOILTURNS_MINIMUM, x0[0] - 5), min(FOILTURNS_MAXIMUM, x0[0] + 5)),
        (max(FOILHEIGHT_MINIMUM, x0[1] - 50), min(FOILHEIGHT_MAXIMUM, x0[1] + 50)),
        (max(FOILTHICKNESS_MINIMUM, x0[2] - 0.3), min(FOILTHICKNESS_MAXIMUM, x0[2] + 0.3)),
        (max(HVTHICK_MINIMUM, x0[3] - 0.3), min(HVTHICK_MAXIMUM, x0[3] + 0.3)),
        (max(HV_LEN_MINIMUM, x0[4] - 1.0), min(HV_LEN_MAXIMUM, x0[4] + 1.0)),
        (max(CORE_MINIMUM, x0[5] - 25), min(CORE_MAXIMUM, x0[5] + 25)),
        (max(0, x0[6] - 25), min(x0[5] + 25, x0[6] + 25)),
    ]

    try:
        # Try newer scipy API first (minimizer_kwargs), fall back to older API (local_search_options)
        try:
            result = dual_annealing(
                objective,
                bounds=bounds,
                x0=x0,
                maxiter=300,
                initial_temp=5230.0,
                restart_temp_ratio=2e-5,
                minimizer_kwargs={'method': 'L-BFGS-B', 'options': {'maxiter': 50}},
                seed=42
            )
        except TypeError:
            # Older scipy version
            result = dual_annealing(
                objective,
                bounds=bounds,
                x0=x0,
                maxiter=300,
                initial_temp=5230.0,
                restart_temp_ratio=2e-5,
                local_search_options={'method': 'L-BFGS-B', 'options': {'maxiter': 50}},
                seed=42
            )

        elapsed = time.time() - start_time
        print(f"  Dual annealing: {eval_count[0]} evaluations in {elapsed:.2f}s, best={result.fun:.2f}")

        return result.x, result.fun

    except Exception as e:
        print(f"  Dual annealing failed: {e}")
        return np.array(x0), objective(x0)


def bayesian_polish(
    initial_params,
    tolerance, put_cooling_ducts,
    progress_callback=None
):
    """
    Smart optimization with Gaussian Process surrogate model.

    Builds a probabilistic model of the objective function and uses it
    to decide where to sample next. Very efficient for expensive functions.

    Args:
        initial_params: tuple/list of (turns, height, thick, hvthick, hvlen, core_dia, core_len)
        tolerance: Tolerance percentage for constraints
        put_cooling_ducts: Whether to include cooling duct calculations
        progress_callback: Optional progress callback function

    Returns:
        tuple: (optimized_params_array, final_price)
    """
    try:
        from skopt import gp_minimize
        from skopt.space import Real, Integer
    except ImportError:
        print("  scikit-optimize not available, skipping Bayesian optimization")
        return np.array(initial_params), 1e18

    import time

    start_time = time.time()
    eval_count = [0]

    def objective(x):
        eval_count[0] += 1
        price = CalculateFinalizedPriceIntolerant_Optimized(
            int(round(x[0])), x[1], x[2], x[3], x[4], x[5], x[6],
            LVRATE, HVRATE, POWERRATING, FREQUENCY,
            materialToBeUsedWire_Resistivity,
            GUARANTEED_NO_LOAD_LOSS, GUARANTEED_LOAD_LOSS, GUARANTEED_UCC,
            tolerance, False, put_cooling_ducts, circular=HV_WIRE_CIRCULAR
        )
        if price < 0:
            return abs(price) + 1e6
        return price

    x0 = list(initial_params)
    x0[0] = int(round(x0[0]))  # Ensure turns is integer for Integer dimension

    # Define search space around initial point with valid bounds
    turns_lo = int(max(FOILTURNS_MINIMUM, x0[0] - 5))
    turns_hi = int(min(FOILTURNS_MAXIMUM, x0[0] + 5))
    if turns_lo >= turns_hi:
        turns_hi = turns_lo + 1  # Ensure valid range

    space = [
        Integer(turns_lo, turns_hi, name='turns'),
        Real(max(FOILHEIGHT_MINIMUM, x0[1] - 50), min(FOILHEIGHT_MAXIMUM, x0[1] + 50), name='height'),
        Real(max(FOILTHICKNESS_MINIMUM, x0[2] - 0.3), min(FOILTHICKNESS_MAXIMUM, x0[2] + 0.3), name='thick'),
        Real(max(HVTHICK_MINIMUM, x0[3] - 0.3), min(HVTHICK_MAXIMUM, x0[3] + 0.3), name='hvthick'),
        Real(max(HV_LEN_MINIMUM, x0[4] - 1.0), min(HV_LEN_MAXIMUM, x0[4] + 1.0), name='hvlen'),
        Real(max(CORE_MINIMUM, x0[5] - 25), min(CORE_MAXIMUM, x0[5] + 25), name='core_dia'),
        Real(max(0, x0[6] - 25), min(x0[5] + 25, x0[6] + 25), name='core_len'),
    ]

    try:
        result = gp_minimize(
            objective,
            space,
            x0=x0,
            n_calls=100,
            n_initial_points=10,
            acq_func='EI',  # Expected Improvement
            random_state=42,
            verbose=False
        )

        elapsed = time.time() - start_time
        print(f"  Bayesian optimization: {eval_count[0]} evaluations in {elapsed:.2f}s, best={result.fun:.2f}")

        return np.array(result.x), result.fun

    except Exception as e:
        print(f"  Bayesian optimization failed: {e}")
        return np.array(x0), 1e18


def ensemble_optimize(
    initial_params,
    tolerance, put_cooling_ducts,
    progress_callback=None
):
    """
    Run multiple optimizers in parallel and return the best result.

    Combines the strengths of different optimization methods:
    - L-BFGS-B: Fast local precision
    - Basin Hopping: Escapes local minima
    - Dual Annealing: Handles discontinuities
    - Bayesian: Smart sampling for expensive functions

    Args:
        initial_params: tuple/list of (turns, height, thick, hvthick, hvlen, core_dia, core_len)
        tolerance: Tolerance percentage for constraints
        put_cooling_ducts: Whether to include cooling duct calculations
        progress_callback: Optional progress callback function

    Returns:
        tuple: (optimized_params_array, final_price, winning_method)
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import time

    start_time = time.time()
    print(f"\n  [ENSEMBLE] Running 4 optimizers in parallel...")

    results = {}

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(local_optimizer_polish, initial_params, tolerance, put_cooling_ducts, None): 'L-BFGS-B',
            executor.submit(basin_hopping_polish, initial_params, tolerance, put_cooling_ducts, None): 'Basin Hopping',
            executor.submit(dual_annealing_polish, initial_params, tolerance, put_cooling_ducts, None): 'Dual Annealing',
            executor.submit(bayesian_polish, initial_params, tolerance, put_cooling_ducts, None): 'Bayesian',
        }

        for future in as_completed(futures):
            method = futures[future]
            try:
                params, price = future.result()
                results[method] = (params, price)
            except Exception as e:
                print(f"  {method} failed: {e}")
                results[method] = (np.array(initial_params), 1e18)

    # Find best result
    best_method = min(results, key=lambda k: results[k][1])
    best_params, best_price = results[best_method]

    elapsed = time.time() - start_time
    print(f"\n  [ENSEMBLE] Complete in {elapsed:.2f}s")
    print(f"  Results summary:")
    for method, (_, price) in sorted(results.items(), key=lambda x: x[1][1]):
        marker = " <-- BEST" if method == best_method else ""
        if price < 1e17:
            print(f"    {method}: {price:.2f}{marker}")
        else:
            print(f"    {method}: FAILED{marker}")

    return best_params, best_price, best_method


# =============================================================================
# GPU CUDA GRID SEARCH (Optional - requires CUDA-capable GPU)
# =============================================================================

if CUDA_AVAILABLE:
    @cuda.jit(device=True)
    def cuda_clamp(number, minimum, maximum):
        if number < minimum:
            return minimum
        elif number > maximum:
            return maximum
        return number

    @cuda.jit(device=True)
    def cuda_calculate_core_section(core_diameter, core_length, core_fill_round, core_fill_rect):
        return ((core_diameter**2 * math.pi) / 400.0) * core_fill_round + \
               (core_length * core_diameter / 100.0) * core_fill_rect

    @cuda.jit(device=True)
    def cuda_calculate_induction(volts_per_turn, core_diameter, core_length, freq, core_fill_round, core_fill_rect):
        core_section = cuda_calculate_core_section(core_diameter, core_length, core_fill_round, core_fill_rect)
        return (volts_per_turn * 10000.0) / (math.sqrt(2.0) * math.pi * freq * core_section)

    @cuda.jit(device=True)
    def cuda_calculate_watts_per_kg(induction):
        return (1.3498 * induction**6) + (-8.1737 * induction**5) + \
               (19.884 * induction**4) + (-24.708 * induction**3) + \
               (16.689 * induction**2) + (-5.5386 * induction) + 0.7462

    @cuda.jit(device=True)
    def cuda_calculate_radial_thickness_lv(lv_turns, lv_thick, n_ducts, lv_insulation_thick, duct_thick):
        return lv_turns * lv_thick + ((lv_turns - 1) * lv_insulation_thick) + (n_ducts * (duct_thick + 0.5))

    @cuda.jit(device=True)
    def cuda_calculate_avg_diameter_lv(lv_turns, lv_thick, core_dia, core_len, n_ducts, dist_core_lv, lv_insulation_thick, duct_thick):
        radial = cuda_calculate_radial_thickness_lv(lv_turns, lv_thick, n_ducts, lv_insulation_thick, duct_thick)
        return core_dia + (2.0 * dist_core_lv) + radial + (2.0 * core_len / math.pi)

    @cuda.jit(device=True)
    def cuda_calculate_total_length_lv(lv_turns, lv_thick, core_dia, core_len, n_ducts, dist_core_lv, lv_insulation_thick, duct_thick):
        avg_dia = cuda_calculate_avg_diameter_lv(lv_turns, lv_thick, core_dia, core_len, n_ducts, dist_core_lv, lv_insulation_thick, duct_thick)
        return avg_dia * math.pi * lv_turns

    @cuda.jit(device=True)
    def cuda_calculate_radial_thickness_hv(lv_height, lv_turns, hv_thick, hv_len, n_ducts, hv_rate, lv_rate, insulation_wire, hv_insulation_thick, duct_thick, circular):
        hv_turns = lv_turns * (hv_rate / lv_rate)
        hv_layer_height = lv_height - 50.0
        # For circular wire, axial space per turn = diameter (thickness)
        wire_axial_size = hv_thick if circular else hv_len
        hv_turns_per_layer = (hv_layer_height / (wire_axial_size + insulation_wire)) - 1.0
        if hv_turns_per_layer <= 0:
            return 1e10  # Invalid
        hv_layer_number = math.ceil(hv_turns / hv_turns_per_layer)
        return hv_layer_number * hv_thick + (hv_layer_number - 1) * hv_insulation_thick + (n_ducts * (duct_thick + 0.5))

    @cuda.jit(device=True)
    def cuda_calculate_avg_diameter_hv(lv_turns, lv_thick, core_dia, lv_height, hv_thick, hv_len, core_len, n_ducts_lv, n_ducts_hv,
                                        dist_core_lv, main_gap, hv_rate, lv_rate, insulation_wire, lv_insulation_thick, hv_insulation_thick, duct_thick, circular):
        radial_lv = cuda_calculate_radial_thickness_lv(lv_turns, lv_thick, n_ducts_lv, lv_insulation_thick, duct_thick)
        radial_hv = cuda_calculate_radial_thickness_hv(lv_height, lv_turns, hv_thick, hv_len, n_ducts_hv, hv_rate, lv_rate, insulation_wire, hv_insulation_thick, duct_thick, circular)
        return core_dia + 2.0 * dist_core_lv + 2.0 * radial_lv + 2.0 * main_gap + radial_hv + (2.0 * core_len / math.pi)

    @cuda.jit(device=True)
    def cuda_calculate_total_length_hv(lv_turns, lv_thick, core_dia, lv_height, hv_thick, hv_len, core_len, n_ducts_lv, n_ducts_hv,
                                        dist_core_lv, main_gap, hv_rate, lv_rate, insulation_wire, lv_insulation_thick, hv_insulation_thick, duct_thick, circular):
        hv_turns = lv_turns * hv_rate / lv_rate
        avg_dia = cuda_calculate_avg_diameter_hv(lv_turns, lv_thick, core_dia, lv_height, hv_thick, hv_len, core_len, n_ducts_lv, n_ducts_hv,
                                                  dist_core_lv, main_gap, hv_rate, lv_rate, insulation_wire, lv_insulation_thick, hv_insulation_thick, duct_thick, circular)
        return avg_dia * math.pi * hv_turns

    @cuda.jit(device=True)
    def cuda_calculate_core_weight(lv_turns, lv_height, lv_thick, hv_thick, hv_len, core_dia, core_len, n_ducts_lv, n_ducts_hv,
                                    dist_core_lv, main_gap, phase_gap, hv_rate, lv_rate, insulation_wire,
                                    lv_insulation_thick, hv_insulation_thick, duct_thick, core_density, core_fill_round, core_fill_rect, circular):
        window_height = lv_height + 40.0
        radial_lv = cuda_calculate_radial_thickness_lv(lv_turns, lv_thick, n_ducts_lv, lv_insulation_thick, duct_thick)
        radial_hv = cuda_calculate_radial_thickness_hv(lv_height, lv_turns, hv_thick, hv_len, n_ducts_hv, hv_rate, lv_rate, insulation_wire, hv_insulation_thick, duct_thick, circular)
        radial_total = radial_lv + radial_hv + main_gap + dist_core_lv
        center_between_legs = (core_dia + radial_total * 2.0) + phase_gap
        rect_weight = (((3.0 * window_height) + 2.0 * (2.0 * center_between_legs + core_dia)) * (core_dia * core_len / 100.0) * core_density * core_fill_rect) / 1e6
        radius = core_dia / 2.0
        square_edge = radius * math.sqrt(math.pi)
        round_weight = (((3.0 * (window_height + 10.0)) + 2.0 * (2.0 * center_between_legs + core_dia)) * (square_edge * square_edge / 100.0) * core_density * core_fill_round) / 1e6
        return (rect_weight + round_weight) * 100.0

    @cuda.jit(device=True)
    def cuda_calculate_section_hv(thickness, length, circular):
        if circular:
            # Circular wire: area = pi * (diameter/2)^2
            radius = thickness / 2.0
            return math.pi * radius * radius
        else:
            # Rectangular wire with rounded corners
            return (thickness * length) - (thickness**2) + (((thickness / 2.0)**2) * math.pi)

    @cuda.jit(device=True)
    def cuda_calculate_section_lv(height, thickness):
        return height * thickness

    @cuda.jit(device=True)
    def cuda_calculate_load_losses(lv_turns, lv_thick, core_dia, lv_height, hv_thick, hv_len, resistivity, power, hv_rate, lv_rate, core_len, n_ducts_lv, n_ducts_hv,
                                    dist_core_lv, main_gap, insulation_wire, lv_insulation_thick, hv_insulation_thick, duct_thick, add_loss_lv, add_loss_hv, circular):
        lv_length = cuda_calculate_total_length_lv(lv_turns, lv_thick, core_dia, core_len, n_ducts_lv, dist_core_lv, lv_insulation_thick, duct_thick)
        hv_length = cuda_calculate_total_length_hv(lv_turns, lv_thick, core_dia, lv_height, hv_thick, hv_len, core_len, n_ducts_lv, n_ducts_hv,
                                                    dist_core_lv, main_gap, hv_rate, lv_rate, insulation_wire, lv_insulation_thick, hv_insulation_thick, duct_thick, circular)
        lv_section = cuda_calculate_section_lv(lv_height, lv_thick)
        hv_section = cuda_calculate_section_hv(hv_thick, hv_len, circular)
        lv_resistance = (lv_length / 1000.0) * resistivity / lv_section
        hv_resistance = (hv_length / 1000.0) * resistivity / hv_section
        hv_current = (power * 1000.0) / (hv_rate * 3.0)
        lv_current = (power * 1000.0) / (lv_rate * 3.0)
        lv_losses = lv_resistance * (lv_current**2) * 3.0 * add_loss_lv
        hv_losses = hv_resistance * (hv_current**2) * 3.0 * add_loss_hv
        return lv_losses + hv_losses

    @cuda.jit(device=True)
    def cuda_calculate_no_load_losses(lv_turns, lv_height, lv_thick, hv_thick, hv_len, core_dia, lv_rate, core_len, n_ducts_lv, n_ducts_hv,
                                       dist_core_lv, main_gap, phase_gap, hv_rate, insulation_wire,
                                       lv_insulation_thick, hv_insulation_thick, duct_thick, core_density, core_fill_round, core_fill_rect, freq, circular):
        core_weight = cuda_calculate_core_weight(lv_turns, lv_height, lv_thick, hv_thick, hv_len, core_dia, core_len, n_ducts_lv, n_ducts_hv,
                                                  dist_core_lv, main_gap, phase_gap, hv_rate, lv_rate, insulation_wire,
                                                  lv_insulation_thick, hv_insulation_thick, duct_thick, core_density, core_fill_round, core_fill_rect, circular)
        volts_per_turn = lv_rate / lv_turns
        induction = cuda_calculate_induction(volts_per_turn, core_dia, core_len, freq, core_fill_round, core_fill_rect)
        watts_per_kg = cuda_calculate_watts_per_kg(induction)
        return watts_per_kg * core_weight * 1.2

    @cuda.jit(device=True)
    def cuda_calculate_stray_diameter(lv_turns, lv_thick, lv_height, hv_thick, hv_len, core_dia, core_len, n_ducts_lv, n_ducts_hv,
                                       dist_core_lv, main_gap, hv_rate, lv_rate, insulation_wire, lv_insulation_thick, hv_insulation_thick, duct_thick, circular):
        radial_lv = cuda_calculate_radial_thickness_lv(lv_turns, lv_thick, n_ducts_lv, lv_insulation_thick, duct_thick)
        main_gap_dia = core_dia + dist_core_lv * 2.0 + radial_lv * 2.0 + (2.0 * core_len / math.pi) + main_gap
        radial_hv = cuda_calculate_radial_thickness_hv(lv_height, lv_turns, hv_thick, hv_len, n_ducts_hv, hv_rate, lv_rate, insulation_wire, hv_insulation_thick, duct_thick, circular)
        reduced_hv = radial_hv / 3.0
        reduced_lv = radial_lv / 3.0
        sd = main_gap_dia + reduced_hv - reduced_lv + ((reduced_hv**2 - reduced_lv**2) / (reduced_lv + reduced_hv + main_gap))
        return sd

    @cuda.jit(device=True)
    def cuda_calculate_ux(power, stray_dia, lv_turns, lv_thick, lv_height, hv_thick, hv_len, freq, lv_rate, n_ducts_lv, n_ducts_hv,
                          hv_rate, insulation_wire, lv_insulation_thick, hv_insulation_thick, duct_thick, main_gap, circular):
        radial_hv = cuda_calculate_radial_thickness_hv(lv_height, lv_turns, hv_thick, hv_len, n_ducts_hv, hv_rate, lv_rate, insulation_wire, hv_insulation_thick, duct_thick, circular)
        reduced_hv = radial_hv / 3.0
        radial_lv = cuda_calculate_radial_thickness_lv(lv_turns, lv_thick, n_ducts_lv, lv_insulation_thick, duct_thick)
        reduced_lv = radial_lv / 3.0
        volts_per_turn = lv_rate / lv_turns
        return (power * stray_dia * freq * (reduced_lv + reduced_hv + main_gap)) / (1210.0 * (volts_per_turn**2) * lv_height)

    @cuda.jit(device=True)
    def cuda_calculate_price(lv_turns, lv_height, lv_thick, hv_thick, hv_len, core_dia, core_len, n_ducts_lv, n_ducts_hv,
                              dist_core_lv, main_gap, phase_gap, hv_rate, lv_rate, insulation_wire,
                              lv_insulation_thick, hv_insulation_thick, duct_thick, core_density, core_fill_round, core_fill_rect,
                              core_price, foil_density, foil_price, wire_density, wire_price, circular):
        # Core weight and price
        wc = cuda_calculate_core_weight(lv_turns, lv_height, lv_thick, hv_thick, hv_len, core_dia, core_len, n_ducts_lv, n_ducts_hv,
                                         dist_core_lv, main_gap, phase_gap, hv_rate, lv_rate, insulation_wire,
                                         lv_insulation_thick, hv_insulation_thick, duct_thick, core_density, core_fill_round, core_fill_rect, circular)
        pc = wc * core_price

        # HV volume and price
        hv_length = cuda_calculate_total_length_hv(lv_turns, lv_thick, core_dia, lv_height, hv_thick, hv_len, core_len, n_ducts_lv, n_ducts_hv,
                                                    dist_core_lv, main_gap, hv_rate, lv_rate, insulation_wire, lv_insulation_thick, hv_insulation_thick, duct_thick, circular)
        hv_section = cuda_calculate_section_hv(hv_thick, hv_len, circular)
        volume_hv = (hv_length * hv_section) / 1000000.0 * 3.0
        whv = volume_hv * wire_density
        phv = whv * wire_price

        # LV volume and price
        lv_length = cuda_calculate_total_length_lv(lv_turns, lv_thick, core_dia, core_len, n_ducts_lv, dist_core_lv, lv_insulation_thick, duct_thick)
        volume_lv = (lv_length * lv_height * lv_thick) / 1000000.0 * 3.0
        wlv = volume_lv * foil_density
        plv = wlv * foil_price

        return pc + phv + plv

    @cuda.jit(device=True)
    def cuda_calculate_finalized_price(lv_turns, lv_height, lv_thick, hv_thick, hv_len, core_dia, core_len, tolerance,
                                        lv_rate, hv_rate, power, freq, resistivity,
                                        guaranteed_nll, guaranteed_ll, guaranteed_ucc, ucc_tol,
                                        dist_core_lv, main_gap, phase_gap, insulation_wire,
                                        lv_insulation_thick, hv_insulation_thick, duct_thick,
                                        core_density, core_fill_round, core_fill_rect, core_price,
                                        foil_density, foil_price, wire_density, wire_price,
                                        add_loss_lv, add_loss_hv, penalty_nll, penalty_ll, penalty_ucc, max_gradient, circular):
        """Calculate price with penalties - simplified version without cooling ducts for GPU."""
        n_ducts_lv = 0.0
        n_ducts_hv = 0.0

        # Calculate losses
        ll = cuda_calculate_load_losses(lv_turns, lv_thick, core_dia, lv_height, hv_thick, hv_len, resistivity, power, hv_rate, lv_rate, core_len, n_ducts_lv, n_ducts_hv,
                                         dist_core_lv, main_gap, insulation_wire, lv_insulation_thick, hv_insulation_thick, duct_thick, add_loss_lv, add_loss_hv, circular)

        nll = cuda_calculate_no_load_losses(lv_turns, lv_height, lv_thick, hv_thick, hv_len, core_dia, lv_rate, core_len, n_ducts_lv, n_ducts_hv,
                                             dist_core_lv, main_gap, phase_gap, hv_rate, insulation_wire,
                                             lv_insulation_thick, hv_insulation_thick, duct_thick, core_density, core_fill_round, core_fill_rect, freq, circular)

        # Calculate impedance
        stray_dia = cuda_calculate_stray_diameter(lv_turns, lv_thick, lv_height, hv_thick, hv_len, core_dia, core_len, n_ducts_lv, n_ducts_hv,
                                                   dist_core_lv, main_gap, hv_rate, lv_rate, insulation_wire, lv_insulation_thick, hv_insulation_thick, duct_thick, circular)
        ux = cuda_calculate_ux(power, stray_dia, lv_turns, lv_thick, lv_height, hv_thick, hv_len, freq, lv_rate, n_ducts_lv, n_ducts_hv,
                               hv_rate, insulation_wire, lv_insulation_thick, hv_insulation_thick, duct_thick, main_gap, circular)
        ur = ll / (10.0 * power)
        ucc = math.sqrt(ux**2 + ur**2)

        # Calculate price
        price = cuda_calculate_price(lv_turns, lv_height, lv_thick, hv_thick, hv_len, core_dia, core_len, n_ducts_lv, n_ducts_hv,
                                      dist_core_lv, main_gap, phase_gap, hv_rate, lv_rate, insulation_wire,
                                      lv_insulation_thick, hv_insulation_thick, duct_thick, core_density, core_fill_round, core_fill_rect,
                                      core_price, foil_density, foil_price, wire_density, wire_price, circular)

        # Calculate penalties
        nll_extra = max(0.0, nll - guaranteed_nll)
        ll_extra = max(0.0, ll - guaranteed_ll)
        ucc_extra = max(0.0, abs(ucc - guaranteed_ucc) - abs(ucc_tol))

        penalty_for_nll = nll_extra * penalty_nll
        penalty_for_ll = ll_extra * penalty_ll
        penalty_for_ucc = ucc_extra * penalty_ucc

        total_price = price + penalty_for_nll + penalty_for_ll + penalty_for_ucc

        # Mark as invalid (1e18) if exceeds tolerance - consistent with MPS/MLX implementations
        if (nll_extra > (guaranteed_nll * tolerance / 100.0)) or \
           (ll_extra > (guaranteed_ll * tolerance / 100.0)) or \
           (ucc_extra > (guaranteed_ucc * tolerance / 100.0)):
            total_price = 1e18

        return total_price

    @cuda.jit
    def cuda_grid_search_kernel(combinations, results, params):
        """
        CUDA kernel for parallel grid search.

        Each thread processes one combination of parameters.
        combinations: [n_combinations, 7] array with [turns, height, thick, hvthick, hvlen, core_dia, core_len]
        results: [n_combinations, 8] array for outputs
        params: flattened array of all constant parameters
        """
        idx = cuda.grid(1)

        if idx >= combinations.shape[0]:
            return

        # Extract combination parameters
        turns = combinations[idx, 0]
        height = combinations[idx, 1]
        thick = combinations[idx, 2]
        hvthick = combinations[idx, 3]
        hvlen = combinations[idx, 4]
        core_dia = combinations[idx, 5]
        core_len = combinations[idx, 6]

        # Extract constants from params array
        tolerance = params[0]
        lv_rate = params[1]
        hv_rate = params[2]
        power = params[3]
        freq = params[4]
        resistivity = params[5]
        guaranteed_nll = params[6]
        guaranteed_ll = params[7]
        guaranteed_ucc = params[8]
        ucc_tol = params[9]
        dist_core_lv = params[10]
        main_gap = params[11]
        phase_gap = params[12]
        insulation_wire = params[13]
        lv_insulation_thick = params[14]
        hv_insulation_thick = params[15]
        duct_thick = params[16]
        core_density = params[17]
        core_fill_round = params[18]
        core_fill_rect = params[19]
        core_price = params[20]
        foil_density = params[21]
        foil_price = params[22]
        wire_density = params[23]
        wire_price = params[24]
        add_loss_lv = params[25]
        add_loss_hv = params[26]
        penalty_nll = params[27]
        penalty_ll = params[28]
        penalty_ucc = params[29]
        max_gradient = params[30]
        circular = params[31] > 0.5  # Convert float to bool

        # Quick induction pre-filter
        core_section = cuda_calculate_core_section(core_dia, core_len, core_fill_round, core_fill_rect)
        volts_per_turn = lv_rate / turns
        induction = (volts_per_turn * 10000.0) / (math.sqrt(2.0) * math.pi * freq * core_section)

        # Skip invalid induction ranges
        if induction > 1.95 or induction < 0.8:
            results[idx, 0] = 1e18
            return

        # Basic validity checks (for rectangular wire, hvlen must be >= hvthick)
        # For circular wire, this check is skipped
        if not circular and hvlen < hvthick:
            results[idx, 0] = 1e18
            return

        # Core length must not exceed core diameter
        if core_len > core_dia:
            results[idx, 0] = 1e18
            return

        hv_layer_height = height - 50.0
        # For circular wire, axial space per turn = diameter (hvthick), for rectangular = hvlen
        wire_axial_size = hvthick if circular else hvlen
        hv_turns_per_layer = (hv_layer_height / (wire_axial_size + insulation_wire)) - 1.0
        if hv_turns_per_layer <= 0:
            results[idx, 0] = 1e18
            return

        # Calculate price with penalties
        price = cuda_calculate_finalized_price(
            turns, height, thick, hvthick, hvlen, core_dia, core_len, tolerance,
            lv_rate, hv_rate, power, freq, resistivity,
            guaranteed_nll, guaranteed_ll, guaranteed_ucc, ucc_tol,
            dist_core_lv, main_gap, phase_gap, insulation_wire,
            lv_insulation_thick, hv_insulation_thick, duct_thick,
            core_density, core_fill_round, core_fill_rect, core_price,
            foil_density, foil_price, wire_density, wire_price,
            add_loss_lv, add_loss_hv, penalty_nll, penalty_ll, penalty_ucc, max_gradient, circular
        )

        # Store results
        results[idx, 0] = price
        results[idx, 1] = turns
        results[idx, 2] = height
        results[idx, 3] = thick
        results[idx, 4] = hvthick
        results[idx, 5] = hvlen
        results[idx, 6] = core_dia
        results[idx, 7] = core_len


def StartGPU(tolerance=25, obround=True, put_cooling_ducts=True, print_result=True, grid_resolution='medium', progress_callback=None):
    """
    GPU-accelerated transformer optimization using CUDA.

    Args:
        tolerance: Tolerance percentage for constraints
        obround: Whether core is obround shape
        put_cooling_ducts: Whether to refine with cooling ducts (post-GPU on CPU)
        print_result: Print detailed results
        grid_resolution: 'coarse', 'medium', or 'fine'
        progress_callback: Optional function(stage, progress, message, eta) for UI updates

    Returns:
        Dictionary with optimal parameters, or None if no valid design found
    """
    if not CUDA_AVAILABLE:
        print("CUDA not available, falling back to CPU parallel search")
        return StartOptimized(tolerance, obround, put_cooling_ducts, print_result, grid_resolution=grid_resolution, progress_callback=progress_callback)

    print(f"Starting GPU-accelerated grid search (resolution: {grid_resolution})...")
    start_time = time.time()

    # Grid resolution presets
    if grid_resolution == 'coarse':
        core_dia_step = 15
        core_len_steps = 5
        turns_step = 2
        height_step = 20
        thick_step = 0.15
        hvthick_step = 0.15
        hvlen_step = 0.4
    elif grid_resolution == 'medium':
        core_dia_step = 10
        core_len_steps = 8
        turns_step = 1
        height_step = 15
        thick_step = 0.1
        hvthick_step = 0.1
        hvlen_step = 0.3
    else:  # fine
        core_dia_step = 5
        core_len_steps = 10
        turns_step = 1
        height_step = 10
        thick_step = 0.08
        hvthick_step = 0.08
        hvlen_step = 0.2

    # Generate parameter arrays
    core_dias = np.arange(CORE_MINIMUM, CORE_MAXIMUM + 1, core_dia_step, dtype=np.float32)
    turns = np.arange(FOILTURNS_MINIMUM, FOILTURNS_MAXIMUM + 1, turns_step, dtype=np.float32)
    heights = np.arange(FOILHEIGHT_MINIMUM, FOILHEIGHT_MAXIMUM + 1, height_step, dtype=np.float32)
    thicks = np.arange(FOILTHICKNESS_MINIMUM, FOILTHICKNESS_MAXIMUM + 0.001, thick_step, dtype=np.float32)
    hvthicks = np.arange(HVTHICK_MINIMUM, HVTHICK_MAXIMUM + 0.001, hvthick_step, dtype=np.float32)
    hvlens = np.arange(HV_LEN_MINIMUM, HV_LEN_MAXIMUM + 0.001, hvlen_step, dtype=np.float32)

    # Fast vectorized combination generation
    print("Generating parameter combinations...")
    combinations = _generate_combinations_fast(core_dias, turns, heights, thicks, hvthicks, hvlens, core_len_steps, obround)
    combinations = combinations.astype(np.float64)  # CUDA kernel uses float64
    n_combinations = len(combinations)

    # Prepare constant parameters array
    params = np.array([
        tolerance,                          # 0
        LVRATE,                             # 1
        HVRATE,                             # 2
        POWERRATING,                        # 3
        FREQUENCY,                          # 4
        materialToBeUsedWire_Resistivity,   # 5
        GUARANTEED_NO_LOAD_LOSS,            # 6
        GUARANTEED_LOAD_LOSS,               # 7
        GUARANTEED_UCC,                     # 8
        UCC_TOLERANCE,                      # 9
        DistanceCoreLV,                     # 10
        MainGap,                            # 11
        PhaseGap,                           # 12
        INSULATION_THICKNESS_WIRE,          # 13
        LVInsulationThickness,              # 14
        HVInsulationThickness,              # 15
        COOLING_DUCT_THICKNESS,             # 16
        CoreDensity,                        # 17
        CoreFillingFactorRound,             # 18
        CoreFillingFactorRectangular,       # 19
        CorePricePerKg,                     # 20
        materialToBeUsedFoil_Density,       # 21
        materialToBeUsedFoil_Price,         # 22
        materialToBeUsedWire_Density,       # 23
        materialToBeUsedWire_Price,         # 24
        AdditionalLossFactorLV,             # 25
        AdditionalLossFactorHV,             # 26
        PENALTY_NLL_FACTOR,                 # 27
        PENALTY_LL_FACTOR,                  # 28
        PENALTY_UCC_FACTOR,                 # 29
        MAX_GRADIENT,                       # 30
        1.0 if HV_WIRE_CIRCULAR else 0.0,   # 31 - circular wire flag
    ], dtype=np.float64)

    # Allocate results array
    results = np.zeros((n_combinations, 8), dtype=np.float64)

    # Estimate total time based on throughput (roughly 1M combinations/sec on modern GPU)
    estimated_throughput = 1000000  # combinations per second estimate
    estimated_kernel_time = n_combinations / estimated_throughput
    estimated_total_time = estimated_kernel_time + 2  # Add overhead for transfer

    # Transfer to GPU
    print("Transferring data to GPU...")
    transfer_start = time.time()
    if progress_callback:
        try:
            progress_callback("CUDA GPU", 0.1, f"Transferring {n_combinations:,} combinations to GPU...", estimated_total_time)
        except (InterruptedError, KeyboardInterrupt):
            raise
        except:
            pass

    d_combinations = cuda.to_device(combinations)
    d_results = cuda.to_device(results)
    d_params = cuda.to_device(params)
    transfer_time = time.time() - transfer_start

    # Configure kernel launch
    threads_per_block = 256
    blocks_per_grid = (n_combinations + threads_per_block - 1) // threads_per_block

    print(f"Launching kernel: {blocks_per_grid} blocks x {threads_per_block} threads")

    # Update ETA estimate
    remaining_eta = max(0, estimated_total_time - transfer_time)
    if progress_callback:
        try:
            progress_callback("CUDA GPU", 0.3, f"Running CUDA kernel ({n_combinations:,} combinations)...", remaining_eta)
        except (InterruptedError, KeyboardInterrupt):
            raise
        except:
            pass

    # Launch kernel
    kernel_start = time.time()
    cuda_grid_search_kernel[blocks_per_grid, threads_per_block](d_combinations, d_results, d_params)
    cuda.synchronize()
    kernel_time = time.time() - kernel_start
    print(f"Kernel execution time: {kernel_time:.2f}s")

    if progress_callback:
        try:
            progress_callback("CUDA GPU", 0.9, f"Kernel complete in {kernel_time:.1f}s, processing results...", 1)
        except (InterruptedError, KeyboardInterrupt):
            raise
        except:
            pass

    # Copy results back
    results = d_results.copy_to_host()

    # Find best result
    valid_mask = results[:, 0] > 0
    if not np.any(valid_mask):
        print("No valid design found!")
        return None

    valid_results = results[valid_mask]
    best_idx = np.argmin(valid_results[:, 0])
    best = valid_results[best_idx]

    best_price = best[0]
    best_turns = best[1]
    best_height = best[2]
    best_thick = best[3]
    best_hvthick = best[4]
    best_hvlen = best[5]
    best_core_dia = best[6]
    best_core_len = best[7]

    elapsed_time = time.time() - start_time

    # Refine with cooling ducts on CPU if requested
    if put_cooling_ducts:
        print("\nRefining with cooling ducts calculation (CPU)...")
        refined_price = CalculateFinalizedPriceIntolerant_Optimized(
            best_turns, best_height, best_thick, best_hvthick, best_hvlen,
            best_core_dia, best_core_len,
            LVRATE, HVRATE, POWERRATING, FREQUENCY, materialToBeUsedWire_Resistivity,
            GUARANTEED_NO_LOAD_LOSS, GUARANTEED_LOAD_LOSS, GUARANTEED_UCC,
            tolerance, False, True
        )
        if refined_price > 0:
            best_price = refined_price

    if print_result:
        print(f"\n{'='*60}")
        print("GPU OPTIMIZATION RESULTS")
        print(f"{'='*60}")
        CalculateFinalizedPrice(best_turns, best_height, best_thick, best_hvthick, best_hvlen,
                                best_core_dia, best_core_len, LVRATE, HVRATE, POWERRATING, FREQUENCY,
                                materialToBeUsedWire_Resistivity, GUARANTEED_NO_LOAD_LOSS,
                                GUARANTEED_LOAD_LOSS, GUARANTEED_UCC, isFinal=True, PutCoolingDucts=put_cooling_ducts)
        print(f"\nCore Diameter: {best_core_dia:.1f} mm")
        print(f"Core Length: {best_core_len:.1f} mm")
        print(f"LV Turns: {best_turns:.0f}")
        print(f"LV Foil Height: {best_height:.1f} mm")
        print(f"LV Foil Thickness: {best_thick:.2f} mm")
        print(f"HV Wire Thickness: {best_hvthick:.2f} mm")
        print(f"HV Wire Length: {best_hvlen:.2f} mm")
        print(f"\nOptimization time: {elapsed_time:.2f}s")
        print(f"Combinations evaluated: {n_combinations:,}")
        print(f"Throughput: {n_combinations/kernel_time:,.0f} combinations/second")
        print(f"{'='*60}")

    return {
        'core_diameter': best_core_dia,
        'core_length': best_core_len,
        'lv_turns': best_turns,
        'lv_height': best_height,
        'lv_thickness': best_thick,
        'hv_thickness': best_hvthick,
        'hv_length': best_hvlen,
        'price': best_price,
        'time': elapsed_time,
        'combinations': n_combinations,
        'throughput': n_combinations / kernel_time
    }


# =============================================================================
# FAST COMBINATION GENERATOR (Vectorized)
# =============================================================================

def _generate_combinations_fast(core_dias, turns_arr, heights, thicks, hvthicks, hvlens, core_len_steps, obround):
    """
    Fast vectorized combination generation using NumPy meshgrid.
    Returns combinations array with columns: [turns, height, thick, hvthick, hvlen, core_dia, core_len]
    """
    gen_start = time.time()

    if obround:
        # For obround, use core_len ratios (0 to 1) applied to each core_dia
        core_len_ratios = np.linspace(0, 1, core_len_steps, dtype=np.float32)

        # Create meshgrid for all parameters
        mesh = np.meshgrid(turns_arr, heights, thicks, hvthicks, hvlens, core_dias, core_len_ratios, indexing='ij')
        combinations = np.stack([m.ravel() for m in mesh], axis=1)

        # Column 6 is core_len_ratio, column 5 is core_dia
        # Convert ratio to actual core_len: core_len = ratio * core_dia
        combinations[:, 6] = combinations[:, 6] * combinations[:, 5]
    else:
        # For non-obround, core_len is always 0
        mesh = np.meshgrid(turns_arr, heights, thicks, hvthicks, hvlens, core_dias, indexing='ij')
        flat = np.stack([m.ravel() for m in mesh], axis=1)
        # Add core_len column (all zeros)
        combinations = np.column_stack([flat, np.zeros(len(flat), dtype=np.float32)])

    print(f"Generated {len(combinations):,} combinations in {time.time() - gen_start:.2f}s")
    return combinations


def _generate_batch_for_core_dia(core_dia, turns_arr, heights, thicks, hvthicks, hvlens, core_len_steps, obround):
    """
    Generate combinations for a single core_dia value (memory efficient).
    Returns array with columns: [turns, height, thick, hvthick, hvlen, core_dia, core_len]
    """
    if obround:
        core_lens = np.linspace(0, core_dia, core_len_steps, dtype=np.float32)
    else:
        core_lens = np.array([0.0], dtype=np.float32)

    # Create meshgrid for this core_dia
    mesh = np.meshgrid(turns_arr, heights, thicks, hvthicks, hvlens, core_lens, indexing='ij')
    batch = np.stack([m.ravel() for m in mesh], axis=1)

    # Add core_dia column: [turns, height, thick, hvthick, hvlen, core_len] -> [turns, height, thick, hvthick, hvlen, core_dia, core_len]
    core_dia_col = np.full((len(batch), 1), core_dia, dtype=np.float32)
    batch = np.column_stack([batch[:, :5], core_dia_col, batch[:, 5]])

    return batch


def _generate_subbatches_for_core_dia(core_dia, turns_arr, heights, thicks, hvthicks, hvlens, core_len_steps, obround, max_batch_size=500_000):
    """
    Generator that yields sub-batches for a single core_dia value to avoid OOM.
    Yields arrays with columns: [turns, height, thick, hvthick, hvlen, core_dia, core_len]
    """
    if obround:
        core_lens = np.linspace(0, core_dia, core_len_steps, dtype=np.float32)
    else:
        core_lens = np.array([0.0], dtype=np.float32)

    # Calculate total size to decide chunking strategy
    total_size = len(turns_arr) * len(heights) * len(thicks) * len(hvthicks) * len(hvlens) * len(core_lens)

    if total_size <= max_batch_size:
        # Small enough, generate all at once
        mesh = np.meshgrid(turns_arr, heights, thicks, hvthicks, hvlens, core_lens, indexing='ij')
        batch = np.stack([m.ravel() for m in mesh], axis=1)
        core_dia_col = np.full((len(batch), 1), core_dia, dtype=np.float32)
        batch = np.column_stack([batch[:, :5], core_dia_col, batch[:, 5]])
        yield batch
    else:
        # Chunk by turns to keep memory bounded
        chunk_size_per_turn = len(heights) * len(thicks) * len(hvthicks) * len(hvlens) * len(core_lens)
        turns_per_chunk = max(1, max_batch_size // chunk_size_per_turn)

        for i in range(0, len(turns_arr), turns_per_chunk):
            turns_chunk = turns_arr[i:i + turns_per_chunk]
            mesh = np.meshgrid(turns_chunk, heights, thicks, hvthicks, hvlens, core_lens, indexing='ij')
            batch = np.stack([m.ravel() for m in mesh], axis=1)
            core_dia_col = np.full((len(batch), 1), core_dia, dtype=np.float32)
            batch = np.column_stack([batch[:, :5], core_dia_col, batch[:, 5]])
            yield batch


def _estimate_total_combinations(core_dias, turns_arr, heights, thicks, hvthicks, hvlens, core_len_steps, obround):
    """Estimate total number of combinations without generating them."""
    base = len(turns_arr) * len(heights) * len(thicks) * len(hvthicks) * len(hvlens)
    if obround:
        return base * len(core_dias) * core_len_steps
    else:
        return base * len(core_dias)


def _generate_batches_streaming(core_dias, turns_arr, heights, thicks, hvthicks, hvlens, core_len_steps, obround, batch_size=2_000_000):
    """
    Memory-efficient streaming batch generator.
    Yields batches of combinations without loading all into memory.
    Each batch is [turns, height, thick, hvthick, hvlen, core_dia, core_len]

    Splits by (core_dia, height) pairs to handle large parameter spaces.
    """
    # Combinations per (core_dia, height) pair
    if obround:
        combs_per_pair = len(turns_arr) * len(thicks) * len(hvthicks) * len(hvlens) * core_len_steps
    else:
        combs_per_pair = len(turns_arr) * len(thicks) * len(hvthicks) * len(hvlens)

    # How many (core_dia, height) pairs fit in one batch?
    pairs_per_batch = max(1, batch_size // combs_per_pair)

    # Create all (core_dia, height) pairs
    all_pairs = [(cd, h) for cd in core_dias for h in heights]

    # Process pairs in groups
    for i in range(0, len(all_pairs), pairs_per_batch):
        batch_pairs = all_pairs[i:i + pairs_per_batch]

        # Group by core_dia to use meshgrid efficiently
        from collections import defaultdict
        cd_to_heights = defaultdict(list)
        for cd, h in batch_pairs:
            cd_to_heights[cd].append(h)

        batch_parts = []
        for cd, hs in cd_to_heights.items():
            hs_arr = np.array(hs, dtype=np.float32)
            cd_arr = np.array([cd], dtype=np.float32)

            if obround:
                core_len_ratios = np.linspace(0, 1, core_len_steps, dtype=np.float32)
                mesh = np.meshgrid(turns_arr, hs_arr, thicks, hvthicks, hvlens, cd_arr, core_len_ratios, indexing='ij')
                part = np.stack([m.ravel() for m in mesh], axis=1)
                part[:, 6] = part[:, 6] * part[:, 5]  # ratio -> actual core_len
            else:
                mesh = np.meshgrid(turns_arr, hs_arr, thicks, hvthicks, hvlens, cd_arr, indexing='ij')
                flat = np.stack([m.ravel() for m in mesh], axis=1)
                part = np.column_stack([flat, flat[:, 5]])  # core_len = core_dia

            batch_parts.append(part)

        yield np.vstack(batch_parts)


# =============================================================================
# PyTorch MPS GPU GRID SEARCH (Apple Silicon) - STREAMING BATCHES
# =============================================================================

def _compute_batch_mps(combs, device, tolerance, lv_rate, hv_rate, power, freq, resistivity,
                       guaranteed_nll, guaranteed_ll, guaranteed_ucc, ucc_tol,
                       dist_core_lv, main_gap, phase_gap, insulation_wire,
                       lv_insulation_thick, hv_insulation_thick, duct_thick,
                       core_density, core_fill_round, core_fill_rect, core_price_val,
                       foil_density, foil_price, wire_density, wire_price,
                       add_loss_lv, add_loss_hv, penalty_nll, penalty_ll, penalty_ucc, circular=False):
    """Compute prices for a batch of combinations on MPS GPU. Returns (prices, valid_mask)."""
    turns = combs[:, 0]
    height = combs[:, 1]
    thick = combs[:, 2]
    hvthick = combs[:, 3]
    hvlen = combs[:, 4]
    core_dia = combs[:, 5]
    core_len = combs[:, 6]

    # Core section and induction
    core_section = ((core_dia**2 * math.pi) / 400.0) * core_fill_round + (core_len * core_dia / 100.0) * core_fill_rect
    volts_per_turn = lv_rate / turns
    induction = (volts_per_turn * 10000.0) / (math.sqrt(2.0) * math.pi * freq * core_section)

    # HV layer calculations
    hv_turns = turns * (hv_rate / lv_rate)
    hv_layer_height = height - 50.0
    # For circular wire, axial space per turn = diameter (thickness)
    wire_axial_size = hvthick if circular else hvlen
    hv_turns_per_layer = torch.clamp((hv_layer_height / (wire_axial_size + insulation_wire)) - 1.0, min=0.001)
    hv_layer_number = torch.ceil(hv_turns / hv_turns_per_layer)

    # Radial thicknesses
    radial_lv = turns * thick + ((turns - 1) * lv_insulation_thick)
    radial_hv = hv_layer_number * hvthick + (hv_layer_number - 1) * hv_insulation_thick

    # Average diameters and lengths
    avg_dia_lv = core_dia + (2.0 * dist_core_lv) + radial_lv + (2.0 * core_len / math.pi)
    avg_dia_hv = core_dia + 2.0 * dist_core_lv + 2.0 * radial_lv + 2.0 * main_gap + radial_hv + (2.0 * core_len / math.pi)
    total_len_lv = avg_dia_lv * math.pi * turns
    total_len_hv = avg_dia_hv * math.pi * hv_turns

    # Sections and resistances
    section_lv = height * thick
    if circular:
        # Circular wire: area = pi * (diameter/2)^2
        section_hv = math.pi * (hvthick / 2.0)**2
    else:
        # Rectangular wire with rounded corners
        section_hv = (hvthick * hvlen) - (hvthick**2) + (((hvthick / 2.0)**2) * math.pi)
    resistance_lv = (total_len_lv / 1000.0) * resistivity / section_lv
    resistance_hv = (total_len_hv / 1000.0) * resistivity / section_hv

    # Currents
    current_lv = (power * 1000.0) / (lv_rate * 3.0)
    current_hv = (power * 1000.0) / (hv_rate * 3.0)

    # IEC 60076 Dowell factor calculation (vectorized for PyTorch)
    # Skin depth: δ = sqrt(ρ / (π × f × μ₀)) in mm
    resistivity_ohm_m = resistivity * 1e-6  # Convert Ω·mm²/m to Ω·m
    delta = math.sqrt(resistivity_ohm_m / (math.pi * freq * MU_0)) * 1000.0  # mm

    # LV Dowell factor (foil winding, n_layers = turns)
    xi_lv = thick / delta
    xi_lv = torch.clamp(xi_lv, min=0.01)  # Avoid numerical issues
    sinh_2xi_lv = torch.sinh(2 * xi_lv)
    sin_2xi_lv = torch.sin(2 * xi_lv)
    cosh_2xi_lv = torch.cosh(2 * xi_lv)
    cos_2xi_lv = torch.cos(2 * xi_lv)
    sinh_xi_lv = torch.sinh(xi_lv)
    sin_xi_lv = torch.sin(xi_lv)
    cosh_xi_lv = torch.cosh(xi_lv)
    cos_xi_lv = torch.cos(xi_lv)
    M_lv = xi_lv * (sinh_2xi_lv + sin_2xi_lv) / (cosh_2xi_lv - cos_2xi_lv + 1e-10)
    D_lv = 2 * xi_lv * (sinh_xi_lv - sin_xi_lv) / (cosh_xi_lv + cos_xi_lv + 1e-10)
    m_lv = turns.float()
    F_R_LV = M_lv + ((m_lv * m_lv - 1) / 3.0) * D_lv
    F_R_LV = torch.clamp(F_R_LV, min=1.12, max=5.0)  # 12% minimum safeguard

    # HV Dowell factor (wire winding, n_layers = hv_layer_number)
    h_hv = hvthick * (math.sqrt(math.pi) / 2.0) if circular else hvthick
    xi_hv = h_hv / delta
    xi_hv = torch.clamp(xi_hv, min=0.01)
    sinh_2xi_hv = torch.sinh(2 * xi_hv)
    sin_2xi_hv = torch.sin(2 * xi_hv)
    cosh_2xi_hv = torch.cosh(2 * xi_hv)
    cos_2xi_hv = torch.cos(2 * xi_hv)
    sinh_xi_hv = torch.sinh(xi_hv)
    sin_xi_hv = torch.sin(xi_hv)
    cosh_xi_hv = torch.cosh(xi_hv)
    cos_xi_hv = torch.cos(xi_hv)
    M_hv = xi_hv * (sinh_2xi_hv + sin_2xi_hv) / (cosh_2xi_hv - cos_2xi_hv + 1e-10)
    D_hv = 2 * xi_hv * (sinh_xi_hv - sin_xi_hv) / (cosh_xi_hv + cos_xi_hv + 1e-10)
    m_hv = hv_layer_number.float()
    F_R_HV = M_hv + ((m_hv * m_hv - 1) / 3.0) * D_hv
    F_R_HV = torch.clamp(F_R_HV, min=1.12, max=5.0)  # 12% minimum safeguard

    # Load losses with calculated Dowell factors
    load_losses = resistance_lv * (current_lv**2) * 3.0 * F_R_LV + resistance_hv * (current_hv**2) * 3.0 * F_R_HV

    # Core weight
    window_height = height + 40.0
    radial_total = radial_lv + radial_hv + main_gap + dist_core_lv
    center_between_legs = (core_dia + radial_total * 2.0) + phase_gap
    rect_weight = (((3.0 * window_height) + 2.0 * (2.0 * center_between_legs + core_dia)) * (core_dia * core_len / 100.0) * core_density * core_fill_rect) / 1e6
    square_edge = (core_dia / 2.0) * math.sqrt(math.pi)
    round_weight = (((3.0 * (window_height + 10.0)) + 2.0 * (2.0 * center_between_legs + core_dia)) * (square_edge * square_edge / 100.0) * core_density * core_fill_round) / 1e6
    core_weight = (rect_weight + round_weight) * 100.0

    # No-load losses
    watts_per_kg = 1.3498*induction**6 - 8.1737*induction**5 + 19.884*induction**4 - 24.708*induction**3 + 16.689*induction**2 - 5.5386*induction + 0.7462
    no_load_losses = watts_per_kg * core_weight * 1.2

    # Impedance
    main_gap_dia = core_dia + dist_core_lv * 2.0 + radial_lv * 2.0 + (2.0 * core_len / math.pi) + main_gap
    reduced_hv, reduced_lv = radial_hv / 3.0, radial_lv / 3.0
    stray_dia = main_gap_dia + reduced_hv - reduced_lv + ((reduced_hv**2 - reduced_lv**2) / (reduced_lv + reduced_hv + main_gap + 0.001))
    ux = (power * stray_dia * freq * (reduced_lv + reduced_hv + main_gap)) / (1210.0 * (volts_per_turn**2) * height)
    ucc = torch.sqrt(ux**2 + (load_losses / (10.0 * power))**2)

    # Price
    price_core = core_weight * core_price_val
    price_hv = (total_len_hv * section_hv / 1000000.0 * 3.0) * wire_density * wire_price
    price_lv = (total_len_lv * height * thick / 1000000.0 * 3.0) * foil_density * foil_price
    base_price = price_core + price_hv + price_lv

    # Penalties
    nll_extra = torch.clamp(no_load_losses - guaranteed_nll, min=0.0)
    ll_extra = torch.clamp(load_losses - guaranteed_ll, min=0.0)
    ucc_extra = torch.clamp(torch.abs(ucc - guaranteed_ucc) - abs(ucc_tol), min=0.0)
    total_price = base_price + nll_extra * penalty_nll + ll_extra * penalty_ll + ucc_extra * penalty_ucc

    # Validity
    # For circular wire, skip the hvlen >= hvthick check (not applicable to round wire)
    wire_size_valid = torch.ones_like(hvlen, dtype=torch.bool) if circular else (hvlen >= hvthick)
    valid = (induction >= 0.8) & (induction <= 1.95) & wire_size_valid & (hv_turns_per_layer > 0) & \
            (nll_extra <= guaranteed_nll * tolerance / 100.0) & (ll_extra <= guaranteed_ll * tolerance / 100.0) & \
            (ucc_extra <= guaranteed_ucc * tolerance / 100.0)

    total_price = torch.where(valid, total_price, torch.tensor(1e18, device=device))
    return total_price, combs


def _compute_batch_mlx(combs, tolerance, lv_rate, hv_rate, power, freq, resistivity,
                       guaranteed_nll, guaranteed_ll, guaranteed_ucc, ucc_tol,
                       dist_core_lv, main_gap, phase_gap, insulation_wire,
                       lv_insulation_thick, hv_insulation_thick, duct_thick,
                       core_density, core_fill_round, core_fill_rect, core_price_val,
                       foil_density, foil_price, wire_density, wire_price,
                       add_loss_lv, add_loss_hv, penalty_nll, penalty_ll, penalty_ucc, circular=False):
    """Compute prices for a batch of combinations on MLX. Returns (prices, combs)."""
    turns = combs[:, 0]
    height = combs[:, 1]
    thick = combs[:, 2]
    hvthick = combs[:, 3]
    hvlen = combs[:, 4]
    core_dia = combs[:, 5]
    core_len = combs[:, 6]

    # Core section and induction
    core_section = ((core_dia**2 * math.pi) / 400.0) * core_fill_round + (core_len * core_dia / 100.0) * core_fill_rect
    volts_per_turn = lv_rate / turns
    induction = (volts_per_turn * 10000.0) / (math.sqrt(2.0) * math.pi * freq * core_section)

    # HV layer calculations
    hv_turns = turns * (hv_rate / lv_rate)
    hv_layer_height = height - 50.0
    # For circular wire, axial space per turn = diameter (thickness)
    wire_axial_size = hvthick if circular else hvlen
    hv_turns_per_layer = mx.maximum((hv_layer_height / (wire_axial_size + insulation_wire)) - 1.0, 0.001)
    hv_layer_number = mx.ceil(hv_turns / hv_turns_per_layer)

    # Radial thicknesses
    radial_lv = turns * thick + ((turns - 1) * lv_insulation_thick)
    radial_hv = hv_layer_number * hvthick + (hv_layer_number - 1) * hv_insulation_thick

    # Average diameters and lengths
    avg_dia_lv = core_dia + (2.0 * dist_core_lv) + radial_lv + (2.0 * core_len / math.pi)
    avg_dia_hv = core_dia + 2.0 * dist_core_lv + 2.0 * radial_lv + 2.0 * main_gap + radial_hv + (2.0 * core_len / math.pi)
    total_len_lv = avg_dia_lv * math.pi * turns
    total_len_hv = avg_dia_hv * math.pi * hv_turns

    # Sections and resistances
    section_lv = height * thick
    if circular:
        # Circular wire: area = pi * (diameter/2)^2
        section_hv = math.pi * (hvthick / 2.0)**2
    else:
        # Rectangular wire with rounded corners
        section_hv = (hvthick * hvlen) - (hvthick**2) + (((hvthick / 2.0)**2) * math.pi)
    resistance_lv = (total_len_lv / 1000.0) * resistivity / section_lv
    resistance_hv = (total_len_hv / 1000.0) * resistivity / section_hv

    # Currents
    current_lv = (power * 1000.0) / (lv_rate * 3.0)
    current_hv = (power * 1000.0) / (hv_rate * 3.0)

    # IEC 60076 Dowell factor calculation (vectorized for MLX)
    # Skin depth: δ = sqrt(ρ / (π × f × μ₀)) in mm
    resistivity_ohm_m = resistivity * 1e-6  # Convert Ω·mm²/m to Ω·m
    delta = math.sqrt(resistivity_ohm_m / (math.pi * freq * MU_0)) * 1000.0  # mm

    # LV Dowell factor (foil winding, n_layers = turns)
    xi_lv = thick / delta
    xi_lv = mx.maximum(xi_lv, 0.01)  # Avoid numerical issues
    sinh_2xi_lv = mx.sinh(2 * xi_lv)
    sin_2xi_lv = mx.sin(2 * xi_lv)
    cosh_2xi_lv = mx.cosh(2 * xi_lv)
    cos_2xi_lv = mx.cos(2 * xi_lv)
    sinh_xi_lv = mx.sinh(xi_lv)
    sin_xi_lv = mx.sin(xi_lv)
    cosh_xi_lv = mx.cosh(xi_lv)
    cos_xi_lv = mx.cos(xi_lv)
    M_lv = xi_lv * (sinh_2xi_lv + sin_2xi_lv) / (cosh_2xi_lv - cos_2xi_lv + 1e-10)
    D_lv = 2 * xi_lv * (sinh_xi_lv - sin_xi_lv) / (cosh_xi_lv + cos_xi_lv + 1e-10)
    m_lv = turns.astype(mx.float32)
    F_R_LV = M_lv + ((m_lv * m_lv - 1) / 3.0) * D_lv
    F_R_LV = mx.clip(F_R_LV, 1.12, 5.0)  # 12% minimum safeguard

    # HV Dowell factor (wire winding, n_layers = hv_layer_number)
    h_hv = hvthick * (math.sqrt(math.pi) / 2.0) if circular else hvthick
    xi_hv = h_hv / delta
    xi_hv = mx.maximum(xi_hv, 0.01)
    sinh_2xi_hv = mx.sinh(2 * xi_hv)
    sin_2xi_hv = mx.sin(2 * xi_hv)
    cosh_2xi_hv = mx.cosh(2 * xi_hv)
    cos_2xi_hv = mx.cos(2 * xi_hv)
    sinh_xi_hv = mx.sinh(xi_hv)
    sin_xi_hv = mx.sin(xi_hv)
    cosh_xi_hv = mx.cosh(xi_hv)
    cos_xi_hv = mx.cos(xi_hv)
    M_hv = xi_hv * (sinh_2xi_hv + sin_2xi_hv) / (cosh_2xi_hv - cos_2xi_hv + 1e-10)
    D_hv = 2 * xi_hv * (sinh_xi_hv - sin_xi_hv) / (cosh_xi_hv + cos_xi_hv + 1e-10)
    m_hv = hv_layer_number.astype(mx.float32)
    F_R_HV = M_hv + ((m_hv * m_hv - 1) / 3.0) * D_hv
    F_R_HV = mx.clip(F_R_HV, 1.12, 5.0)  # 12% minimum safeguard

    # Load losses with calculated Dowell factors
    load_losses = resistance_lv * (current_lv**2) * 3.0 * F_R_LV + resistance_hv * (current_hv**2) * 3.0 * F_R_HV

    # Core weight
    window_height = height + 40.0
    radial_total = radial_lv + radial_hv + main_gap + dist_core_lv
    center_between_legs = (core_dia + radial_total * 2.0) + phase_gap
    rect_weight = (((3.0 * window_height) + 2.0 * (2.0 * center_between_legs + core_dia)) * (core_dia * core_len / 100.0) * core_density * core_fill_rect) / 1e6
    square_edge = (core_dia / 2.0) * math.sqrt(math.pi)
    round_weight = (((3.0 * (window_height + 10.0)) + 2.0 * (2.0 * center_between_legs + core_dia)) * (square_edge * square_edge / 100.0) * core_density * core_fill_round) / 1e6
    core_weight = (rect_weight + round_weight) * 100.0

    # No-load losses
    watts_per_kg = 1.3498*induction**6 - 8.1737*induction**5 + 19.884*induction**4 - 24.708*induction**3 + 16.689*induction**2 - 5.5386*induction + 0.7462
    no_load_losses = watts_per_kg * core_weight * 1.2

    # Impedance
    main_gap_dia = core_dia + dist_core_lv * 2.0 + radial_lv * 2.0 + (2.0 * core_len / math.pi) + main_gap
    reduced_hv, reduced_lv = radial_hv / 3.0, radial_lv / 3.0
    stray_dia = main_gap_dia + reduced_hv - reduced_lv + ((reduced_hv**2 - reduced_lv**2) / (reduced_lv + reduced_hv + main_gap + 0.001))
    ux = (power * stray_dia * freq * (reduced_lv + reduced_hv + main_gap)) / (1210.0 * (volts_per_turn**2) * height)
    ucc = mx.sqrt(ux**2 + (load_losses / (10.0 * power))**2)

    # Price
    price_core = core_weight * core_price_val
    price_hv = (total_len_hv * section_hv / 1000000.0 * 3.0) * wire_density * wire_price
    price_lv = (total_len_lv * height * thick / 1000000.0 * 3.0) * foil_density * foil_price
    base_price = price_core + price_hv + price_lv

    # Penalties
    nll_extra = mx.maximum(no_load_losses - guaranteed_nll, 0.0)
    ll_extra = mx.maximum(load_losses - guaranteed_ll, 0.0)
    ucc_extra = mx.maximum(mx.abs(ucc - guaranteed_ucc) - abs(ucc_tol), 0.0)
    total_price = base_price + nll_extra * penalty_nll + ll_extra * penalty_ll + ucc_extra * penalty_ucc

    # Validity
    # For circular wire, skip the hvlen >= hvthick check (not applicable to round wire)
    wire_size_valid = mx.ones(hvlen.shape, dtype=mx.bool_) if circular else (hvlen >= hvthick)
    valid = (induction >= 0.8) & (induction <= 1.95) & wire_size_valid & (hv_turns_per_layer > 0) & \
            (nll_extra <= guaranteed_nll * tolerance / 100.0) & (ll_extra <= guaranteed_ll * tolerance / 100.0) & \
            (ucc_extra <= guaranteed_ucc * tolerance / 100.0)

    total_price = mx.where(valid, total_price, mx.array(1e18))
    return total_price, combs


def StartMPS(tolerance=25, obround=True, put_cooling_ducts=True, print_result=True, grid_resolution='medium', progress_callback=None):
    """
    PyTorch MPS GPU-accelerated transformer optimization for Apple Silicon.
    Uses streaming batch processing to minimize memory usage.
    """
    if not MPS_AVAILABLE:
        print("PyTorch MPS not available, falling back to CPU parallel search")
        return StartOptimized(tolerance, obround, put_cooling_ducts, print_result, grid_resolution=grid_resolution, progress_callback=progress_callback)

    print(f"Starting PyTorch MPS GPU grid search (resolution: {grid_resolution})...")
    start_time = time.time()
    device = torch.device("mps")

    # Grid resolution presets - balanced for practical runtimes
    # Target: coarse ~5M, medium ~50M, fine ~200M combinations
    if grid_resolution == 'coarse':
        # ~5M combinations, ~15-30s on GPU
        core_dia_step, core_len_steps, turns_step, height_step = 30, 3, 5, 50
        thick_step, hvthick_step, hvlen_step = 0.5, 0.5, 1.5
    elif grid_resolution == 'medium':
        # ~50M combinations, ~1-2 min on GPU
        core_dia_step, core_len_steps, turns_step, height_step = 20, 5, 3, 30
        thick_step, hvthick_step, hvlen_step = 0.3, 0.3, 1.0
    else:  # fine
        # ~200M combinations, ~3-5 min on GPU
        core_dia_step, core_len_steps, turns_step, height_step = 15, 6, 2, 20
        thick_step, hvthick_step, hvlen_step = 0.2, 0.2, 0.6

    # Generate parameter arrays (small - just the grid points)
    core_dias = np.arange(CORE_MINIMUM, CORE_MAXIMUM + 1, core_dia_step, dtype=np.float32)
    turns_arr = np.arange(FOILTURNS_MINIMUM, FOILTURNS_MAXIMUM + 1, turns_step, dtype=np.float32)
    heights = np.arange(FOILHEIGHT_MINIMUM, FOILHEIGHT_MAXIMUM + 1, height_step, dtype=np.float32)
    thicks = np.arange(FOILTHICKNESS_MINIMUM, FOILTHICKNESS_MAXIMUM + 0.001, thick_step, dtype=np.float32)
    hvthicks = np.arange(HVTHICK_MINIMUM, HVTHICK_MAXIMUM + 0.001, hvthick_step, dtype=np.float32)
    hvlens = np.arange(HV_LEN_MINIMUM, HV_LEN_MAXIMUM + 0.001, hvlen_step, dtype=np.float32)

    # Estimate total without generating
    n_total = _estimate_total_combinations(core_dias, turns_arr, heights, thicks, hvthicks, hvlens, core_len_steps, obround)
    print(f"Total combinations to evaluate: {n_total:,}")

    # Constants for GPU computation
    params = {
        'tolerance': tolerance, 'lv_rate': float(LVRATE), 'hv_rate': float(HVRATE),
        'power': float(POWERRATING), 'freq': float(FREQUENCY),
        'resistivity': float(materialToBeUsedWire_Resistivity),
        'guaranteed_nll': float(GUARANTEED_NO_LOAD_LOSS), 'guaranteed_ll': float(GUARANTEED_LOAD_LOSS),
        'guaranteed_ucc': float(GUARANTEED_UCC), 'ucc_tol': float(UCC_TOLERANCE),
        'dist_core_lv': float(DistanceCoreLV), 'main_gap': float(MainGap), 'phase_gap': float(PhaseGap),
        'insulation_wire': float(INSULATION_THICKNESS_WIRE),
        'lv_insulation_thick': float(LVInsulationThickness), 'hv_insulation_thick': float(HVInsulationThickness),
        'duct_thick': float(COOLING_DUCT_THICKNESS), 'core_density': float(CoreDensity),
        'core_fill_round': float(CoreFillingFactorRound), 'core_fill_rect': float(CoreFillingFactorRectangular),
        'core_price_val': float(CorePricePerKg), 'foil_density': float(materialToBeUsedFoil_Density),
        'foil_price': float(materialToBeUsedFoil_Price), 'wire_density': float(materialToBeUsedWire_Density),
        'wire_price': float(materialToBeUsedWire_Price), 'add_loss_lv': float(AdditionalLossFactorLV),
        'add_loss_hv': float(AdditionalLossFactorHV), 'penalty_nll': float(PENALTY_NLL_FACTOR),
        'penalty_ll': float(PENALTY_LL_FACTOR), 'penalty_ucc': float(PENALTY_UCC_FACTOR),
        'circular': HV_WIRE_CIRCULAR
    }

    # Streaming batch processing - 2M combinations per batch for memory efficiency
    batch_size = 2_000_000
    print(f"Processing streaming batches on MPS GPU (~{batch_size:,} per batch)...")
    kernel_start = time.time()

    best_price = 1e18
    best_params_result = None
    n_processed = 0
    batch_num = 0

    # Stream batches - never hold all combinations in memory
    for batch in _generate_batches_streaming(core_dias, turns_arr, heights, thicks, hvthicks, hvlens, core_len_steps, obround, batch_size):
        batch_num += 1
        n_processed += len(batch)

        # Transfer to GPU and compute
        combs = torch.tensor(batch, device=device)
        prices, _ = _compute_batch_mps(combs, device, **params)

        # Find best in this batch
        batch_best_idx = torch.argmin(prices).item()
        batch_best_price = prices[batch_best_idx].item()

        if batch_best_price < best_price:
            best_price = batch_best_price
            best_params_result = combs[batch_best_idx].cpu().numpy()

        # Progress with ETA
        pct = 100.0 * n_processed / n_total
        elapsed = time.time() - kernel_start
        if n_processed > 0 and n_processed < n_total:
            eta = (elapsed / n_processed) * (n_total - n_processed)
        else:
            eta = None
        print(f"  Batch {batch_num}: {n_processed:,}/{n_total:,} ({pct:.1f}%) - ETA: {eta:.1f}s" if eta else f"  Batch {batch_num}: {n_processed:,}/{n_total:,} ({pct:.1f}%)")

        # Call progress callback with ETA
        if progress_callback:
            try:
                progress_callback("MPS GPU", pct / 100.0, f"Batch {batch_num}: {n_processed:,}/{n_total:,}", eta)
            except (InterruptedError, KeyboardInterrupt):
                raise
            except:
                pass

        # Free memory immediately
        del combs, prices, batch

    kernel_time = time.time() - kernel_start
    print(f"GPU computation time: {kernel_time:.2f}s")

    if best_price >= 1e17 or best_params_result is None:
        print("No valid design found!")
        return None

    best_turns = float(best_params_result[0])
    best_height = float(best_params_result[1])
    best_thick = float(best_params_result[2])
    best_hvthick = float(best_params_result[3])
    best_hvlen = float(best_params_result[4])
    best_core_dia = float(best_params_result[5])
    best_core_len = float(best_params_result[6])

    elapsed_time = time.time() - start_time

    # Refine with cooling ducts on CPU if requested
    if put_cooling_ducts:
        print("\nRefining with cooling ducts calculation (CPU)...")
        refined_price = CalculateFinalizedPriceIntolerant_Optimized(
            best_turns, best_height, best_thick, best_hvthick, best_hvlen,
            best_core_dia, best_core_len,
            LVRATE, HVRATE, POWERRATING, FREQUENCY, materialToBeUsedWire_Resistivity,
            GUARANTEED_NO_LOAD_LOSS, GUARANTEED_LOAD_LOSS, GUARANTEED_UCC,
            tolerance, False, True
        )
        if refined_price > 0:
            best_price = refined_price

    if print_result:
        print(f"\n{'='*60}")
        print("MPS GPU OPTIMIZATION RESULTS")
        print(f"{'='*60}")
        CalculateFinalizedPrice(best_turns, best_height, best_thick, best_hvthick, best_hvlen,
                                best_core_dia, best_core_len, LVRATE, HVRATE, POWERRATING, FREQUENCY,
                                materialToBeUsedWire_Resistivity, GUARANTEED_NO_LOAD_LOSS,
                                GUARANTEED_LOAD_LOSS, GUARANTEED_UCC, isFinal=True, PutCoolingDucts=put_cooling_ducts)
        print(f"\nCore Diameter: {best_core_dia:.1f} mm")
        print(f"Core Length: {best_core_len:.1f} mm")
        print(f"LV Turns: {best_turns:.0f}")
        print(f"LV Foil Height: {best_height:.1f} mm")
        print(f"LV Foil Thickness: {best_thick:.2f} mm")
        print(f"HV Wire Thickness: {best_hvthick:.2f} mm")
        print(f"HV Wire Length: {best_hvlen:.2f} mm")
        print(f"\nOptimization time: {elapsed_time:.2f}s")
        print(f"Combinations evaluated: {n_total:,}")
        print(f"Throughput: {n_total/kernel_time:,.0f} combinations/second")
        print(f"{'='*60}")

    return {
        'core_diameter': best_core_dia,
        'core_length': best_core_len,
        'lv_turns': best_turns,
        'lv_height': best_height,
        'lv_thickness': best_thick,
        'hv_thickness': best_hvthick,
        'hv_length': best_hvlen,
        'price': best_price,
        'time': elapsed_time,
        'combinations': n_total,
        'throughput': n_total / kernel_time
    }


# =============================================================================
# MLX GPU GRID SEARCH (Apple Silicon) - STREAMING BATCHES
# =============================================================================

def StartMLX(tolerance=25, obround=True, put_cooling_ducts=True, print_result=True, grid_resolution='medium', progress_callback=None):
    """
    MLX GPU-accelerated transformer optimization for Apple Silicon.
    Uses streaming batch processing to minimize memory usage.
    """
    if not MLX_AVAILABLE:
        print("MLX not available, falling back to CPU parallel search")
        return StartOptimized(tolerance, obround, put_cooling_ducts, print_result, grid_resolution=grid_resolution, progress_callback=progress_callback)

    print(f"Starting MLX GPU grid search (resolution: {grid_resolution})...")
    start_time = time.time()

    # Grid resolution presets - balanced for practical runtimes
    # Target: coarse ~5M, medium ~50M, fine ~200M combinations
    if grid_resolution == 'coarse':
        # ~5M combinations, ~15-30s on GPU
        core_dia_step, core_len_steps, turns_step, height_step = 30, 3, 5, 50
        thick_step, hvthick_step, hvlen_step = 0.5, 0.5, 1.5
    elif grid_resolution == 'medium':
        # ~50M combinations, ~1-2 min on GPU
        core_dia_step, core_len_steps, turns_step, height_step = 20, 5, 3, 30
        thick_step, hvthick_step, hvlen_step = 0.3, 0.3, 1.0
    else:  # fine
        # ~200M combinations, ~3-5 min on GPU
        core_dia_step, core_len_steps, turns_step, height_step = 15, 6, 2, 20
        thick_step, hvthick_step, hvlen_step = 0.2, 0.2, 0.6

    # Generate parameter arrays (small - just the grid points)
    core_dias = np.arange(CORE_MINIMUM, CORE_MAXIMUM + 1, core_dia_step, dtype=np.float32)
    turns_arr = np.arange(FOILTURNS_MINIMUM, FOILTURNS_MAXIMUM + 1, turns_step, dtype=np.float32)
    heights = np.arange(FOILHEIGHT_MINIMUM, FOILHEIGHT_MAXIMUM + 1, height_step, dtype=np.float32)
    thicks = np.arange(FOILTHICKNESS_MINIMUM, FOILTHICKNESS_MAXIMUM + 0.001, thick_step, dtype=np.float32)
    hvthicks = np.arange(HVTHICK_MINIMUM, HVTHICK_MAXIMUM + 0.001, hvthick_step, dtype=np.float32)
    hvlens = np.arange(HV_LEN_MINIMUM, HV_LEN_MAXIMUM + 0.001, hvlen_step, dtype=np.float32)

    # Estimate total without generating
    n_total = _estimate_total_combinations(core_dias, turns_arr, heights, thicks, hvthicks, hvlens, core_len_steps, obround)
    print(f"Total combinations to evaluate: {n_total:,}")

    # Constants for GPU computation
    params = {
        'tolerance': tolerance, 'lv_rate': float(LVRATE), 'hv_rate': float(HVRATE),
        'power': float(POWERRATING), 'freq': float(FREQUENCY),
        'resistivity': float(materialToBeUsedWire_Resistivity),
        'guaranteed_nll': float(GUARANTEED_NO_LOAD_LOSS), 'guaranteed_ll': float(GUARANTEED_LOAD_LOSS),
        'guaranteed_ucc': float(GUARANTEED_UCC), 'ucc_tol': float(UCC_TOLERANCE),
        'dist_core_lv': float(DistanceCoreLV), 'main_gap': float(MainGap), 'phase_gap': float(PhaseGap),
        'insulation_wire': float(INSULATION_THICKNESS_WIRE),
        'lv_insulation_thick': float(LVInsulationThickness), 'hv_insulation_thick': float(HVInsulationThickness),
        'duct_thick': float(COOLING_DUCT_THICKNESS), 'core_density': float(CoreDensity),
        'core_fill_round': float(CoreFillingFactorRound), 'core_fill_rect': float(CoreFillingFactorRectangular),
        'core_price_val': float(CorePricePerKg), 'foil_density': float(materialToBeUsedFoil_Density),
        'foil_price': float(materialToBeUsedFoil_Price), 'wire_density': float(materialToBeUsedWire_Density),
        'wire_price': float(materialToBeUsedWire_Price), 'add_loss_lv': float(AdditionalLossFactorLV),
        'add_loss_hv': float(AdditionalLossFactorHV), 'penalty_nll': float(PENALTY_NLL_FACTOR),
        'penalty_ll': float(PENALTY_LL_FACTOR), 'penalty_ucc': float(PENALTY_UCC_FACTOR),
        'circular': HV_WIRE_CIRCULAR
    }

    # Streaming batch processing - 2M combinations per batch for memory efficiency
    batch_size = 2_000_000
    print(f"Processing streaming batches on MLX GPU (~{batch_size:,} per batch)...")
    kernel_start = time.time()

    best_price = 1e18
    best_params_result = None
    n_processed = 0
    batch_num = 0

    # Stream batches - never hold all combinations in memory
    for batch in _generate_batches_streaming(core_dias, turns_arr, heights, thicks, hvthicks, hvlens, core_len_steps, obround, batch_size):
        batch_num += 1
        n_processed += len(batch)

        # Transfer to MLX and compute
        combs = mx.array(batch)
        prices, _ = _compute_batch_mlx(combs, **params)
        mx.eval(prices)

        # Find best in this batch
        batch_best_idx = int(mx.argmin(prices).item())
        batch_best_price = float(prices[batch_best_idx].item())

        if batch_best_price < best_price:
            best_price = batch_best_price
            best_params_result = np.array(combs[batch_best_idx].tolist())

        # Progress with ETA
        pct = 100.0 * n_processed / n_total
        elapsed = time.time() - kernel_start
        if n_processed > 0 and n_processed < n_total:
            eta = (elapsed / n_processed) * (n_total - n_processed)
        else:
            eta = None
        print(f"  Batch {batch_num}: {n_processed:,}/{n_total:,} ({pct:.1f}%) - ETA: {eta:.1f}s" if eta else f"  Batch {batch_num}: {n_processed:,}/{n_total:,} ({pct:.1f}%)")

        # Call progress callback with ETA
        if progress_callback:
            try:
                progress_callback("MLX GPU", pct / 100.0, f"Batch {batch_num}: {n_processed:,}/{n_total:,}", eta)
            except (InterruptedError, KeyboardInterrupt):
                raise
            except:
                pass

        # Free memory immediately
        del combs, prices, batch

    kernel_time = time.time() - kernel_start
    print(f"GPU computation time: {kernel_time:.2f}s")

    if best_price >= 1e17 or best_params_result is None:
        print("No valid design found!")
        return None

    best_turns = float(best_params_result[0])
    best_height = float(best_params_result[1])
    best_thick = float(best_params_result[2])
    best_hvthick = float(best_params_result[3])
    best_hvlen = float(best_params_result[4])
    best_core_dia = float(best_params_result[5])
    best_core_len = float(best_params_result[6])

    elapsed_time = time.time() - start_time

    # Refine with cooling ducts on CPU if requested
    if put_cooling_ducts:
        print("\nRefining with cooling ducts calculation (CPU)...")
        refined_price = CalculateFinalizedPriceIntolerant_Optimized(
            best_turns, best_height, best_thick, best_hvthick, best_hvlen,
            best_core_dia, best_core_len,
            LVRATE, HVRATE, POWERRATING, FREQUENCY, materialToBeUsedWire_Resistivity,
            GUARANTEED_NO_LOAD_LOSS, GUARANTEED_LOAD_LOSS, GUARANTEED_UCC,
            tolerance, False, True
        )
        if refined_price > 0:
            best_price = refined_price

    if print_result:
        print(f"\n{'='*60}")
        print("MLX GPU OPTIMIZATION RESULTS")
        print(f"{'='*60}")
        CalculateFinalizedPrice(best_turns, best_height, best_thick, best_hvthick, best_hvlen,
                                best_core_dia, best_core_len, LVRATE, HVRATE, POWERRATING, FREQUENCY,
                                materialToBeUsedWire_Resistivity, GUARANTEED_NO_LOAD_LOSS,
                                GUARANTEED_LOAD_LOSS, GUARANTEED_UCC, isFinal=True, PutCoolingDucts=put_cooling_ducts)
        print(f"\nCore Diameter: {best_core_dia:.1f} mm")
        print(f"Core Length: {best_core_len:.1f} mm")
        print(f"LV Turns: {best_turns:.0f}")
        print(f"LV Foil Height: {best_height:.1f} mm")
        print(f"LV Foil Thickness: {best_thick:.2f} mm")
        print(f"HV Wire Thickness: {best_hvthick:.2f} mm")
        print(f"HV Wire Length: {best_hvlen:.2f} mm")
        print(f"\nOptimization time: {elapsed_time:.2f}s")
        print(f"Combinations evaluated: {n_total:,}")
        print(f"Throughput: {n_total/kernel_time:,.0f} combinations/second")
        print(f"{'='*60}")

    return {
        'core_diameter': best_core_dia,
        'core_length': best_core_len,
        'lv_turns': best_turns,
        'lv_height': best_height,
        'lv_thickness': best_thick,
        'hv_thickness': best_hvthick,
        'hv_length': best_hvlen,
        'price': best_price,
        'time': elapsed_time,
        'combinations': n_total,
        'throughput': n_total / kernel_time
    }


def StartMPSHybrid(tolerance=25, obround=True, put_cooling_ducts=True, print_result=True, search_depth='normal', progress_callback=None):
    """
    Two-stage hybrid optimization for maximum accuracy without evaluating trillions.

    Stage 1: Coarse GPU search to find top candidates
    Stage 2: Fine GPU search in zoomed regions around best candidates
    Stage 3: Local CPU search WITH cooling ducts for final refinement

    Args:
        search_depth: Controls thoroughness of search
            - 'fast': Quick search (~10s) - good for rough estimates
            - 'normal': Balanced search (~30s) - good default
            - 'thorough': Detailed search (~2-5min) - better results
            - 'exhaustive': Maximum precision (~10-30min) - best possible results
        progress_callback: Optional callback function(stage, progress, message, eta_seconds)
            - stage: 1, 2, or 3
            - progress: 0.0 to 1.0 (overall progress)
            - message: Status message string
            - eta_seconds: Estimated time remaining (or None)
    """
    def report_progress(stage, progress, message, eta=None):
        """Report progress via callback and print."""
        if progress_callback:
            try:
                progress_callback(stage, progress, message, eta)
            except (InterruptedError, KeyboardInterrupt):
                raise
            except:
                pass  # Ignore other callback errors
    if not MPS_AVAILABLE:
        print("PyTorch MPS not available, falling back to CPU")
        return StartSmartOptimized(tolerance, obround, put_cooling_ducts, use_de=True, print_result=print_result)

    # =========================================================================
    # SEARCH DEPTH PRESETS
    # =========================================================================
    if search_depth == 'fast':
        # ~10s total - quick estimate
        coarse_steps = {'core_dia': 30, 'core_len': 3, 'turns': 5, 'height': 50, 'thick': 0.5, 'hvthick': 0.5, 'hvlen': 1.5}
        fine_steps = {'core_dia': 10, 'turns': 2, 'height': 20, 'thick': 0.15, 'hvthick': 0.15, 'hvlen': 0.4}
        local_ranges = {'turns': 2, 'height': 20, 'thick': 0.2, 'hvthick': 0.2, 'hvlen': 0.6, 'core_dia': 10, 'core_len': 15}
        local_steps = {'height': 10, 'thick': 0.1, 'hvthick': 0.1, 'hvlen': 0.2, 'core_dia': 5, 'core_len': 10}
        n_regions = 2
        fine_core_len_steps = 5
    elif search_depth == 'thorough':
        # ~2-5min - detailed search
        coarse_steps = {'core_dia': 20, 'core_len': 5, 'turns': 3, 'height': 30, 'thick': 0.3, 'hvthick': 0.3, 'hvlen': 0.8}
        fine_steps = {'core_dia': 4, 'turns': 1, 'height': 8, 'thick': 0.04, 'hvthick': 0.04, 'hvlen': 0.08}
        # Stage 3 local search - keep combinations manageable (~3-5M max)
        local_ranges = {'turns': 3, 'height': 30, 'thick': 0.3, 'hvthick': 0.3, 'hvlen': 1.0, 'core_dia': 15, 'core_len': 20}
        local_steps = {'height': 8, 'thick': 0.08, 'hvthick': 0.08, 'hvlen': 0.15, 'core_dia': 4, 'core_len': 6}
        n_regions = 5
        fine_core_len_steps = 12
    elif search_depth == 'exhaustive':
        # ~10-30min - maximum precision
        coarse_steps = {'core_dia': 15, 'core_len': 6, 'turns': 2, 'height': 25, 'thick': 0.25, 'hvthick': 0.25, 'hvlen': 0.6}
        fine_steps = {'core_dia': 2, 'turns': 1, 'height': 5, 'thick': 0.02, 'hvthick': 0.02, 'hvlen': 0.05}
        # Stage 3 local search - high resolution for fine details (~10-20M combinations)
        local_ranges = {'turns': 5, 'height': 50, 'thick': 0.5, 'hvthick': 0.5, 'hvlen': 2.0, 'core_dia': 25, 'core_len': 40}
        local_steps = {'height': 4, 'thick': 0.04, 'hvthick': 0.04, 'hvlen': 0.08, 'core_dia': 3, 'core_len': 4}
        n_regions = 7
        fine_core_len_steps = 15
    else:  # normal (default)
        # ~30s - balanced
        coarse_steps = {'core_dia': 25, 'core_len': 4, 'turns': 4, 'height': 40, 'thick': 0.4, 'hvthick': 0.4, 'hvlen': 1.2}
        fine_steps = {'core_dia': 5, 'turns': 1, 'height': 10, 'thick': 0.05, 'hvthick': 0.05, 'hvlen': 0.1}
        local_ranges = {'turns': 3, 'height': 30, 'thick': 0.3, 'hvthick': 0.3, 'hvlen': 1.0, 'core_dia': 15, 'core_len': 20}
        local_steps = {'height': 10, 'thick': 0.05, 'hvthick': 0.05, 'hvlen': 0.1, 'core_dia': 5, 'core_len': 10}
        n_regions = 3
        fine_core_len_steps = 8

    print("=" * 60)
    print(f"HYBRID GPU OPTIMIZATION (depth: {search_depth})")
    print("Stage 1: Coarse GPU search for promising regions")
    print("Stage 2: Fine GPU zoom into best candidates")
    print("Stage 3: Local CPU search WITH cooling ducts")
    print("=" * 60)
    total_start = time.time()
    device = torch.device("mps")

    report_progress(1, 0.0, "Starting optimization...", None)

    # =========================================================================
    # STAGE 1: COARSE SEARCH - Find top candidates
    # =========================================================================
    print("\n[STAGE 1] Coarse GPU search...")
    report_progress(1, 0.01, "Stage 1: Coarse GPU search...", None)

    # Coarse grid - large steps
    core_dia_step = coarse_steps['core_dia']
    core_len_steps = coarse_steps['core_len']
    turns_step = coarse_steps['turns']
    height_step = coarse_steps['height']
    thick_step = coarse_steps['thick']
    hvthick_step = coarse_steps['hvthick']
    hvlen_step = coarse_steps['hvlen']

    core_dias = np.arange(CORE_MINIMUM, CORE_MAXIMUM + 1, core_dia_step, dtype=np.float32)
    turns_arr = np.arange(FOILTURNS_MINIMUM, FOILTURNS_MAXIMUM + 1, turns_step, dtype=np.float32)
    heights = np.arange(FOILHEIGHT_MINIMUM, FOILHEIGHT_MAXIMUM + 1, height_step, dtype=np.float32)
    thicks = np.arange(FOILTHICKNESS_MINIMUM, FOILTHICKNESS_MAXIMUM + 0.001, thick_step, dtype=np.float32)
    hvthicks = np.arange(HVTHICK_MINIMUM, HVTHICK_MAXIMUM + 0.001, hvthick_step, dtype=np.float32)
    hvlens = np.arange(HV_LEN_MINIMUM, HV_LEN_MAXIMUM + 0.001, hvlen_step, dtype=np.float32)

    n_coarse = _estimate_total_combinations(core_dias, turns_arr, heights, thicks, hvthicks, hvlens, core_len_steps, obround)
    print(f"  Coarse grid: {n_coarse:,} combinations")

    params = {
        'tolerance': tolerance, 'lv_rate': float(LVRATE), 'hv_rate': float(HVRATE),
        'power': float(POWERRATING), 'freq': float(FREQUENCY),
        'resistivity': float(materialToBeUsedWire_Resistivity),
        'guaranteed_nll': float(GUARANTEED_NO_LOAD_LOSS), 'guaranteed_ll': float(GUARANTEED_LOAD_LOSS),
        'guaranteed_ucc': float(GUARANTEED_UCC), 'ucc_tol': float(UCC_TOLERANCE),
        'dist_core_lv': float(DistanceCoreLV), 'main_gap': float(MainGap), 'phase_gap': float(PhaseGap),
        'insulation_wire': float(INSULATION_THICKNESS_WIRE),
        'lv_insulation_thick': float(LVInsulationThickness), 'hv_insulation_thick': float(HVInsulationThickness),
        'duct_thick': float(COOLING_DUCT_THICKNESS), 'core_density': float(CoreDensity),
        'core_fill_round': float(CoreFillingFactorRound), 'core_fill_rect': float(CoreFillingFactorRectangular),
        'core_price_val': float(CorePricePerKg), 'foil_density': float(materialToBeUsedFoil_Density),
        'foil_price': float(materialToBeUsedFoil_Price), 'wire_density': float(materialToBeUsedWire_Density),
        'wire_price': float(materialToBeUsedWire_Price), 'add_loss_lv': float(AdditionalLossFactorLV),
        'add_loss_hv': float(AdditionalLossFactorHV), 'penalty_nll': float(PENALTY_NLL_FACTOR),
        'penalty_ll': float(PENALTY_LL_FACTOR), 'penalty_ucc': float(PENALTY_UCC_FACTOR),
        'circular': HV_WIRE_CIRCULAR
    }

    # Collect all coarse results with prices
    all_prices = []
    all_params = []

    stage1_batch_start = time.time()
    stage1_processed = 0
    for batch in _generate_batches_streaming(core_dias, turns_arr, heights, thicks, hvthicks, hvlens, core_len_steps, obround, batch_size=2_000_000):
        combs = torch.tensor(batch, device=device)
        prices, _ = _compute_batch_mps(combs, device, **params)

        # Get valid results
        valid_mask = prices < 1e17
        valid_prices = prices[valid_mask].cpu().numpy()
        valid_combs = combs[valid_mask].cpu().numpy()

        all_prices.extend(valid_prices.tolist())
        all_params.extend(valid_combs.tolist())

        # Progress with ETA for Stage 1
        stage1_processed += len(batch)
        stage1_pct = min(stage1_processed / n_coarse, 0.99) if n_coarse > 0 else 0.99
        stage1_elapsed = time.time() - stage1_batch_start
        if stage1_processed > 0 and stage1_processed < n_coarse:
            stage1_eta = (stage1_elapsed / stage1_processed) * (n_coarse - stage1_processed)
        else:
            stage1_eta = None
        report_progress(1, 0.01 + stage1_pct * 0.32, f"Stage 1: {stage1_processed:,}/{n_coarse:,}", stage1_eta)

        del combs, prices, batch

    if not all_prices:
        print("  No valid designs found in coarse search!")
        return None

    # Find top 5 candidates (different regions of parameter space)
    all_prices = np.array(all_prices)
    all_params = np.array(all_params)

    # Sort by price
    sorted_idx = np.argsort(all_prices)
    top_candidates = []

    for idx in sorted_idx:
        candidate = all_params[idx]
        price = all_prices[idx]

        # Check if this candidate is in a different region than existing ones
        is_unique_region = True
        for existing in top_candidates:
            # Different region = significant difference in core_dia or turns
            if abs(candidate[5] - existing[5]) < core_dia_step * 2 and abs(candidate[0] - existing[0]) < turns_step * 2:
                is_unique_region = False
                break

        if is_unique_region:
            top_candidates.append(candidate)
            print(f"  Top candidate {len(top_candidates)}: price={price:.2f}, core_dia={candidate[5]:.0f}, turns={candidate[0]:.0f}")

        if len(top_candidates) >= n_regions:
            break

    stage1_time = time.time() - total_start
    print(f"  Stage 1 completed in {stage1_time:.1f}s")
    report_progress(1, 0.33, f"Stage 1 complete: Found {len(top_candidates)} candidates", None)

    # =========================================================================
    # STAGE 2: FINE ZOOM - Search around each top candidate
    # =========================================================================
    print("\n[STAGE 2] Fine zoom into top candidates...")
    report_progress(2, 0.34, "Stage 2: Fine GPU zoom...", None)

    best_overall_price = 1e18
    best_overall_params = None

    # Track Stage 2 progress with ETA
    stage2_batch_start = time.time()
    stage2_total_processed = 0
    stage2_total_estimated = 0  # Will be estimated as we go

    for i, candidate in enumerate(top_candidates):
        cand_turns, cand_height, cand_thick, cand_hvthick, cand_hvlen, cand_core_dia, cand_core_len = candidate

        # Define search bounds around candidate (±1 coarse step)
        cd_min = max(CORE_MINIMUM, cand_core_dia - core_dia_step)
        cd_max = min(CORE_MAXIMUM, cand_core_dia + core_dia_step)
        turns_min = max(FOILTURNS_MINIMUM, cand_turns - turns_step)
        turns_max = min(FOILTURNS_MAXIMUM, cand_turns + turns_step)
        h_min = max(FOILHEIGHT_MINIMUM, cand_height - height_step)
        h_max = min(FOILHEIGHT_MAXIMUM, cand_height + height_step)
        th_min = max(FOILTHICKNESS_MINIMUM, cand_thick - thick_step)
        th_max = min(FOILTHICKNESS_MAXIMUM, cand_thick + thick_step)
        hvth_min = max(HVTHICK_MINIMUM, cand_hvthick - hvthick_step)
        hvth_max = min(HVTHICK_MAXIMUM, cand_hvthick + hvthick_step)
        hvl_min = max(HV_LEN_MINIMUM, cand_hvlen - hvlen_step)
        hvl_max = min(HV_LEN_MAXIMUM, cand_hvlen + hvlen_step)

        # Fine grid within this region (using fine_steps preset)
        fine_core_dias = np.arange(cd_min, cd_max + 1, fine_steps['core_dia'], dtype=np.float32)
        fine_turns = np.arange(turns_min, turns_max + 1, fine_steps['turns'], dtype=np.float32)
        fine_heights = np.arange(h_min, h_max + 1, fine_steps['height'], dtype=np.float32)
        fine_thicks = np.arange(th_min, th_max + 0.001, fine_steps['thick'], dtype=np.float32)
        fine_hvthicks = np.arange(hvth_min, hvth_max + 0.001, fine_steps['hvthick'], dtype=np.float32)
        fine_hvlens = np.arange(hvl_min, hvl_max + 0.001, fine_steps['hvlen'], dtype=np.float32)

        n_fine = _estimate_total_combinations(fine_core_dias, fine_turns, fine_heights, fine_thicks, fine_hvthicks, fine_hvlens, fine_core_len_steps, obround)
        print(f"  Region {i+1}: {n_fine:,} combinations around core_dia={cand_core_dia:.0f}, turns={cand_turns:.0f}")

        # Estimate total for all regions (assume similar size per region)
        if i == 0:
            stage2_total_estimated = n_fine * len(top_candidates)

        region_best_price = 1e18
        region_best_params = None
        region_processed = 0

        for batch in _generate_batches_streaming(fine_core_dias, fine_turns, fine_heights, fine_thicks, fine_hvthicks, fine_hvlens, fine_core_len_steps, obround, batch_size=2_000_000):
            combs = torch.tensor(batch, device=device)
            prices, _ = _compute_batch_mps(combs, device, **params)

            best_idx = torch.argmin(prices).item()
            best_price = prices[best_idx].item()

            if best_price < region_best_price:
                region_best_price = best_price
                region_best_params = combs[best_idx].cpu().numpy()

            # Update progress with ETA
            region_processed += len(batch)
            stage2_total_processed += len(batch)
            stage2_elapsed = time.time() - stage2_batch_start
            if stage2_total_processed > 0 and stage2_total_estimated > 0:
                stage2_pct = min(stage2_total_processed / stage2_total_estimated, 0.99)
                if stage2_total_processed < stage2_total_estimated:
                    stage2_eta = (stage2_elapsed / stage2_total_processed) * (stage2_total_estimated - stage2_total_processed)
                else:
                    stage2_eta = None
            else:
                stage2_pct = (i / len(top_candidates)) if len(top_candidates) > 0 else 0
                stage2_eta = None
            region_progress = 0.34 + stage2_pct * 0.33
            report_progress(2, region_progress, f"Stage 2: Region {i+1}/{len(top_candidates)}, {stage2_total_processed:,} processed", stage2_eta)

            del combs, prices, batch

        print(f"    Best in region: price={region_best_price:.2f}")

        if region_best_price < best_overall_price:
            best_overall_price = region_best_price
            best_overall_params = region_best_params

    stage2_time = time.time() - total_start - stage1_time
    print(f"  Stage 2 completed in {stage2_time:.1f}s")
    report_progress(2, 0.67, f"Stage 2 complete: Best GPU price = {best_overall_price:.2f}", None)

    if best_overall_params is None:
        print("No valid design found!")
        report_progress(3, 1.0, "No valid design found", None)
        return None

    # =========================================================================
    # STAGE 3: Local CPU search WITH cooling ducts (GPU doesn't include ducts)
    # =========================================================================
    print(f"\n[STAGE 3] Local CPU refinement WITH cooling ducts...")
    print(f"  (GPU price {best_overall_price:.2f} was calculated without cooling ducts)")
    report_progress(3, 0.68, "Stage 3: Local CPU refinement...", None)

    # GPU gives us starting point - now search locally WITH cooling ducts
    gpu_turns = int(round(best_overall_params[0]))
    gpu_height = float(best_overall_params[1])
    gpu_thick = float(best_overall_params[2])
    gpu_hvthick = float(best_overall_params[3])
    gpu_hvlen = float(best_overall_params[4])
    gpu_core_dia = float(best_overall_params[5])
    gpu_core_len = float(best_overall_params[6])

    # Local grid search around GPU result with proper cooling duct evaluation
    best_price = 1e18
    best_turns = gpu_turns
    best_height = gpu_height
    best_thick = gpu_thick
    best_hvthick = gpu_hvthick
    best_hvlen = gpu_hvlen
    best_core_dia = gpu_core_dia
    best_core_len = gpu_core_len

    # Search ranges around GPU result (using local_ranges and local_steps presets)
    lr = local_ranges
    ls = local_steps
    turns_range = range(max(FOILTURNS_MINIMUM, gpu_turns - lr['turns']), min(FOILTURNS_MAXIMUM, gpu_turns + lr['turns'] + 1))
    height_range = np.arange(max(FOILHEIGHT_MINIMUM, gpu_height - lr['height']), min(FOILHEIGHT_MAXIMUM, gpu_height + lr['height'] + 1), ls['height'])
    thick_range = np.arange(max(FOILTHICKNESS_MINIMUM, gpu_thick - lr['thick']), min(FOILTHICKNESS_MAXIMUM, gpu_thick + lr['thick'] + 0.001), ls['thick'])
    hvthick_range = np.arange(max(HVTHICK_MINIMUM, gpu_hvthick - lr['hvthick']), min(HVTHICK_MAXIMUM, gpu_hvthick + lr['hvthick'] + 0.001), ls['hvthick'])
    hvlen_range = np.arange(max(HV_LEN_MINIMUM, gpu_hvlen - lr['hvlen']), min(HV_LEN_MAXIMUM, gpu_hvlen + lr['hvlen'] + 0.001), ls['hvlen'])
    core_dia_range = np.arange(max(CORE_MINIMUM, gpu_core_dia - lr['core_dia']), min(CORE_MAXIMUM, gpu_core_dia + lr['core_dia'] + 1), ls['core_dia'])

    if obround:
        core_len_range = np.arange(max(0, gpu_core_len - lr['core_len']), min(gpu_core_dia, gpu_core_len + lr['core_len'] + 1), ls['core_len'])
    else:
        core_len_range = [0]

    n_local = len(turns_range) * len(height_range) * len(thick_range) * len(hvthick_range) * len(hvlen_range) * len(core_dia_range) * len(core_len_range)
    print(f"  Local search: {n_local:,} combinations (with cooling ducts)")

    stage3_start = time.time()

    # Use streaming for large searches (>10M combinations) to avoid memory issues
    if n_local > 10_000_000:
        print(f"  Using STREAMING mode (memory-efficient for large searches)")
        report_progress(3, 0.68, f"Stage 3: Streaming {n_local:,} combinations...", n_local / 50000)

        best_result = streaming_local_refinement(
            turns_range, height_range, thick_range, hvthick_range, hvlen_range,
            core_dia_range, core_len_range,
            LVRATE, HVRATE, POWERRATING, FREQUENCY,
            materialToBeUsedWire_Resistivity,
            GUARANTEED_NO_LOAD_LOSS, GUARANTEED_LOAD_LOSS, GUARANTEED_UCC,
            tolerance, put_cooling_ducts, HV_WIRE_CIRCULAR,
            progress_callback=progress_callback,
            chunk_size=100000
        )

        if best_result is not None and best_result[0] < best_price:
            best_price = best_result[0]
            best_turns = best_result[1]
            best_height = best_result[2]
            best_thick = best_result[3]
            best_hvthick = best_result[4]
            best_hvlen = best_result[5]
            best_core_dia = best_result[6]
            best_core_len = best_result[7]
    else:
        print(f"  Using BATCH mode (fast for smaller searches)")
        report_progress(3, 0.68, f"Stage 3: Processing {n_local:,} combinations...", None)

        # Generate all combinations as flattened arrays (OK for <10M combinations)
        turns_arr, height_arr, thick_arr, hvthick_arr, hvlen_arr, core_dia_arr, core_len_arr = \
            generate_local_combinations(turns_range, height_range, thick_range,
                                         hvthick_range, hvlen_range,
                                         core_dia_range, core_len_range)

        # Run parallel kernel with progress reporting
        results = parallel_local_refinement_with_progress(
            turns_arr, height_arr, thick_arr, hvthick_arr, hvlen_arr,
            core_dia_arr, core_len_arr,
            LVRATE, HVRATE, POWERRATING, FREQUENCY,
            materialToBeUsedWire_Resistivity,
            GUARANTEED_NO_LOAD_LOSS, GUARANTEED_LOAD_LOSS, GUARANTEED_UCC,
            tolerance, put_cooling_ducts, HV_WIRE_CIRCULAR,
            progress_callback=progress_callback,
            batch_size=2000
        )

        # Find best result (price is in column 0)
        valid_mask = (results[:, 0] > 0) & (results[:, 0] < 1e17)
        if np.any(valid_mask):
            valid_results = results[valid_mask]
            best_idx = np.argmin(valid_results[:, 0])
            best_result = valid_results[best_idx]

            if best_result[0] < best_price:
                best_price = best_result[0]
                best_turns = best_result[1]
                best_height = best_result[2]
                best_thick = best_result[3]
                best_hvthick = best_result[4]
                best_hvlen = best_result[5]
                best_core_dia = best_result[6]
                best_core_len = best_result[7]

    stage3_grid_time = time.time() - total_start - stage1_time - stage2_time
    print(f"  Stage 3 grid search completed in {stage3_grid_time:.1f}s")

    # Phase 2: Ensemble optimizer polish for maximum precision
    if best_price < 1e17:
        print(f"\n  [STAGE 3b] Ensemble optimizer polish...")
        print(f"  Starting from grid best: price={best_price:.2f}")
        report_progress(3, 0.92, "Stage 3: Ensemble optimization...", 60)

        polished_params, polished_price, winning_method = ensemble_optimize(
            [best_turns, best_height, best_thick, best_hvthick, best_hvlen, best_core_dia, best_core_len],
            tolerance, put_cooling_ducts,
            progress_callback
        )

        if polished_price < best_price:
            improvement = best_price - polished_price
            print(f"  {winning_method} improved price by {improvement:.2f} (new: {polished_price:.2f})")
            best_price = polished_price
            best_turns = int(round(polished_params[0]))
            best_height = polished_params[1]
            best_thick = polished_params[2]
            best_hvthick = polished_params[3]
            best_hvlen = polished_params[4]
            best_core_dia = polished_params[5]
            best_core_len = polished_params[6]
        else:
            print(f"  Grid result was already optimal")

    stage3_time = time.time() - total_start - stage1_time - stage2_time
    print(f"  Stage 3 total time: {stage3_time:.1f}s")

    if best_price >= 1e17:
        print(f"  No valid design found in local search!")
        # Fall back to GPU result
        best_turns = gpu_turns
        best_height = gpu_height
        best_thick = gpu_thick
        best_hvthick = gpu_hvthick
        best_hvlen = gpu_hvlen
        best_core_dia = gpu_core_dia
        best_core_len = gpu_core_len
        best_price = CalculateFinalizedPriceIntolerant_Optimized(
            best_turns, best_height, best_thick, best_hvthick, best_hvlen,
            best_core_dia, best_core_len,
            LVRATE, HVRATE, POWERRATING, FREQUENCY, materialToBeUsedWire_Resistivity,
            GUARANTEED_NO_LOAD_LOSS, GUARANTEED_LOAD_LOSS, GUARANTEED_UCC,
            tolerance, False, put_cooling_ducts, circular=HV_WIRE_CIRCULAR
        )
        if best_price < 0:
            best_price = abs(best_price)
    else:
        print(f"  Evaluated {n_local:,} designs, best price: {best_price:.2f}")

    total_time = time.time() - total_start
    report_progress(3, 1.0, f"Complete! Best price: {best_price:.2f}", None)

    if print_result:
        print(f"\n{'='*60}")
        print("HYBRID OPTIMIZATION RESULTS")
        print(f"{'='*60}")
        CalculateFinalizedPrice(best_turns, best_height, best_thick, best_hvthick, best_hvlen,
                                best_core_dia, best_core_len, LVRATE, HVRATE, POWERRATING, FREQUENCY,
                                materialToBeUsedWire_Resistivity, GUARANTEED_NO_LOAD_LOSS,
                                GUARANTEED_LOAD_LOSS, GUARANTEED_UCC, isFinal=True, PutCoolingDucts=put_cooling_ducts,
                                circular=HV_WIRE_CIRCULAR)
        print(f"\nCore Diameter: {best_core_dia:.1f} mm")
        print(f"Core Length: {best_core_len:.1f} mm")
        print(f"LV Turns: {best_turns:.0f}")
        print(f"LV Foil Height: {best_height:.1f} mm")
        print(f"LV Foil Thickness: {best_thick:.2f} mm")
        if HV_WIRE_CIRCULAR:
            print(f"HV Wire Diameter: {best_hvthick:.2f} mm")
        else:
            print(f"HV Wire Thickness: {best_hvthick:.2f} mm")
            print(f"HV Wire Length: {best_hvlen:.2f} mm")
        print(f"\nTotal optimization time: {total_time:.1f}s")
        print(f"  Stage 1 (coarse GPU): {stage1_time:.1f}s")
        print(f"  Stage 2 (fine GPU): {stage2_time:.1f}s")
        print(f"  Stage 3 (local CPU w/ ducts): {stage3_time:.1f}s")
        print(f"{'='*60}")

    return {
        'core_diameter': best_core_dia,
        'core_length': best_core_len,
        'lv_turns': best_turns,
        'lv_height': best_height,
        'lv_thickness': best_thick,
        'hv_thickness': best_hvthick,
        'hv_length': best_hvlen,
        'price': best_price,
        'time': total_time
    }


def StartCUDAHybrid(tolerance=25, obround=True, put_cooling_ducts=True, print_result=True, search_depth='normal', progress_callback=None):
    """
    CUDA GPU hybrid optimization - same algorithm as MPS but for NVIDIA GPUs.

    Stage 1: Coarse GPU search to find top candidates
    Stage 2: Fine GPU search in zoomed regions around best candidates
    Stage 3: Local CPU search WITH cooling ducts for final refinement
    """
    def report_progress(stage, progress, message, eta=None):
        if progress_callback:
            try:
                progress_callback(stage, progress, message, eta)
            except (InterruptedError, KeyboardInterrupt):
                raise
            except:
                pass

    if not CUDA_AVAILABLE:
        print("CUDA not available, falling back to CPU")
        return StartSmartOptimized(tolerance, obround, put_cooling_ducts, use_de=True, print_result=print_result)

    # Search depth presets (same as MPS)
    if search_depth == 'fast':
        coarse_steps = {'core_dia': 30, 'core_len': 3, 'turns': 5, 'height': 50, 'thick': 0.5, 'hvthick': 0.5, 'hvlen': 1.5}
        fine_steps = {'core_dia': 10, 'turns': 2, 'height': 20, 'thick': 0.15, 'hvthick': 0.15, 'hvlen': 0.4}
        local_ranges = {'turns': 2, 'height': 20, 'thick': 0.2, 'hvthick': 0.2, 'hvlen': 0.6, 'core_dia': 10, 'core_len': 15}
        local_steps = {'height': 10, 'thick': 0.1, 'hvthick': 0.1, 'hvlen': 0.2, 'core_dia': 5, 'core_len': 10}
        n_regions = 2
        fine_core_len_steps = 5
    elif search_depth == 'thorough':
        coarse_steps = {'core_dia': 20, 'core_len': 5, 'turns': 3, 'height': 30, 'thick': 0.3, 'hvthick': 0.3, 'hvlen': 0.8}
        fine_steps = {'core_dia': 4, 'turns': 1, 'height': 8, 'thick': 0.04, 'hvthick': 0.04, 'hvlen': 0.08}
        # Stage 3 local search - keep combinations manageable (~3-5M max)
        local_ranges = {'turns': 3, 'height': 30, 'thick': 0.3, 'hvthick': 0.3, 'hvlen': 1.0, 'core_dia': 15, 'core_len': 20}
        local_steps = {'height': 8, 'thick': 0.08, 'hvthick': 0.08, 'hvlen': 0.15, 'core_dia': 4, 'core_len': 6}
        n_regions = 5
        fine_core_len_steps = 12
    elif search_depth == 'exhaustive':
        coarse_steps = {'core_dia': 15, 'core_len': 6, 'turns': 2, 'height': 25, 'thick': 0.25, 'hvthick': 0.25, 'hvlen': 0.6}
        fine_steps = {'core_dia': 2, 'turns': 1, 'height': 5, 'thick': 0.02, 'hvthick': 0.02, 'hvlen': 0.05}
        # Stage 3 local search - high resolution for fine details (~10-20M combinations)
        local_ranges = {'turns': 5, 'height': 50, 'thick': 0.5, 'hvthick': 0.5, 'hvlen': 2.0, 'core_dia': 25, 'core_len': 40}
        local_steps = {'height': 4, 'thick': 0.04, 'hvthick': 0.04, 'hvlen': 0.08, 'core_dia': 3, 'core_len': 4}
        n_regions = 7
        fine_core_len_steps = 15
    else:  # normal
        coarse_steps = {'core_dia': 25, 'core_len': 4, 'turns': 4, 'height': 40, 'thick': 0.4, 'hvthick': 0.4, 'hvlen': 1.2}
        fine_steps = {'core_dia': 5, 'turns': 1, 'height': 10, 'thick': 0.05, 'hvthick': 0.05, 'hvlen': 0.1}
        local_ranges = {'turns': 3, 'height': 30, 'thick': 0.3, 'hvthick': 0.3, 'hvlen': 1.0, 'core_dia': 15, 'core_len': 20}
        local_steps = {'height': 10, 'thick': 0.05, 'hvthick': 0.05, 'hvlen': 0.1, 'core_dia': 5, 'core_len': 10}
        n_regions = 3
        fine_core_len_steps = 8

    print("=" * 60)
    print(f"CUDA HYBRID GPU OPTIMIZATION (depth: {search_depth})")
    print("Stage 1: Coarse GPU search for promising regions")
    print("Stage 2: Fine GPU zoom into best candidates")
    print("Stage 3: Local CPU search WITH cooling ducts")
    print("=" * 60)
    total_start = time.time()
    device = torch.device("cuda")

    report_progress(1, 0.0, "Starting CUDA optimization...", None)

    # STAGE 1: COARSE SEARCH
    print("\n[STAGE 1] Coarse CUDA GPU search...")
    report_progress(1, 0.01, "Stage 1: Coarse CUDA search...", None)

    core_dia_step = coarse_steps['core_dia']
    core_len_steps = coarse_steps['core_len']
    turns_step = coarse_steps['turns']
    height_step = coarse_steps['height']
    thick_step = coarse_steps['thick']
    hvthick_step = coarse_steps['hvthick']
    hvlen_step = coarse_steps['hvlen']

    core_dias = np.arange(CORE_MINIMUM, CORE_MAXIMUM + 1, core_dia_step, dtype=np.float32)
    turns_arr = np.arange(FOILTURNS_MINIMUM, FOILTURNS_MAXIMUM + 1, turns_step, dtype=np.float32)
    heights = np.arange(FOILHEIGHT_MINIMUM, FOILHEIGHT_MAXIMUM + 1, height_step, dtype=np.float32)
    thicks = np.arange(FOILTHICKNESS_MINIMUM, FOILTHICKNESS_MAXIMUM + 0.001, thick_step, dtype=np.float32)
    hvthicks = np.arange(HVTHICK_MINIMUM, HVTHICK_MAXIMUM + 0.001, hvthick_step, dtype=np.float32)
    hvlens = np.arange(HV_LEN_MINIMUM, HV_LEN_MAXIMUM + 0.001, hvlen_step, dtype=np.float32)

    n_coarse = _estimate_total_combinations(core_dias, turns_arr, heights, thicks, hvthicks, hvlens, core_len_steps, obround)
    print(f"  Coarse grid: {n_coarse:,} combinations")

    params = {
        'tolerance': tolerance, 'lv_rate': float(LVRATE), 'hv_rate': float(HVRATE),
        'power': float(POWERRATING), 'freq': float(FREQUENCY),
        'resistivity': float(materialToBeUsedWire_Resistivity),
        'guaranteed_nll': float(GUARANTEED_NO_LOAD_LOSS), 'guaranteed_ll': float(GUARANTEED_LOAD_LOSS),
        'guaranteed_ucc': float(GUARANTEED_UCC), 'ucc_tol': float(UCC_TOLERANCE),
        'dist_core_lv': float(DistanceCoreLV), 'main_gap': float(MainGap), 'phase_gap': float(PhaseGap),
        'insulation_wire': float(INSULATION_THICKNESS_WIRE),
        'lv_insulation_thick': float(LVInsulationThickness), 'hv_insulation_thick': float(HVInsulationThickness),
        'duct_thick': float(COOLING_DUCT_THICKNESS), 'core_density': float(CoreDensity),
        'core_fill_round': float(CoreFillingFactorRound), 'core_fill_rect': float(CoreFillingFactorRectangular),
        'core_price_val': float(CorePricePerKg), 'foil_density': float(materialToBeUsedFoil_Density),
        'foil_price': float(materialToBeUsedFoil_Price), 'wire_density': float(materialToBeUsedWire_Density),
        'wire_price': float(materialToBeUsedWire_Price), 'add_loss_lv': float(AdditionalLossFactorLV),
        'add_loss_hv': float(AdditionalLossFactorHV), 'penalty_nll': float(PENALTY_NLL_FACTOR),
        'penalty_ll': float(PENALTY_LL_FACTOR), 'penalty_ucc': float(PENALTY_UCC_FACTOR),
        'circular': HV_WIRE_CIRCULAR
    }

    all_prices = []
    all_params = []

    stage1_batch_start = time.time()
    stage1_processed = 0
    for batch in _generate_batches_streaming(core_dias, turns_arr, heights, thicks, hvthicks, hvlens, core_len_steps, obround, batch_size=2_000_000):
        combs = torch.tensor(batch, device=device)
        prices, _ = _compute_batch_mps(combs, device, **params)  # Works on CUDA too

        valid_mask = prices < 1e17
        valid_prices = prices[valid_mask].cpu().numpy()
        valid_combs = combs[valid_mask].cpu().numpy()

        all_prices.extend(valid_prices.tolist())
        all_params.extend(valid_combs.tolist())

        # Progress with ETA for Stage 1
        stage1_processed += len(batch)
        stage1_pct = min(stage1_processed / n_coarse, 0.99) if n_coarse > 0 else 0.99
        stage1_elapsed = time.time() - stage1_batch_start
        if stage1_processed > 0 and stage1_processed < n_coarse:
            stage1_eta = (stage1_elapsed / stage1_processed) * (n_coarse - stage1_processed)
        else:
            stage1_eta = None
        report_progress(1, 0.01 + stage1_pct * 0.32, f"Stage 1: {stage1_processed:,}/{n_coarse:,}", stage1_eta)

        del combs, prices, batch
        torch.cuda.empty_cache()

    if not all_prices:
        print("  No valid designs found in coarse search!")
        return None

    all_prices = np.array(all_prices)
    all_params = np.array(all_params)
    sorted_idx = np.argsort(all_prices)
    top_candidates = []

    for idx in sorted_idx:
        candidate = all_params[idx]
        price = all_prices[idx]
        is_unique_region = True
        for existing in top_candidates:
            if abs(candidate[5] - existing[5]) < core_dia_step * 2 and abs(candidate[0] - existing[0]) < turns_step * 2:
                is_unique_region = False
                break
        if is_unique_region:
            top_candidates.append(candidate)
            print(f"  Top candidate {len(top_candidates)}: price={price:.2f}, core_dia={candidate[5]:.0f}, turns={candidate[0]:.0f}")
        if len(top_candidates) >= n_regions:
            break

    stage1_time = time.time() - total_start
    print(f"  Stage 1 completed in {stage1_time:.1f}s")
    report_progress(1, 0.33, f"Stage 1 complete: Found {len(top_candidates)} candidates", None)

    # STAGE 2: FINE ZOOM
    print("\n[STAGE 2] Fine zoom into top candidates...")
    report_progress(2, 0.34, "Stage 2: Fine CUDA zoom...", None)

    best_overall_price = 1e18
    best_overall_params = None

    # Track Stage 2 progress with ETA
    stage2_batch_start = time.time()
    stage2_total_processed = 0
    stage2_total_estimated = 0

    for i, candidate in enumerate(top_candidates):
        cand_turns, cand_height, cand_thick, cand_hvthick, cand_hvlen, cand_core_dia, cand_core_len = candidate

        cd_min = max(CORE_MINIMUM, cand_core_dia - core_dia_step)
        cd_max = min(CORE_MAXIMUM, cand_core_dia + core_dia_step)
        turns_min = max(FOILTURNS_MINIMUM, cand_turns - turns_step)
        turns_max = min(FOILTURNS_MAXIMUM, cand_turns + turns_step)
        h_min = max(FOILHEIGHT_MINIMUM, cand_height - height_step)
        h_max = min(FOILHEIGHT_MAXIMUM, cand_height + height_step)
        th_min = max(FOILTHICKNESS_MINIMUM, cand_thick - thick_step)
        th_max = min(FOILTHICKNESS_MAXIMUM, cand_thick + thick_step)
        hvth_min = max(HVTHICK_MINIMUM, cand_hvthick - hvthick_step)
        hvth_max = min(HVTHICK_MAXIMUM, cand_hvthick + hvthick_step)
        hvl_min = max(HV_LEN_MINIMUM, cand_hvlen - hvlen_step)
        hvl_max = min(HV_LEN_MAXIMUM, cand_hvlen + hvlen_step)

        fine_core_dias = np.arange(cd_min, cd_max + 1, fine_steps['core_dia'], dtype=np.float32)
        fine_turns = np.arange(turns_min, turns_max + 1, fine_steps['turns'], dtype=np.float32)
        fine_heights = np.arange(h_min, h_max + 1, fine_steps['height'], dtype=np.float32)
        fine_thicks = np.arange(th_min, th_max + 0.001, fine_steps['thick'], dtype=np.float32)
        fine_hvthicks = np.arange(hvth_min, hvth_max + 0.001, fine_steps['hvthick'], dtype=np.float32)
        fine_hvlens = np.arange(hvl_min, hvl_max + 0.001, fine_steps['hvlen'], dtype=np.float32)

        n_fine = _estimate_total_combinations(fine_core_dias, fine_turns, fine_heights, fine_thicks, fine_hvthicks, fine_hvlens, fine_core_len_steps, obround)
        print(f"  Region {i+1}: {n_fine:,} combinations around core_dia={cand_core_dia:.0f}, turns={cand_turns:.0f}")

        # Estimate total for all regions (assume similar size per region)
        if i == 0:
            stage2_total_estimated = n_fine * len(top_candidates)

        region_best_price = 1e18
        region_best_params = None
        region_processed = 0

        for batch in _generate_batches_streaming(fine_core_dias, fine_turns, fine_heights, fine_thicks, fine_hvthicks, fine_hvlens, fine_core_len_steps, obround, batch_size=2_000_000):
            combs = torch.tensor(batch, device=device)
            prices, _ = _compute_batch_mps(combs, device, **params)

            best_idx = torch.argmin(prices).item()
            best_price = prices[best_idx].item()

            if best_price < region_best_price:
                region_best_price = best_price
                region_best_params = combs[best_idx].cpu().numpy()

            # Update progress with ETA
            region_processed += len(batch)
            stage2_total_processed += len(batch)
            stage2_elapsed = time.time() - stage2_batch_start
            if stage2_total_processed > 0 and stage2_total_estimated > 0:
                stage2_pct = min(stage2_total_processed / stage2_total_estimated, 0.99)
                if stage2_total_processed < stage2_total_estimated:
                    stage2_eta = (stage2_elapsed / stage2_total_processed) * (stage2_total_estimated - stage2_total_processed)
                else:
                    stage2_eta = None
            else:
                stage2_pct = (i / len(top_candidates)) if len(top_candidates) > 0 else 0
                stage2_eta = None
            region_progress = 0.34 + stage2_pct * 0.33
            report_progress(2, region_progress, f"Stage 2: Region {i+1}/{len(top_candidates)}, {stage2_total_processed:,} processed", stage2_eta)

            del combs, prices, batch
            torch.cuda.empty_cache()

        print(f"    Best in region: price={region_best_price:.2f}")

        if region_best_price < best_overall_price:
            best_overall_price = region_best_price
            best_overall_params = region_best_params

    stage2_time = time.time() - total_start - stage1_time
    print(f"  Stage 2 completed in {stage2_time:.1f}s")
    report_progress(2, 0.67, f"Stage 2 complete: Best GPU price = {best_overall_price:.2f}", None)

    if best_overall_params is None:
        print("No valid design found!")
        report_progress(3, 1.0, "No valid design found", None)
        return None

    # STAGE 3: LOCAL CPU REFINEMENT
    print(f"\n[STAGE 3] Local CPU refinement WITH cooling ducts...")
    report_progress(3, 0.68, "Stage 3: Local CPU refinement...", None)

    gpu_turns = int(round(best_overall_params[0]))
    gpu_height = float(best_overall_params[1])
    gpu_thick = float(best_overall_params[2])
    gpu_hvthick = float(best_overall_params[3])
    gpu_hvlen = float(best_overall_params[4])
    gpu_core_dia = float(best_overall_params[5])
    gpu_core_len = float(best_overall_params[6])

    best_price = 1e18
    best_turns = gpu_turns
    best_height = gpu_height
    best_thick = gpu_thick
    best_hvthick = gpu_hvthick
    best_hvlen = gpu_hvlen
    best_core_dia = gpu_core_dia
    best_core_len = gpu_core_len

    lr = local_ranges
    ls = local_steps
    turns_range = range(max(FOILTURNS_MINIMUM, gpu_turns - lr['turns']), min(FOILTURNS_MAXIMUM, gpu_turns + lr['turns'] + 1))
    height_range = np.arange(max(FOILHEIGHT_MINIMUM, gpu_height - lr['height']), min(FOILHEIGHT_MAXIMUM, gpu_height + lr['height'] + 1), ls['height'])
    thick_range = np.arange(max(FOILTHICKNESS_MINIMUM, gpu_thick - lr['thick']), min(FOILTHICKNESS_MAXIMUM, gpu_thick + lr['thick'] + 0.001), ls['thick'])
    hvthick_range = np.arange(max(HVTHICK_MINIMUM, gpu_hvthick - lr['hvthick']), min(HVTHICK_MAXIMUM, gpu_hvthick + lr['hvthick'] + 0.001), ls['hvthick'])
    hvlen_range = np.arange(max(HV_LEN_MINIMUM, gpu_hvlen - lr['hvlen']), min(HV_LEN_MAXIMUM, gpu_hvlen + lr['hvlen'] + 0.001), ls['hvlen'])
    core_dia_range = np.arange(max(CORE_MINIMUM, gpu_core_dia - lr['core_dia']), min(CORE_MAXIMUM, gpu_core_dia + lr['core_dia'] + 1), ls['core_dia'])

    if obround:
        core_len_range = np.arange(max(0, gpu_core_len - lr['core_len']), min(gpu_core_dia, gpu_core_len + lr['core_len'] + 1), ls['core_len'])
    else:
        core_len_range = [0]

    n_local = len(turns_range) * len(height_range) * len(thick_range) * len(hvthick_range) * len(hvlen_range) * len(core_dia_range) * len(core_len_range)
    print(f"  Local search: {n_local:,} combinations (with cooling ducts)")

    stage3_start = time.time()

    # Use streaming for large searches (>10M combinations) to avoid memory issues
    if n_local > 10_000_000:
        print(f"  Using STREAMING mode (memory-efficient for large searches)")
        report_progress(3, 0.68, f"Stage 3: Streaming {n_local:,} combinations...", n_local / 50000)

        best_result = streaming_local_refinement(
            turns_range, height_range, thick_range, hvthick_range, hvlen_range,
            core_dia_range, core_len_range,
            LVRATE, HVRATE, POWERRATING, FREQUENCY,
            materialToBeUsedWire_Resistivity,
            GUARANTEED_NO_LOAD_LOSS, GUARANTEED_LOAD_LOSS, GUARANTEED_UCC,
            tolerance, put_cooling_ducts, HV_WIRE_CIRCULAR,
            progress_callback=progress_callback,
            chunk_size=100000
        )

        if best_result is not None and best_result[0] < best_price:
            best_price = best_result[0]
            best_turns = best_result[1]
            best_height = best_result[2]
            best_thick = best_result[3]
            best_hvthick = best_result[4]
            best_hvlen = best_result[5]
            best_core_dia = best_result[6]
            best_core_len = best_result[7]
    else:
        print(f"  Using BATCH mode (fast for smaller searches)")
        report_progress(3, 0.68, f"Stage 3: Processing {n_local:,} combinations...", None)

        # Generate all combinations as flattened arrays (OK for <10M combinations)
        turns_arr, height_arr, thick_arr, hvthick_arr, hvlen_arr, core_dia_arr, core_len_arr = \
            generate_local_combinations(turns_range, height_range, thick_range,
                                         hvthick_range, hvlen_range,
                                         core_dia_range, core_len_range)

        # Run parallel kernel with progress reporting
        results = parallel_local_refinement_with_progress(
            turns_arr, height_arr, thick_arr, hvthick_arr, hvlen_arr,
            core_dia_arr, core_len_arr,
            LVRATE, HVRATE, POWERRATING, FREQUENCY,
            materialToBeUsedWire_Resistivity,
            GUARANTEED_NO_LOAD_LOSS, GUARANTEED_LOAD_LOSS, GUARANTEED_UCC,
            tolerance, put_cooling_ducts, HV_WIRE_CIRCULAR,
            progress_callback=progress_callback,
            batch_size=2000
        )

        # Find best result (price is in column 0)
        valid_mask = (results[:, 0] > 0) & (results[:, 0] < 1e17)
        if np.any(valid_mask):
            valid_results = results[valid_mask]
            best_idx = np.argmin(valid_results[:, 0])
            best_result = valid_results[best_idx]

            if best_result[0] < best_price:
                best_price = best_result[0]
                best_turns = best_result[1]
                best_height = best_result[2]
                best_thick = best_result[3]
                best_hvthick = best_result[4]
                best_hvlen = best_result[5]
                best_core_dia = best_result[6]
                best_core_len = best_result[7]

    stage3_grid_time = time.time() - total_start - stage1_time - stage2_time
    print(f"  Stage 3 grid search completed in {stage3_grid_time:.1f}s")

    # Phase 2: Ensemble optimizer polish for maximum precision
    if best_price < 1e17:
        print(f"\n  [STAGE 3b] Ensemble optimizer polish...")
        print(f"  Starting from grid best: price={best_price:.2f}")
        report_progress(3, 0.92, "Stage 3: Ensemble optimization...", 60)

        polished_params, polished_price, winning_method = ensemble_optimize(
            [best_turns, best_height, best_thick, best_hvthick, best_hvlen, best_core_dia, best_core_len],
            tolerance, put_cooling_ducts,
            progress_callback
        )

        if polished_price < best_price:
            improvement = best_price - polished_price
            print(f"  {winning_method} improved price by {improvement:.2f} (new: {polished_price:.2f})")
            best_price = polished_price
            best_turns = int(round(polished_params[0]))
            best_height = polished_params[1]
            best_thick = polished_params[2]
            best_hvthick = polished_params[3]
            best_hvlen = polished_params[4]
            best_core_dia = polished_params[5]
            best_core_len = polished_params[6]
        else:
            print(f"  Grid result was already optimal")

    stage3_time = time.time() - total_start - stage1_time - stage2_time
    print(f"  Stage 3 total time: {stage3_time:.1f}s")

    if best_price >= 1e17:
        print(f"  No valid design found in local search!")
        best_turns, best_height, best_thick = gpu_turns, gpu_height, gpu_thick
        best_hvthick, best_hvlen = gpu_hvthick, gpu_hvlen
        best_core_dia, best_core_len = gpu_core_dia, gpu_core_len
        best_price = CalculateFinalizedPriceIntolerant_Optimized(
            best_turns, best_height, best_thick, best_hvthick, best_hvlen,
            best_core_dia, best_core_len,
            LVRATE, HVRATE, POWERRATING, FREQUENCY, materialToBeUsedWire_Resistivity,
            GUARANTEED_NO_LOAD_LOSS, GUARANTEED_LOAD_LOSS, GUARANTEED_UCC,
            tolerance, False, put_cooling_ducts, circular=HV_WIRE_CIRCULAR
        )
        if best_price < 0:
            best_price = abs(best_price)
    else:
        print(f"  Evaluated {n_local:,} designs, best price: {best_price:.2f}")

    total_time = time.time() - total_start
    report_progress(3, 1.0, f"Complete! Best price: {best_price:.2f}", None)

    if print_result:
        print(f"\n{'='*60}")
        print("CUDA HYBRID OPTIMIZATION RESULTS")
        print(f"{'='*60}")
        CalculateFinalizedPrice(best_turns, best_height, best_thick, best_hvthick, best_hvlen,
                                best_core_dia, best_core_len, LVRATE, HVRATE, POWERRATING, FREQUENCY,
                                materialToBeUsedWire_Resistivity, GUARANTEED_NO_LOAD_LOSS,
                                GUARANTEED_LOAD_LOSS, GUARANTEED_UCC, isFinal=True, PutCoolingDucts=put_cooling_ducts,
                                circular=HV_WIRE_CIRCULAR)
        print(f"\nCore Diameter: {best_core_dia:.1f} mm")
        print(f"Core Length: {best_core_len:.1f} mm")
        print(f"LV Turns: {best_turns:.0f}")
        print(f"LV Foil Height: {best_height:.1f} mm")
        print(f"LV Foil Thickness: {best_thick:.2f} mm")
        if HV_WIRE_CIRCULAR:
            print(f"HV Wire Diameter: {best_hvthick:.2f} mm")
        else:
            print(f"HV Wire Thickness: {best_hvthick:.2f} mm")
            print(f"HV Wire Length: {best_hvlen:.2f} mm")
        print(f"\nTotal optimization time: {total_time:.1f}s")
        print(f"  Stage 1 (coarse CUDA): {stage1_time:.1f}s")
        print(f"  Stage 2 (fine CUDA): {stage2_time:.1f}s")
        print(f"  Stage 3 (local CPU w/ ducts): {stage3_time:.1f}s")
        print(f"{'='*60}")

    return {
        'core_diameter': best_core_dia,
        'core_length': best_core_len,
        'lv_turns': best_turns,
        'lv_height': best_height,
        'lv_thickness': best_thick,
        'hv_thickness': best_hvthick,
        'hv_length': best_hvlen,
        'price': best_price,
        'time': total_time
    }


def _format_time(seconds):
    """Format seconds into human-readable time string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def _calculate_total_combinations(core_dias, turns, heights, thicks, hvthicks, hvlens, obround):
    """Calculate total number of parameter combinations for progress tracking."""
    n_core = len(core_dias)
    n_turns = len(turns)
    n_height = len(heights)
    n_thick = len(thicks)
    n_hvthick = len(hvthicks)
    n_hvlen = len(hvlens)

    # Base combinations per core diameter (without core length variations)
    base_combinations = n_turns * n_height * n_thick * n_hvthick * n_hvlen

    if obround:
        # Each core diameter has different core_length iterations
        # core_len goes from 0 to core_dia with step core_dia/10
        # So approximately 10 iterations per core diameter
        total = 0
        for core_dia in core_dias:
            core_len_steps = max(1, int(core_dia / max(core_dia / 10.0, 1.0)) + 1)
            total += base_combinations * core_len_steps
        return total
    else:
        return n_core * base_combinations


def StartOptimized(tolerance=25, obround=True, put_cooling_ducts=True, print_result=True, batch_size=10, grid_resolution='coarse', progress_callback=None):
    """Optimized entry point using parallel grid search with progress tracking.

    Args:
        tolerance: Tolerance percentage for constraints
        obround: Whether core is obround shape
        put_cooling_ducts: Whether to calculate cooling ducts
        print_result: Print detailed results
        batch_size: Number of core diameters to process per batch (for progress updates)
        grid_resolution: 'coarse' (fast ~5-15s), 'medium' (~30-60s), 'fine' (thorough, slow)
        progress_callback: Optional function(stage, progress, message, eta) for UI updates
    """
    print("Starting optimized parallel grid search...")
    start_time = time.time()

    # Grid resolution presets - dramatically affects speed
    if grid_resolution == 'coarse':
        # Fast: ~5-15 seconds
        core_dia_step = 20 if obround else 5
        turns_step = 3
        height_step = 25
        thick_step = 0.2
        hvthick_step = 0.2
        hvlen_step = 0.5
        print("Grid: COARSE (fast mode)")
    elif grid_resolution == 'fine':
        # Thorough but slow
        core_dia_step = 5 if obround else 1
        turns_step = 1
        height_step = 5
        thick_step = 0.05
        hvthick_step = 0.05
        hvlen_step = 0.1
        print("Grid: FINE (thorough mode - this will take a while)")
    else:  # medium
        # Balanced: ~30-60 seconds
        core_dia_step = 10 if obround else 2
        turns_step = 2
        height_step = 10
        thick_step = 0.1
        hvthick_step = 0.1
        hvlen_step = 0.25
        print("Grid: MEDIUM (balanced mode)")

    core_dias = np.arange(CORE_MINIMUM, CORE_MAXIMUM, core_dia_step, dtype=np.float64)
    turns = np.arange(FOILTURNS_MINIMUM, FOILTURNS_MAXIMUM, turns_step, dtype=np.float64)
    heights = np.arange(FOILHEIGHT_MINIMUM, FOILHEIGHT_MAXIMUM, height_step, dtype=np.float64)
    thicks = np.arange(FOILTHICKNESS_MINIMUM, FOILTHICKNESS_MAXIMUM, thick_step, dtype=np.float64)
    hvthicks = np.arange(HVTHICK_MINIMUM, HVTHICK_MAXIMUM, hvthick_step, dtype=np.float64)
    hvlens = np.arange(HV_LEN_MINIMUM, HV_LEN_MAXIMUM, hvlen_step, dtype=np.float64)
    core_lens = np.array([0.0], dtype=np.float64)

    # Calculate total combinations for progress tracking
    total_combinations = _calculate_total_combinations(core_dias, turns, heights, thicks, hvthicks, hvlens, obround)
    print(f"Total parameter combinations to evaluate: {total_combinations:,}")

    # Process in batches for progress updates
    n_core = len(core_dias)
    n_batches = max(1, (n_core + batch_size - 1) // batch_size)

    all_results = []
    completed_cores = 0

    print(f"\nProcessing {n_core} core diameters in {n_batches} batches...")
    print("-" * 50)

    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, n_core)
        batch_core_dias = core_dias[batch_start:batch_end]

        batch_results = parallel_grid_search_kernel(
            batch_core_dias, core_lens, turns, heights, thicks, hvthicks, hvlens,
            float(tolerance), obround, put_cooling_ducts, HV_WIRE_CIRCULAR,
            LVRATE, HVRATE, POWERRATING, FREQUENCY, materialToBeUsedWire_Resistivity,
            GUARANTEED_NO_LOAD_LOSS, GUARANTEED_LOAD_LOSS, GUARANTEED_UCC,
            CoreFillingFactorRound, CoreFillingFactorRectangular, INSULATION_THICKNESS_WIRE
        )
        all_results.append(batch_results)

        # Update progress
        completed_cores += len(batch_core_dias)
        progress_pct = (completed_cores / n_core) * 100
        elapsed = time.time() - start_time

        if completed_cores < n_core and elapsed > 0:
            rate = completed_cores / elapsed  # cores per second
            remaining_cores = n_core - completed_cores
            eta_seconds = remaining_cores / rate if rate > 0 else 0
            eta_str = _format_time(eta_seconds)
            print(f"Progress: {progress_pct:5.1f}% | Completed: {completed_cores}/{n_core} | "
                  f"Elapsed: {_format_time(elapsed)} | ETA: {eta_str}")
            # Call progress callback
            if progress_callback:
                try:
                    progress_callback("CPU Parallel", progress_pct / 100.0,
                                    f"Batch {batch_idx+1}/{n_batches}: {completed_cores}/{n_core} cores",
                                    eta_seconds)
                except (InterruptedError, KeyboardInterrupt):
                    raise
                except:
                    pass
        else:
            print(f"Progress: {progress_pct:5.1f}% | Completed: {completed_cores}/{n_core} | "
                  f"Elapsed: {_format_time(elapsed)}")
            # Call progress callback
            if progress_callback:
                try:
                    progress_callback("CPU Parallel", progress_pct / 100.0,
                                    f"Batch {batch_idx+1}/{n_batches}: {completed_cores}/{n_core} cores",
                                    None)
                except (InterruptedError, KeyboardInterrupt):
                    raise
                except:
                    pass

    print("-" * 50)

    # Combine all batch results
    results = np.vstack(all_results)

    valid_mask = results[:, 8] == 1.0
    if not np.any(valid_mask):
        print("No valid design found!")
        return None

    valid_results = results[valid_mask]
    best_idx = np.argmin(valid_results[:, 0])
    best = valid_results[best_idx]

    end_time = time.time()
    total_time = end_time - start_time

    if print_result:
        print("=" * 50)
        print(f"OPTIMIZED SEARCH COMPLETED in {_format_time(total_time)}")
        print(f"Evaluated {total_combinations:,} combinations")
        print(f"Speed: {total_combinations / total_time:,.0f} evaluations/second")
        print("=" * 50)
        print(f"Best Price: {best[0]:.2f}")
        print(f"Core Diameter: {best[6]:.1f}")
        print(f"Core Length: {best[7]:.1f}")
        print(f"LV Turns: {best[1]:.0f}")
        print(f"LV Height: {best[2]:.1f}")
        print(f"LV Thickness: {best[3]:.2f}")
        print(f"HV Thickness: {best[4]:.2f}")
        print(f"HV Length: {best[5]:.2f}")
        print("=" * 50)
        print("\nVerifying with detailed calculation:")
        CalculateFinalizedPrice(
            int(best[1]), best[2], best[3], best[4], best[5], best[6], best[7],
            LVRATE, HVRATE, POWERRATING, FREQUENCY, materialToBeUsedWire_Resistivity,
            GUARANTEED_NO_LOAD_LOSS, GUARANTEED_LOAD_LOSS, GUARANTEED_UCC,
            isFinal=True, PutCoolingDucts=put_cooling_ducts, circular=HV_WIRE_CIRCULAR
        )

    return _build_result_dict(
        lv_turns=best[1], lv_height=best[2], lv_thickness=best[3],
        hv_thickness=best[4], hv_length=best[5],
        core_diameter=best[6], core_length=best[7],
        total_price=best[0], elapsed_time=total_time, put_cooling_ducts=put_cooling_ducts,
        circular=HV_WIRE_CIRCULAR
    )


# =============================================================================
# SCIPY-BASED OPTIMIZATION
# =============================================================================

def objective_for_scipy(x, put_cooling_ducts=True, tolerance=25):
    """Objective function wrapper for scipy optimizers.

    Uses current module-level values for LVRATE, HVRATE, etc.
    """
    lv_turns = int(round(x[0]))
    lv_height = x[1]
    lv_thickness = x[2]
    hv_thickness = x[3]
    hv_length = x[4]
    core_diameter = x[5]
    core_length = x[6]

    # Basic validity checks
    if lv_turns < 5:
        return 1e12
    # For rectangular wire, hv_length must be >= hv_thickness
    if not HV_WIRE_CIRCULAR and hv_length < hv_thickness:
        return 1e12
    # Core length must not exceed core diameter
    if core_length > core_diameter:
        return 1e12

    try:
        # Pass current module values explicitly (may have been changed by UI)
        price = CalculateFinalizedPriceIntolerant_Optimized(
            lv_turns, lv_height, lv_thickness, hv_thickness, hv_length,
            core_diameter, core_length,
            LVRATE_VAL=LVRATE, HVRATE_VAL=HVRATE, POWER=POWERRATING, FREQ=FREQUENCY,
            MaterialResistivity=materialToBeUsedWire_Resistivity,
            GUARANTEEDNLL=GUARANTEED_NO_LOAD_LOSS, GUARANTEEDLL=GUARANTEED_LOAD_LOSS,
            GUARANTEEDUCC=GUARANTEED_UCC,
            tolerance=tolerance, PutCoolingDucts=put_cooling_ducts, circular=HV_WIRE_CIRCULAR
        )
        if price < 0:
            return abs(price) + 1e6
        return price
    except:
        return 1e12


def get_parameter_bounds(obround=True):
    """Return parameter bounds for scipy optimizers."""
    core_len_max = CORE_MAXIMUM if obround else 0.1
    return [
        (FOILTURNS_MINIMUM, FOILTURNS_MAXIMUM),
        (FOILHEIGHT_MINIMUM, FOILHEIGHT_MAXIMUM),
        (FOILTHICKNESS_MINIMUM, FOILTHICKNESS_MAXIMUM),
        (HVTHICK_MINIMUM, HVTHICK_MAXIMUM),
        (HV_LEN_MINIMUM, HV_LEN_MAXIMUM),
        (CORE_MINIMUM, CORE_MAXIMUM),
        (CORELENGTH_MINIMUM, core_len_max),
    ]


def smart_coarse_search(n_samples=3000, obround=True, put_cooling_ducts=True, tolerance=25):
    """Use Latin Hypercube Sampling to efficiently sample the parameter space."""
    if not SCIPY_AVAILABLE:
        print("scipy not available, falling back to parallel grid search")
        return None

    print(f"Running Latin Hypercube Sampling with {n_samples} samples...")
    start_time = time.time()

    bounds = get_parameter_bounds(obround)
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])

    sampler = qmc.LatinHypercube(d=7,seed=42)
    samples = sampler.random(n=n_samples)
    scaled_samples = qmc.scale(samples, lower, upper)

    best_price = 1e18
    best_point = None

    for i, sample in enumerate(scaled_samples):
        price = objective_for_scipy(sample, put_cooling_ducts, tolerance)
        if 0 < price < best_price:
            best_price = price
            best_point = sample.copy()

        if (i + 1) % 500 == 0:
            print(f"  Evaluated {i+1}/{n_samples} samples, best so far: {best_price:.2f}")

    elapsed = time.time() - start_time
    print(f"Coarse search completed in {elapsed:.2f}s, best price: {best_price:.2f}")

    return best_point, best_price


def fine_search_scipy(initial_point, obround=True, put_cooling_ducts=True, tolerance=25, method='L-BFGS-B'):
    """Use scipy.optimize for fine-tuning around the initial point."""
    if not SCIPY_AVAILABLE:
        return initial_point, objective_for_scipy(initial_point, put_cooling_ducts, tolerance)

    print(f"Running fine search with {method}...")
    start_time = time.time()

    bounds = get_parameter_bounds(obround)

    result = minimize(
        fun=lambda x: objective_for_scipy(x, put_cooling_ducts, tolerance),
        x0=initial_point,
        method=method,
        bounds=bounds,
        options={'maxiter': 100, 'disp': False}
    )

    elapsed = time.time() - start_time
    print(f"Fine search completed in {elapsed:.2f}s, final price: {result.fun:.2f}")

    return result.x, result.fun


def global_search_de(obround=True, put_cooling_ducts=True, tolerance=25, maxiter=50, popsize=10, progress_callback=None):
    """Use Differential Evolution for global optimization."""
    if not SCIPY_AVAILABLE:
        print("scipy not available, falling back to parallel grid search")
        return StartOptimized(tolerance, obround, put_cooling_ducts)

    def report_progress(progress, message, eta=None):
        if progress_callback:
            try:
                progress_callback("DE", progress, message, eta)
            except:
                pass

    print("Running Differential Evolution global search...")
    start_time = time.time()
    report_progress(0.0, "Starting Differential Evolution...", None)

    bounds = get_parameter_bounds(obround)

    # Callback for iteration progress
    iteration_count = [0]  # Use list for mutable in closure

    def de_callback(xk, convergence):
        iteration_count[0] += 1
        elapsed = time.time() - start_time
        progress = min(iteration_count[0] / maxiter, 0.99)
        if iteration_count[0] > 1 and iteration_count[0] < maxiter:
            eta = (elapsed / iteration_count[0]) * (maxiter - iteration_count[0])
        else:
            eta = None
        report_progress(progress, f"DE iteration {iteration_count[0]}/{maxiter}", eta)

    result = differential_evolution(
        func=lambda x: objective_for_scipy(x, put_cooling_ducts, tolerance),
        bounds=bounds,
        maxiter=maxiter,
        popsize=popsize,
        workers=1,
        polish=True,
        seed=42,
        disp=True,
        callback=de_callback
    )

    elapsed = time.time() - start_time
    print(f"DE completed in {elapsed:.2f}s")
    report_progress(1.0, f"DE completed in {elapsed:.2f}s", None)

    best = result.x
    return {
        "price": result.fun,
        "corediameter": best[5],
        "corelength": best[6],
        "lvfoilturns": int(round(best[0])),
        "lvfoilheight": best[1],
        "lvfoilthickness": best[2],
        "hvthickness": best[3],
        "hvlength": best[4]
    }


def global_search_de_multiseed(obround=True, put_cooling_ducts=True, tolerance=25, maxiter=50, popsize=10,
                                n_seeds=5, progress_callback=None):
    """
    Multi-seed Differential Evolution for more robust global optimization.

    Runs multiple DE optimizations with different predetermined seeds in parallel,
    then keeps the best (cheapest) result. This helps escape local minima.

    Args:
        obround: Whether core is obround shape
        put_cooling_ducts: Whether to calculate cooling ducts
        tolerance: Tolerance percentage for constraints
        maxiter: Maximum iterations per DE run
        popsize: Population size per DE run
        n_seeds: Number of different seeds to try (default 5)
        progress_callback: Optional function(stage, progress, message, eta) for UI updates

    Returns:
        Dictionary with optimal parameters
    """
    if not SCIPY_AVAILABLE:
        print("scipy not available, falling back to parallel grid search")
        return StartOptimized(tolerance, obround, put_cooling_ducts)

    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Predetermined seeds for reproducibility (up to 15)
    ALL_SEEDS = [42, 123, 456, 789, 1001, 2023, 3141, 5926, 8675, 3090, 1337, 2718, 6174, 9999, 4242]
    SEEDS = ALL_SEEDS[:min(n_seeds, 15)]

    print(f"Running Multi-Seed Differential Evolution ({n_seeds} seeds)...")
    print(f"Seeds: {SEEDS}")
    start_time = time.time()

    bounds = get_parameter_bounds(obround)

    def report_progress(stage, progress, message, eta=None):
        if progress_callback:
            try:
                progress_callback(stage, progress, message, eta)
            except:
                pass

    def run_single_de(seed_idx, seed):
        """Run a single DE optimization with given seed."""
        try:
            result = differential_evolution(
                func=lambda x: objective_for_scipy(x, put_cooling_ducts, tolerance),
                bounds=bounds,
                maxiter=maxiter,
                popsize=popsize,
                workers=1,
                polish=True,
                seed=seed,
                disp=False  # Quiet mode for parallel runs
            )
            return seed_idx, seed, result.fun, result.x
        except Exception as e:
            print(f"  Seed {seed} failed: {e}")
            return seed_idx, seed, float('inf'), None

    report_progress("Multi-DE", 0.0, f"Starting {n_seeds} parallel DE runs...", None)

    # Run all seeds in parallel using ThreadPoolExecutor
    results = []
    completed = 0

    with ThreadPoolExecutor(max_workers=min(n_seeds, 4)) as executor:
        futures = {executor.submit(run_single_de, i, seed): (i, seed) for i, seed in enumerate(SEEDS)}

        for future in as_completed(futures):
            seed_idx, seed = futures[future]
            try:
                result = future.result()
                results.append(result)
                completed += 1

                _, s, price, _ = result
                if price < float('inf'):
                    print(f"  Seed {s}: price = {price:.2f}")

                progress = completed / n_seeds
                elapsed = time.time() - start_time
                eta = (elapsed / completed) * (n_seeds - completed) if completed > 0 else None
                report_progress("Multi-DE", progress * 0.9, f"Completed {completed}/{n_seeds} seeds", eta)

            except Exception as e:
                print(f"  Seed {seed} exception: {e}")
                completed += 1

    # Find the best result
    valid_results = [(idx, seed, price, x) for idx, seed, price, x in results if x is not None and price < float('inf')]

    if not valid_results:
        print("All DE runs failed!")
        return None

    # Sort by price and get the best
    valid_results.sort(key=lambda r: r[2])
    best_idx, best_seed, best_price, best_x = valid_results[0]

    elapsed = time.time() - start_time
    print(f"\nMulti-seed DE completed in {elapsed:.2f}s")
    print(f"Best result from seed {best_seed}: price = {best_price:.2f}")

    # Show all results for comparison
    print("\nAll results (sorted by price):")
    for idx, seed, price, x in valid_results[:5]:
        print(f"  Seed {seed}: {price:.2f}")

    report_progress("Multi-DE", 1.0, f"Best price: {best_price:.2f} (seed {best_seed})", None)

    return {
        "price": best_price,
        "corediameter": best_x[5],
        "corelength": best_x[6],
        "lvfoilturns": int(round(best_x[0])),
        "lvfoilheight": best_x[1],
        "lvfoilthickness": best_x[2],
        "hvthickness": best_x[3],
        "hvlength": best_x[4],
        "best_seed": best_seed,
        "n_seeds_tried": n_seeds
    }


def StartMultiSeedDE(tolerance=25, obround=True, put_cooling_ducts=True, print_result=True,
                     n_seeds=5, progress_callback=None):
    """
    Multi-seed Differential Evolution optimization.

    Runs multiple DE optimizations with different seeds and keeps the best result.
    More robust than single-seed DE for finding the global optimum.
    """
    print("=" * 60)
    print(f"MULTI-SEED DIFFERENTIAL EVOLUTION ({n_seeds} seeds)")
    print("=" * 60)
    start_time = time.time()

    interim = global_search_de_multiseed(
        obround=obround,
        put_cooling_ducts=put_cooling_ducts,
        tolerance=tolerance,
        n_seeds=n_seeds,
        progress_callback=progress_callback
    )

    if interim is None:
        print("Multi-seed DE failed!")
        return None

    best_price = interim["price"]
    lv_turns = interim["lvfoilturns"]
    lv_height = interim["lvfoilheight"]
    lv_thickness = interim["lvfoilthickness"]
    hv_thickness = interim["hvthickness"]
    hv_length = interim["hvlength"]
    core_diameter = interim["corediameter"]
    core_length = interim["corelength"]
    best_seed = interim.get("best_seed", "unknown")

    total_time = time.time() - start_time

    if print_result:
        print("=" * 60)
        print(f"OPTIMIZATION COMPLETED in {total_time:.2f} seconds")
        print(f"Best seed: {best_seed}")
        print("=" * 60)
        print(f"Best Price: {best_price:.2f}")
        print(f"Core Diameter: {core_diameter:.1f}")
        print(f"Core Length: {core_length:.1f}")
        print(f"LV Turns: {lv_turns}")
        print(f"LV Height: {lv_height:.1f}")
        print(f"LV Thickness: {lv_thickness:.2f}")
        if HV_WIRE_CIRCULAR:
            print(f"HV Wire Diameter: {hv_thickness:.2f}")
        else:
            print(f"HV Thickness: {hv_thickness:.2f}")
            print(f"HV Length: {hv_length:.2f}")
        print("=" * 60)
        print("\nVerifying with detailed calculation:")
        CalculateFinalizedPrice(
            lv_turns, lv_height, lv_thickness, hv_thickness, hv_length,
            core_diameter, core_length, LVRATE, HVRATE, POWERRATING, FREQUENCY,
            materialToBeUsedWire_Resistivity, GUARANTEED_NO_LOAD_LOSS,
            GUARANTEED_LOAD_LOSS, GUARANTEED_UCC,
            isFinal=True, PutCoolingDucts=put_cooling_ducts, circular=HV_WIRE_CIRCULAR
        )

    return _build_result_dict(
        lv_turns=lv_turns, lv_height=lv_height, lv_thickness=lv_thickness,
        hv_thickness=hv_thickness, hv_length=hv_length,
        core_diameter=core_diameter, core_length=core_length,
        total_price=best_price, elapsed_time=total_time, put_cooling_ducts=put_cooling_ducts,
        circular=HV_WIRE_CIRCULAR
    )


def StartSmartOptimized(tolerance=25, obround=True, put_cooling_ducts=True, use_de=False, n_lhs_samples=3000, print_result=True, progress_callback=None):
    """Combined smart optimization: LHS coarse search + L-BFGS-B fine search.

    Args:
        progress_callback: Optional function(stage, progress, message, eta_seconds) for UI updates
    """
    def report_progress(stage, progress, message, eta=None):
        if progress_callback:
            try:
                progress_callback(stage, progress, message, eta)
            except:
                pass

    print("=" * 60)
    print("SMART OPTIMIZED TRANSFORMER SEARCH")
    print("=" * 60)
    start_time = time.time()
    report_progress("Smart", 0.0, "Starting smart optimization...", None)

    if use_de:
        report_progress("DE", 0.01, "Running Differential Evolution...", None)
        interim = global_search_de(obround, put_cooling_ducts, tolerance, progress_callback=progress_callback)
        best_price = interim["price"]
        lv_turns = interim["lvfoilturns"]
        lv_height = interim["lvfoilheight"]
        lv_thickness = interim["lvfoilthickness"]
        hv_thickness = interim["hvthickness"]
        hv_length = interim["hvlength"]
        core_diameter = interim["corediameter"]
        core_length = interim["corelength"]
    else:
        report_progress("LHS", 0.01, f"LHS coarse search ({n_lhs_samples} samples)...", None)
        lhs_start = time.time()
        coarse_result = smart_coarse_search(n_lhs_samples, obround, put_cooling_ducts, tolerance)

        if coarse_result is None or coarse_result[0] is None:
            print("Coarse search failed, falling back to parallel grid search")
            return StartOptimized(tolerance, obround, put_cooling_ducts, print_result)

        lhs_elapsed = time.time() - lhs_start
        report_progress("LHS", 0.5, f"LHS complete in {lhs_elapsed:.1f}s, starting L-BFGS-B...", None)

        initial_point, initial_price = coarse_result
        lbfgsb_start = time.time()
        final_point, best_price = fine_search_scipy(initial_point, obround, put_cooling_ducts, tolerance)
        lbfgsb_elapsed = time.time() - lbfgsb_start
        report_progress("L-BFGS-B", 0.95, f"L-BFGS-B complete in {lbfgsb_elapsed:.1f}s", None)

        lv_turns = int(round(final_point[0]))
        lv_height = final_point[1]
        lv_thickness = final_point[2]
        hv_thickness = final_point[3]
        hv_length = final_point[4]
        core_diameter = final_point[5]
        core_length = final_point[6]

    total_time = time.time() - start_time
    report_progress("Complete", 1.0, f"Complete! Best price: {best_price:.2f} in {total_time:.1f}s", None)

    if print_result:
        print("=" * 60)
        print(f"OPTIMIZATION COMPLETED in {total_time:.2f} seconds")
        print("=" * 60)
        print(f"Best Price: {best_price:.2f}")
        print(f"Core Diameter: {core_diameter:.1f}")
        print(f"Core Length: {core_length:.1f}")
        print(f"LV Turns: {lv_turns}")
        print(f"LV Height: {lv_height:.1f}")
        print(f"LV Thickness: {lv_thickness:.2f}")
        print(f"HV Thickness: {hv_thickness:.2f}")
        print(f"HV Length: {hv_length:.2f}")
        print("=" * 60)
        print("\nVerifying with detailed calculation:")
        CalculateFinalizedPrice(
            lv_turns, lv_height, lv_thickness, hv_thickness, hv_length,
            core_diameter, core_length, LVRATE, HVRATE, POWERRATING, FREQUENCY,
            materialToBeUsedWire_Resistivity, GUARANTEED_NO_LOAD_LOSS,
            GUARANTEED_LOAD_LOSS, GUARANTEED_UCC,
            isFinal=True, PutCoolingDucts=put_cooling_ducts, circular=HV_WIRE_CIRCULAR
        )

    return _build_result_dict(
        lv_turns=lv_turns, lv_height=lv_height, lv_thickness=lv_thickness,
        hv_thickness=hv_thickness, hv_length=hv_length,
        core_diameter=core_diameter, core_length=core_length,
        total_price=best_price, elapsed_time=total_time, put_cooling_ducts=put_cooling_ducts,
        circular=HV_WIRE_CIRCULAR
    )


# =============================================================================
# RESULT HELPER
# =============================================================================

def _build_result_dict(lv_turns, lv_height, lv_thickness, hv_thickness, hv_length,
                       core_diameter, core_length, total_price, elapsed_time, put_cooling_ducts=True, circular=False):
    """Build comprehensive result dictionary with all design details.

    Uses current module-level values for LVRATE, HVRATE, POWERRATING, etc.
    """
    lv_turns = int(round(lv_turns))

    # Get current module values (may have been changed by UI)
    _lvrate = LVRATE
    _hvrate = HVRATE
    _power = POWERRATING
    _freq = FREQUENCY
    _resistivity = materialToBeUsedWire_Resistivity

    # Calculate cooling ducts - pass all parameters explicitly
    n_ducts_lv, n_ducts_hv = (0, 0)
    if put_cooling_ducts:
        n_ducts_lv, n_ducts_hv = CalculateNumberOfCoolingDucts(
            lv_turns, lv_height, lv_thickness, hv_thickness, hv_length,
            core_diameter, core_length,
            LVRATE=_lvrate, HVRATE=_hvrate, POWER=_power, FREQ=_freq,
            MaterialResistivity=_resistivity,
            GUARANTEEDNLL=GUARANTEED_NO_LOAD_LOSS, GUARANTEEDLL=GUARANTEED_LOAD_LOSS,
            GUARANTEEDUCC=GUARANTEED_UCC, circular=circular
        )

    # Calculate losses - pass all parameters explicitly
    load_loss, _, _ = CalculateLoadLosses(
        lv_turns, lv_thickness, core_diameter, lv_height, hv_thickness, hv_length,
        _resistivity, _power, _hvrate, _lvrate, core_length,
        n_ducts_lv, n_ducts_hv, circular
    )

    no_load_loss = CalculateNoLoadLosses(
        lv_turns, lv_height, lv_thickness, hv_thickness, hv_length,
        core_diameter, _lvrate, core_length, n_ducts_lv, n_ducts_hv, circular
    )

    # Calculate impedance
    stray_dia = CalculateStrayDiameter(
        lv_turns, lv_thickness, lv_height, hv_thickness, hv_length,
        core_diameter, core_length, n_ducts_lv, n_ducts_hv, circular
    )
    ux = CalculateUx(_power, stray_dia, lv_turns, lv_thickness, lv_height,
                     hv_thickness, hv_length, _freq, _lvrate, n_ducts_lv, n_ducts_hv, circular)
    ur = CalculateUr(load_loss, _power)
    impedance = CalculateImpedance(ux, ur)

    # Calculate weights (multiply by 3 for 3-phase transformer)
    core_weight = CalculateCoreWeight(lv_turns, lv_height, lv_thickness, hv_thickness, hv_length,
                                      core_diameter, core_length, n_ducts_lv, n_ducts_hv, circular)

    lv_volume = CalculateVolumeLV(lv_turns, lv_thickness, core_diameter, lv_height, core_length, n_ducts_lv) * 3
    lv_weight = CalculateWeightOfVolume(lv_volume, materialToBeUsedFoil_Density)

    hv_volume = CalculateVolumeHV(lv_turns, lv_thickness, core_diameter, lv_height, hv_thickness, hv_length,
                                  core_length, n_ducts_hv, circular) * 3
    hv_weight = CalculateWeightOfVolume(hv_volume, materialToBeUsedWire_Density)

    # Calculate prices
    core_price = CalculatePriceOfWeight(core_weight, CorePricePerKg)
    lv_price = CalculatePriceOfWeight(lv_weight, materialToBeUsedFoil_Price)
    hv_price = CalculatePriceOfWeight(hv_weight, materialToBeUsedWire_Price)

    return {
        # Design parameters
        'core_diameter': core_diameter,
        'core_length': core_length,
        'lv_turns': lv_turns,
        'lv_height': lv_height,
        'lv_thickness': lv_thickness,
        'hv_thickness': hv_thickness,
        'hv_length': hv_length,
        # Performance
        'no_load_loss': no_load_loss,
        'load_loss': load_loss,
        'impedance': impedance,
        # Weights
        'core_weight': core_weight,
        'lv_weight': lv_weight,
        'hv_weight': hv_weight,
        # Prices
        'core_price': core_price,
        'lv_price': lv_price,
        'hv_price': hv_price,
        'total_price': total_price,
        # Timing
        'time': elapsed_time,
        # Cooling ducts
        'n_ducts_lv': n_ducts_lv,
        'n_ducts_hv': n_ducts_hv
    }


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def StartFast(tolerance=25, obround=True, put_cooling_ducts=True, method='de', print_result=True, grid_resolution='coarse', search_depth='normal', progress_callback=None, n_seeds=5, constraints=None):
    """
    Fast transformer optimization with multiple algorithm options.

    Args:
        tolerance: Tolerance percentage for constraints
        obround: Whether core is obround shape
        put_cooling_ducts: Whether to calculate cooling ducts
        method: Optimization method:
            - 'hybrid': Two-stage GPU (RECOMMENDED - best accuracy)
            - 'mps': PyTorch MPS GPU (Apple Silicon)
            - 'mlx': Apple MLX GPU (Apple Silicon)
            - 'gpu': CUDA GPU (NVIDIA GPUs)
            - 'de': Differential Evolution (CPU, good quality)
            - 'multi_de': Multi-seed DE (CPU, more robust than single DE)
            - 'smart': LHS + L-BFGS-B (CPU)
            - 'parallel': Numba parallel CPU grid search
        print_result: Print detailed results
        grid_resolution: For mps/mlx/gpu/parallel methods - 'coarse', 'medium', 'fine'
        search_depth: For hybrid method - 'fast', 'normal', 'thorough', 'exhaustive'
        progress_callback: Optional function(stage, progress, message, eta) for UI updates
        n_seeds: For multi_de method - number of parallel DE seeds (1-15, default 5)
        constraints: Dict of locked parameters, e.g. {'core_diameter': 200.0, 'lv_turns': 30}
            Supported keys: core_diameter, core_length, lv_turns, lv_height, lv_thickness,
                           hv_thickness, hv_length

    Returns:
        Dictionary with optimal parameters, or None if no valid design found

    RECOMMENDATION:
        Use method='hybrid' with search_depth:
        - 'fast': ~10s - quick estimate
        - 'normal': ~30s - balanced (default)
        - 'thorough': ~2-5min - better results
        - 'exhaustive': ~10-30min - best possible results

    Speed comparison (Apple Silicon Mac):
        - 'hybrid' + 'exhaustive': ~10-30min (BEST ACCURACY)
        - 'hybrid' + 'thorough': ~2-5min (VERY GOOD)
        - 'hybrid' + 'normal': ~30s (GOOD DEFAULT)
        - 'de': ~3-5s (quick but may miss global optimum)
    """
    import sys

    # Apply constraints by modifying global MIN/MAX values
    global CORE_MINIMUM, CORE_MAXIMUM, CORELENGTH_MINIMUM
    global FOILTURNS_MINIMUM, FOILTURNS_MAXIMUM
    global FOILHEIGHT_MINIMUM, FOILHEIGHT_MAXIMUM
    global FOILTHICKNESS_MINIMUM, FOILTHICKNESS_MAXIMUM
    global HVTHICK_MINIMUM, HVTHICK_MAXIMUM
    global HV_LEN_MINIMUM, HV_LEN_MAXIMUM

    # Save original values to restore later
    orig_values = {
        'CORE_MINIMUM': CORE_MINIMUM, 'CORE_MAXIMUM': CORE_MAXIMUM,
        'CORELENGTH_MINIMUM': CORELENGTH_MINIMUM,
        'FOILTURNS_MINIMUM': FOILTURNS_MINIMUM, 'FOILTURNS_MAXIMUM': FOILTURNS_MAXIMUM,
        'FOILHEIGHT_MINIMUM': FOILHEIGHT_MINIMUM, 'FOILHEIGHT_MAXIMUM': FOILHEIGHT_MAXIMUM,
        'FOILTHICKNESS_MINIMUM': FOILTHICKNESS_MINIMUM, 'FOILTHICKNESS_MAXIMUM': FOILTHICKNESS_MAXIMUM,
        'HVTHICK_MINIMUM': HVTHICK_MINIMUM, 'HVTHICK_MAXIMUM': HVTHICK_MAXIMUM,
        'HV_LEN_MINIMUM': HV_LEN_MINIMUM, 'HV_LEN_MAXIMUM': HV_LEN_MAXIMUM,
    }

    constraints = constraints or {}

    # Apply constraints by setting MIN = MAX to the constrained value
    if 'core_diameter' in constraints:
        val = float(constraints['core_diameter'])
        CORE_MINIMUM = val
        CORE_MAXIMUM = val + 0.001
    if 'core_length' in constraints:
        val = float(constraints['core_length'])
        CORELENGTH_MINIMUM = val
    if 'lv_turns' in constraints:
        val = int(constraints['lv_turns'])
        FOILTURNS_MINIMUM = val
        FOILTURNS_MAXIMUM = val + 1
    if 'lv_height' in constraints:
        val = float(constraints['lv_height'])
        FOILHEIGHT_MINIMUM = val
        FOILHEIGHT_MAXIMUM = val + 0.001
    if 'lv_thickness' in constraints:
        val = float(constraints['lv_thickness'])
        FOILTHICKNESS_MINIMUM = val
        FOILTHICKNESS_MAXIMUM = val + 0.001
    if 'hv_thickness' in constraints:
        val = float(constraints['hv_thickness'])
        HVTHICK_MINIMUM = val
        HVTHICK_MAXIMUM = val + 0.001
    if 'hv_length' in constraints:
        val = float(constraints['hv_length'])
        HV_LEN_MINIMUM = val
        HV_LEN_MAXIMUM = val + 0.001

    print(f"\n{'='*60}")
    print(f"TRANSFORMER OPTIMIZATION - Method: {method.upper()}")
    print(f"Available GPU backends: {', '.join(GPU_OPTIONS) if GPU_OPTIONS else 'None'}")
    if constraints:
        print(f"Constraints: {constraints}")
    print(f"{'='*60}\n")
    sys.stdout.flush()

    # Ensure JIT functions are compiled before starting optimization
    ensure_jit_ready()

    try:
        if method == 'hybrid':
            result = StartMPSHybrid(tolerance, obround, put_cooling_ducts, print_result, search_depth=search_depth, progress_callback=progress_callback)
        elif method == 'cuda_hybrid':
            result = StartCUDAHybrid(tolerance, obround, put_cooling_ducts, print_result, search_depth=search_depth, progress_callback=progress_callback)
        elif method == 'mps':
            result = StartMPS(tolerance, obround, put_cooling_ducts, print_result, grid_resolution=grid_resolution, progress_callback=progress_callback)
        elif method == 'mlx':
            result = StartMLX(tolerance, obround, put_cooling_ducts, print_result, grid_resolution=grid_resolution, progress_callback=progress_callback)
        elif method == 'gpu':
            result = StartGPU(tolerance, obround, put_cooling_ducts, print_result, grid_resolution=grid_resolution, progress_callback=progress_callback)
        elif method == 'parallel':
            result = StartOptimized(tolerance, obround, put_cooling_ducts, print_result, grid_resolution=grid_resolution, progress_callback=progress_callback)
        elif method == 'smart':
            result = StartSmartOptimized(tolerance, obround, put_cooling_ducts, use_de=False, print_result=print_result, progress_callback=progress_callback)
        elif method == 'de':
            result = StartSmartOptimized(tolerance, obround, put_cooling_ducts, use_de=True, print_result=print_result, progress_callback=progress_callback)
        elif method == 'multi_de':
            result = StartMultiSeedDE(tolerance, obround, put_cooling_ducts, print_result=print_result, n_seeds=n_seeds, progress_callback=progress_callback)
        else:
            print(f"Unknown method '{method}', using 'hybrid'")
            result = StartMPSHybrid(tolerance, obround, put_cooling_ducts, print_result, search_depth=search_depth, progress_callback=progress_callback)

        # Ensure result has all required fields (GPU methods return minimal data)
        if result and 'no_load_loss' not in result:
            # Use helper to build full result with performance metrics
            result = _build_result_dict(
                lv_turns=result['lv_turns'],
                lv_height=result['lv_height'],
                lv_thickness=result['lv_thickness'],
                hv_thickness=result['hv_thickness'],
                hv_length=result['hv_length'],
                core_diameter=result['core_diameter'],
                core_length=result['core_length'],
                total_price=result['price'],
                elapsed_time=result.get('time', 0),
                put_cooling_ducts=put_cooling_ducts,
                circular=HV_WIRE_CIRCULAR
            )

        return result

    except Exception as e:
        print(f"Optimization failed with error: {e}")
        print("Falling back to original BucketFillingSmart...")
        return BucketFillingSmart(printValuesFinal=print_result, tolerance=tolerance, obround=obround, PutCoolingDuct=put_cooling_ducts)

    finally:
        # Restore original MIN/MAX values
        CORE_MINIMUM = orig_values['CORE_MINIMUM']
        CORE_MAXIMUM = orig_values['CORE_MAXIMUM']
        CORELENGTH_MINIMUM = orig_values['CORELENGTH_MINIMUM']
        FOILTURNS_MINIMUM = orig_values['FOILTURNS_MINIMUM']
        FOILTURNS_MAXIMUM = orig_values['FOILTURNS_MAXIMUM']
        FOILHEIGHT_MINIMUM = orig_values['FOILHEIGHT_MINIMUM']
        FOILHEIGHT_MAXIMUM = orig_values['FOILHEIGHT_MAXIMUM']
        FOILTHICKNESS_MINIMUM = orig_values['FOILTHICKNESS_MINIMUM']
        FOILTHICKNESS_MAXIMUM = orig_values['FOILTHICKNESS_MAXIMUM']
        HVTHICK_MINIMUM = orig_values['HVTHICK_MINIMUM']
        HVTHICK_MAXIMUM = orig_values['HVTHICK_MAXIMUM']
        HV_LEN_MINIMUM = orig_values['HV_LEN_MINIMUM']
        HV_LEN_MAXIMUM = orig_values['HV_LEN_MAXIMUM']


def StartBFNJITOpt(tolerance=1, printValuesProc=False, printValuesFinal=False, obround=True, PutCoolingDuct=True):
    """Wrapper for original BucketFillingSmart with timing."""
    startTime = time.time()
    BucketFillingSmart(printValuesProc=printValuesProc, printValuesFinal=printValuesFinal, obround=obround, PutCoolingDuct=PutCoolingDuct, tolerance=tolerance)
    endTime = time.time()
    print("Process Has Took : " + str(endTime - startTime) + " SECONDS")


# =============================================================================
# JIT PRE-COMPILATION (Warmup at module load)
# =============================================================================
 
def _warmup_jit():
    """Pre-compile all JIT functions with dummy data to avoid first-call delay."""
    # Trigger compilation of core calculation functions
    Clamp(1.0, 0.0, 2.0)
    CalculateVoltsPerTurns(400.0, 20.0)
    CalculateCoreSection(150.0, 50.0)
    CalculateInduction(20.0, 150.0, 50.0)
    CalculateWattsPerKG(1.5)
    CalculateRadialThicknessLV(20, 1.0, 0)
    CalculateAverageDiameterLV(20, 1.0, 150.0, 50.0, 0)
    CalculateTotalLengthCoilLV(20, 1.0, 150.0, 50.0, 0)
    CalculateRadialThicknessHV(400.0, 20, 2.0, 5.0, 0)
    CalculateAverageDiameterHV(20, 1.0, 150.0, 400.0, 2.0, 5.0, 50.0, 0)
    CalculateTotalLengthCoilHV(20, 1.0, 150.0, 400.0, 2.0, 5.0, 50.0, 0)
    CalculateCoreWeight(20, 400.0, 1.0, 2.0, 5.0, 150.0, 50.0, 0, 0)
    CalculateVolumeLV(20, 1.0, 150.0, 400.0, 50.0, 0)
    CalculateSectionHV(2.0, 5.0)
    CalculateSectionLV(400.0, 1.0)
    CalculateVolumeHV(20, 1.0, 150.0, 400.0, 2.0, 5.0, 50.0, 0)
    CalculateWeightOfVolume(100.0, 8.9)
    CalculatePriceOfWeight(100.0, 10.0)
    CalculateResistanceLV(0.021, 1000.0, 400.0)
    CalculateResistanceHV(0.021, 5000.0, 10.0)
    CalculateCurrent(1000.0, 400.0)
    CalculateCurrentLV(1000.0, 400.0)
    CalculateCurrentHV(1000.0, 20000.0)
    CalculateLoadLosses(20, 1.0, 150.0, 400.0, 2.0, 5.0, 0.021, 1000.0, 20000.0, 400.0, 50.0, 0, 0)
    CalculateNoLoadLosses(20, 400.0, 1.0, 2.0, 5.0, 150.0, 400.0, 50.0, 0, 0)
    CalculateStrayDiameter(20, 1.0, 400.0, 2.0, 5.0, 150.0, 50.0, 0, 0)
    CalculateUr(5000.0, 1000.0)
    CalculateUx(1000.0, 200.0, 20, 1.0, 400.0, 2.0, 5.0, 50.0, 400.0, 0, 0)
    CalculateImpedance(5.0, 0.5)
    CalculatePrice(20, 400.0, 1.0, 2.0, 5.0, 150.0, 50.0, 0, 0, False)

    # Cooling duct functions
    CalculateEfficencyOfCoolingDuct(400.0)
    CalculateEfficencyOfMainGap(400.0)
    CalculateTotalInsulationThicknessLV(20)
    CalculateTotalInsulationThicknessHV(5)
    CalculateHeatFluxLV(2000.0, 400.0, 20, 1.0, 150.0, 50.0)
    CalculateHeatFluxHV(2000.0, 350.0, 400.0, 20, 1.0, 150.0, 2.0, 5.0, 50.0)
    AidingFormulaHVOpenDucts(1.0, 0.9, 0.8, 0)
    AidingFormulaHVCloseDucts(1.0, 0.9, 0.8, 1)
    AidingFormulaLVOpenDucts(1.0, 0.9, 0.8, 0)
    AidingFormulaLVCloseDuct(1.0, 0.9, 0.8, 1)
    CalculateGradientHeatLV(20, 1.0, 0.9, 0.8, 0)
    CalculateGradientHeatHV(5, 1.0, 0.9, 0.8, 0)
    CalculateNumberOfCoolingDucts(20, 400.0, 1.0, 2.0, 5.0, 150.0, 50.0)
    CalculateNumberOfCoolingDucts_WithLosses(20, 400.0, 1.0, 2.0, 5.0, 150.0, 50.0, 0.021, 1000.0, 20000.0, 400.0, False)

    # Main price calculation functions
    CalculateFinalizedPriceIntolerant(20, 400.0, 1.0, 2.0, 5.0, 150.0, 50.0, tolerance=25, PutCoolingDucts=True)
    CalculateFinalizedPriceIntolerant_Optimized(20, 400.0, 1.0, 2.0, 5.0, 150.0, 50.0, tolerance=25, PutCoolingDucts=True)

    # Parallel grid search kernel (with minimal arrays)
    _core_arr = np.array([150.0], dtype=np.float64)
    _len_arr = np.array([0.0], dtype=np.float64)
    _turns_arr = np.array([20.0], dtype=np.float64)
    _height_arr = np.array([400.0], dtype=np.float64)
    _thick_arr = np.array([1.0], dtype=np.float64)
    _hvthick_arr = np.array([2.0], dtype=np.float64)
    _hvlen_arr = np.array([5.0], dtype=np.float64)
    parallel_grid_search_kernel(
        _core_arr, _len_arr, _turns_arr, _height_arr, _thick_arr, _hvthick_arr, _hvlen_arr,
        25.0, True, True, False,  # circular=False for warmup
        LVRATE, HVRATE, POWERRATING, FREQUENCY, materialToBeUsedWire_Resistivity,
        GUARANTEED_NO_LOAD_LOSS, GUARANTEED_LOAD_LOSS, GUARANTEED_UCC,
        CoreFillingFactorRound, CoreFillingFactorRectangular, INSULATION_THICKNESS_WIRE
    )


# Lazy JIT warmup - only runs when first optimization is called
_jit_warmed_up = False

def ensure_jit_ready():
    """Ensure JIT functions are compiled. Call before first optimization."""
    global _jit_warmed_up
    if not _jit_warmed_up:
        import sys
        print("Pre-compiling JIT functions...", end=" ", flush=True)
        sys.stdout.flush()
        _warmup_start = time.time()
        _warmup_jit()
        print(f"done ({time.time() - _warmup_start:.1f}s)", flush=True)
        sys.stdout.flush()
        _jit_warmed_up = True


if __name__ == '__main__':
    # RECOMMENDED: Two-stage hybrid with search depth options
    StartFast(method='hybrid',search_depth="thorough")  # ~30s - balanced default

    # Search depth options for hybrid method:
    #StartFast(method='hybrid', search_depth='fast')       # ~10s - quick estimate
    #StartFast(method='hybrid', search_depth='normal')     # ~30s - balanced (default)
    #StartFast(method='hybrid', search_depth='thorough')   # ~2-5min - better results
    #StartFast(method='hybrid', search_depth='exhaustive') # ~10-30min - best possible

    # Alternative GPU methods (faster but less thorough):
    #StartFast(method='mps', grid_resolution='fine')   # PyTorch MPS (~30-60s)
    #StartFast(method='mlx', grid_resolution='fine')   # Apple MLX (~30-60s)
    #StartFast(method='gpu', grid_resolution='fine')   # CUDA NVIDIA (~30-60s)

    # CPU methods (good for quick estimates):
    #StartFast(method='de')        # Differential Evolution (~3-5s)
    #StartFast(method='smart')     # LHS + L-BFGS-B (~3s)
    #StartFast(method='parallel', grid_resolution='coarse')  # CPU parallel (~15s)
    #StartBFNJITOpt(printValuesFinal=True, tolerance=25)  # Original method
