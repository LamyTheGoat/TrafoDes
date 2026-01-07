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

AdditionalLossFactorLV = 1.12
AdditionalLossFactorHV = 1.12

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
    return LVRate / LVTurns


@njit(fastmath=True)
def CalculateCoreSection(CoreDiameter, CoreLength):
    return ((CoreDiameter**2 * math.pi) / (4 * 100)) * CoreFillingFactorRound + \
           (CoreLength * CoreDiameter / 100) * CoreFillingFactorRectangular


@njit(fastmath=True)
def CalculateInduction(VolsPerTurn, CoreDiameter, CoreLength):
    return (VolsPerTurn * 10000) / (math.sqrt(2) * math.pi * FREQUENCY * CalculateCoreSection(CoreDiameter, CoreLength))


@njit(fastmath=True)
def CalculateWattsPerKG(Induction):
    return (1.3498 * Induction**6) + (-8.1737 * Induction**5) + \
           (19.884 * Induction**4) + (-24.708 * Induction**3) + \
           (16.689 * Induction**2) + (-5.5386 * Induction) + 0.7462


@njit(fastmath=True)
def CalculateRadialThicknessLV(LVNumberOfTurns, LVFoilThiccnes, NumberOfDucts=0, ThicknessOfDucts=COOLING_DUCT_THICKNESS):
    return LVNumberOfTurns * LVFoilThiccnes + ((LVNumberOfTurns - 1) * LVInsulationThickness) + \
           (NumberOfDucts * (ThicknessOfDucts + 0.5))


@njit(fastmath=True)
def CalculateAverageDiameterLV(LVNumberOfTurns, LVFoilThiccnes, CoreDiameter, CoreLength, NumberOfDucts):
    return CoreDiameter + (2 * DistanceCoreLV) + CalculateRadialThicknessLV(LVNumberOfTurns, LVFoilThiccnes, NumberOfDucts) + \
           (2 * CoreLength / math.pi)


@njit(fastmath=True)
def CalculateTotalLengthCoilLV(LVNumberOfTurns, LVFoilThiccnes, CoreDiameter, CoreLength, NumberOfDucts):
    return CalculateAverageDiameterLV(LVNumberOfTurns, LVFoilThiccnes, CoreDiameter, CoreLength, NumberOfDucts) * \
           math.pi * LVNumberOfTurns


@njit(fastmath=True)
def CalculateRadialThiccnessHV(LVFoilHeight, LVNumberOfTurns, HVWireThickness, HVWireLength, NumberOfDucts=0, ThicknessOfDucts=COOLING_DUCT_THICKNESS):
    HVNumberOfTurns = LVNumberOfTurns * (HVRATE / LVRATE)
    HVLayerHeight = LVFoilHeight - 50
    HVTurnsPerLayer = (HVLayerHeight / (HVWireLength + INSULATION_THICKNESS_WIRE)) - 1
    HVLayerNumber = math.ceil(HVNumberOfTurns / HVTurnsPerLayer)
    return HVLayerNumber * HVWireThickness + (HVLayerNumber - 1) * HVInsulationThickness + \
           (NumberOfDucts * (ThicknessOfDucts + 0.5))


@njit(fastmath=True)
def CalculateAverageDiameterHV(LVNumberOfTurns, LVFoilThiccnes, CoreDiameter, LVFoilHeight, HVWireThickness, HVWireLength, CoreLength, NumberOfDucts):
    return CoreDiameter + 2 * DistanceCoreLV + 2 * CalculateRadialThicknessLV(LVNumberOfTurns, LVFoilThiccnes, NumberOfDucts) + \
           2 * MainGap + CalculateRadialThiccnessHV(LVFoilHeight, LVNumberOfTurns, HVWireThickness, HVWireLength, NumberOfDucts) + \
           (2 * CoreLength / math.pi)


@njit(fastmath=True)
def CalculateTotalLengthCoilHV(LVNumberOfTurns, LVFoilThiccnes, CoreDiameter, LVFoilHeight, HVWireThickness, HVWireLength, CoreLength, NumberOfDucts):
    HVNumberOfTurns = LVNumberOfTurns * HVRATE / LVRATE
    return CalculateAverageDiameterHV(LVNumberOfTurns, LVFoilThiccnes, CoreDiameter, LVFoilHeight, HVWireThickness, HVWireLength, CoreLength, NumberOfDucts) * \
           math.pi * HVNumberOfTurns


@njit(fastmath=True)
def CalculateCoreWeight(LVNumberOfTurns, LVFoilHeight, LVFoilThiccnes, HVWireThickness, HVWireLength, CoreDiameter, CoreLength, NumberOfDuctsLv, NumberOfDuctsHv):
    FoilHeight = LVFoilHeight
    WindowHeight = FoilHeight + 40
    RadialThickness = CalculateRadialThicknessLV(LVNumberOfTurns, LVFoilThiccnes, NumberOfDuctsLv) + \
                      CalculateRadialThiccnessHV(LVFoilHeight, LVNumberOfTurns, HVWireThickness, HVWireLength, NumberOfDuctsHv) + \
                      MainGap + DistanceCoreLV
    CenterBetweenLegs = (CoreDiameter + RadialThickness * 2) + PhaseGap
    rectWeight = (((3 * WindowHeight) + 2 * (2 * CenterBetweenLegs + CoreDiameter)) * (CoreDiameter * CoreLength / 100) * CoreDensity * CoreFillingFactorRectangular) / 1e6
    radius = CoreDiameter / 2
    squareEdge = radius * math.sqrt(math.pi)
    roundWeight = (((3 * (WindowHeight + 10)) + 2 * (2 * CenterBetweenLegs + CoreDiameter)) * (squareEdge * squareEdge / 100) * CoreDensity * CoreFillingFactorRound) / 1e6
    return (rectWeight + roundWeight) * 100


@njit(fastmath=True)
def CalculateVolumeLV(LVNumberOfTurns, LVFoilThiccnes, CoreDiameter, LVFoilHeight, CoreLength, NumberOfDucts):
    length = CalculateTotalLengthCoilLV(LVNumberOfTurns, LVFoilThiccnes, CoreDiameter, CoreLength, NumberOfDucts)
    return (length * LVFoilHeight * LVFoilThiccnes) / 1000000


@njit(fastmath=True)
def CalculateSectionHV(Thickness, Length):
    return (Thickness * Length) - (Thickness**2) + (((Thickness / 2)**2) * math.pi)


@njit(fastmath=True)
def CalculateSectionLV(HeightLV, ThiccnessLV):
    return HeightLV * ThiccnessLV


@njit(fastmath=True)
def CalculateVolumeHV(LVNumberOfTurns, LVFoilThiccnes, CoreDiameter, LVFoilHeight, HVWireThickness, HVWireLength, CoreLength, NumberOfDucts):
    length = CalculateTotalLengthCoilHV(LVNumberOfTurns, LVFoilThiccnes, CoreDiameter, LVFoilHeight, HVWireThickness, HVWireLength, CoreLength, NumberOfDucts)
    return (length * CalculateSectionHV(HVWireThickness, HVWireLength)) / 1000000


@njit(fastmath=True)
def CalculateWeightOfVolume(Volume, MaterialDensity):
    return Volume * MaterialDensity


@njit(fastmath=True)
def CalculatePriceOfWeight(Weight, MaterialPrice):
    return Weight * MaterialPrice


@njit(fastmath=True)
def CalculateResistanceLV(MaterialResistivity, Length, Section):
    return (Length / 1000) * MaterialResistivity / Section


@njit(fastmath=True)
def CalculateResistanceHV(MaterialResistivity, Length, Section):
    return (Length / 1000) * MaterialResistivity / Section


@njit(fastmath=True)
def CalculateCurrent(Power, Voltage):
    return (Power * 1000) / (Voltage * 3)


@njit(fastmath=True)
def CalculateCurrentLV(Power, VoltageLV):
    return CalculateCurrent(Power, VoltageLV)


@njit(fastmath=True)
def CalculateCurrentHV(Power, VoltageHV):
    return CalculateCurrent(Power, VoltageHV)


@njit(fastmath=True)
def CalculateLoadLosses(LVNumberOfTurns, LVFoilThiccnes, CoreDiameter, LVFoilHeight, HVWireThickness, HVWireLength, MaterialResistivity, Power, HVRating, LVRating, CoreLength, NumberOfDuctsLv, NumberOfDuctsHv):
    lvLength = CalculateTotalLengthCoilLV(LVNumberOfTurns, LVFoilThiccnes, CoreDiameter, CoreLength, NumberOfDuctsLv)
    hvLength = CalculateTotalLengthCoilHV(LVNumberOfTurns, LVFoilThiccnes, CoreDiameter, LVFoilHeight, HVWireThickness, HVWireLength, CoreLength, NumberOfDuctsHv)
    lvSection = CalculateSectionLV(LVFoilHeight, LVFoilThiccnes)
    hvSection = CalculateSectionHV(HVWireThickness, HVWireLength)
    lvResistance = CalculateResistanceLV(MaterialResistivity, lvLength, lvSection)
    hvResistance = CalculateResistanceHV(MaterialResistivity, hvLength, hvSection)
    hvCurrent = CalculateCurrentHV(Power, HVRating)
    lvCurrent = CalculateCurrentLV(Power, LVRating)
    lvLosses = lvResistance * (lvCurrent**2) * 3 * AdditionalLossFactorLV
    hvLosses = hvResistance * (hvCurrent**2) * 3 * AdditionalLossFactorHV
    return lvLosses + hvLosses, lvLosses, hvLosses


@njit(fastmath=True)
def CalculateNoLoadLosses(LVNumberOfTurns, LVFoilHeight, LVFoilThiccnes, HVWireThickness, HVWireLength, CoreDiameter, LVRate, CoreLength, NumberOfDuctsLv, NumberOfDuctsHv):
    CoreWeight = CalculateCoreWeight(LVNumberOfTurns, LVFoilHeight, LVFoilThiccnes, HVWireThickness, HVWireLength, CoreDiameter, CoreLength, NumberOfDuctsLv, NumberOfDuctsHv)
    Induction = CalculateInduction(CalculateVoltsPerTurns(LVRate, LVNumberOfTurns), CoreDiameter, CoreLength)
    WattsPerKG = CalculateWattsPerKG(Induction)
    return WattsPerKG * CoreWeight * 1.2


@njit(fastmath=True)
def CalculateStrayDiameter(LVNumberOfTurns, LVFoilThiccness, LVFoilHeight, HVWireThickness, HVWireLength, DiameterOfCore, CoreLength, NumberOfDuctsLV, NumberOfDuctsHV):
    MainGapDiameter = DiameterOfCore + DistanceCoreLV * 2 + CalculateRadialThicknessLV(LVNumberOfTurns, LVFoilThiccness, NumberOfDuctsLV) * 2 + \
                      (2 * CoreLength / math.pi) + MainGap
    HVRadialThickness = CalculateRadialThiccnessHV(LVFoilHeight, LVNumberOfTurns, HVWireThickness, HVWireLength, NumberOfDuctsHV)
    ReducedWidthHV = HVRadialThickness / 3
    LVRadialThickness = CalculateRadialThicknessLV(LVNumberOfTurns, LVFoilThiccness, NumberOfDuctsLV)
    ReducedWidthLV = LVRadialThickness / 3
    SD = MainGapDiameter + ReducedWidthHV - ReducedWidthLV + ((ReducedWidthHV**2 - ReducedWidthLV**2) / (ReducedWidthLV + ReducedWidthHV + MainGap))
    return SD


@njit(fastmath=True)
def CalculateUr(LoadLosses, Power):
    return LoadLosses / (10 * Power)


@njit(fastmath=True)
def CalculateUx(Power, StrayDiameter, LVNumberOfTurns, LVFoilThiccness, LVFoilHeight, HVWireThickness, HVWireLength, Frequency, LVRate, NumberOfDuctsLV, NumberOfDuctsHV):
    HVRadialThickness = CalculateRadialThiccnessHV(LVFoilHeight, LVNumberOfTurns, HVWireThickness, HVWireLength, NumberOfDuctsHV)
    ReducedWidthHV = HVRadialThickness / 3
    LVRadialThickness = CalculateRadialThicknessLV(LVNumberOfTurns, LVFoilThiccness, NumberOfDuctsLV)
    ReducedWidthLV = LVRadialThickness / 3
    return (Power * StrayDiameter * Frequency * (ReducedWidthLV + ReducedWidthHV + MainGap)) / \
           (1210 * (CalculateVoltsPerTurns(LVRate, LVNumberOfTurns)**2) * LVFoilHeight)


@njit(fastmath=True)
def CalculateImpedance(Ux, Ur):
    return math.sqrt(Ux**2 + Ur**2)


@njit(fastmath=True)
def CalculatePrice(LVNumberOfTurns, LVFoilHeight, LVFoilThiccnes, HVWireThickness, HVWireLength, CoreDiameter, CoreLength, NumberOfDuctsLV, NumberOfDuctsHV, printValues=False):
    wc = CalculateCoreWeight(LVNumberOfTurns, LVFoilHeight, LVFoilThiccnes, HVWireThickness, HVWireLength, CoreDiameter, CoreLength, NumberOfDuctsLV, NumberOfDuctsHV)
    pc = CalculatePriceOfWeight(wc, materialCore_Price)
    volumeHV = CalculateVolumeHV(LVNumberOfTurns, LVFoilThiccnes, CoreDiameter, LVFoilHeight, HVWireThickness, HVWireLength, CoreLength, NumberOfDuctsHV) * 3
    whv = CalculateWeightOfVolume(volumeHV, materialToBeUsedWire_Density)
    phv = CalculatePriceOfWeight(whv, materialToBeUsedWire_Price)
    volumeLV = CalculateVolumeLV(LVNumberOfTurns, LVFoilThiccnes, CoreDiameter, LVFoilHeight, CoreLength, NumberOfDuctsLV) * 3
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
    return min((CoolingDuctThickness / (0.949 * (FoilHeight**0.25))), 1)


@njit(fastmath=True)
def CalculateEfficencyOfMainGap(FoilHeight, DistanceBetweenCoils=MainGap):
    return min(((DistanceBetweenCoils - 0.5) / (0.949 * (FoilHeight**0.25))), 1)


@njit(fastmath=True)
def CalculateTotalInsulationThicknessLV(LVTurns, insulationThicknessLV=LVInsulationThickness):
    return (LVTurns - 1) * insulationThicknessLV


@njit(fastmath=True)
def CalculateTotalInsulationThicknessHV(NumLayerHV, insulationThicknessHV=HVInsulationThickness):
    return (NumLayerHV + 1) * insulationThicknessHV


@njit(fastmath=True)
def CalculateHeatFluxLV(LoadLossesLV, FoilHeight, LVNumberOfTurns, LVFoilThiccnes, CoreDiameter, CoreLength=0):
    AverageLengthLV = CalculateTotalLengthCoilLV(LVNumberOfTurns, LVFoilThiccnes, CoreDiameter, CoreLength, 0) / LVNumberOfTurns
    val = (((LoadLossesLV / 3) * 1.03) / (AverageLengthLV * FoilHeight))
    return val * 10**4


@njit(fastmath=True)
def CalculateHeatFluxHV(LoadLossesHV, HVHeight, FoilHeight, LVNumberOfTurns, LVFoilThiccnes, CoreDiameter, HVWireThickness, HVWireLength, CoreLength=0):
    AverageLengthHV = CalculateAverageDiameterHV(LVNumberOfTurns, LVFoilThiccnes, CoreDiameter, FoilHeight, HVWireThickness, HVWireLength, CoreLength, 0) * math.pi
    val = (((LoadLossesHV / 3) * 1.03) / (AverageLengthHV * HVHeight))
    return val * 10**4


@njit(fastmath=True)
def AidingFormulaHVOpenDucts(HeatFluxHV, MainGapEfficiency, EfficiencyDuct, NumberDucts, DistanceBetWeenCoreLV=DistanceCoreLV, CCDucts=COOLING_DUCT_CENTER_TO_CENTER_DISTANCE, WidthDuct=COOLING_DUCT_WIDTH):
    return HeatFluxHV / (2 * (0.5 + (((CCDucts - WidthDuct) / CCDucts) * ((0.5 * MainGapEfficiency) + (NumberDucts * EfficiencyDuct)))))


@njit(fastmath=True)
def AidingFormulaHVCloseDucts(HeatFluxHV, MainGapEfficiency, EfficiencyDuct, NumberDucts, DistanceBetWeenCoreLV=DistanceCoreLV, CCDucts=COOLING_DUCT_CENTER_TO_CENTER_DISTANCE, WidthDuct=COOLING_DUCT_WIDTH):
    return HeatFluxHV / (2 * (0.5 + (((CCDucts - WidthDuct) / CCDucts) * ((0.5 * MainGapEfficiency) + ((NumberDucts - 1) * EfficiencyDuct)))))


@njit(fastmath=True)
def AidingFormulaLVOpenDucts(HeatFluxLV, MainGapEfficiency, EfficiencyDuct, NumberDucts, DistanceBetWeenCoreLV=DistanceCoreLV, CCDucts=COOLING_DUCT_CENTER_TO_CENTER_DISTANCE, WidthDuct=COOLING_DUCT_WIDTH):
    coefCLV = 0.5 if DistanceCoreLV > 3 else 0.318
    return HeatFluxLV / (2 * (coefCLV + (((CCDucts - WidthDuct) / CCDucts) * ((0.5 * MainGapEfficiency) + (NumberDucts * EfficiencyDuct)))))


@njit(fastmath=True)
def AidingFormulaLVCloseDuct(HeatFluxLV, MainGapEfficiency, EfficiencyDuct, NumberDucts, DistanceBetWeenCoreLV=DistanceCoreLV, CCDucts=COOLING_DUCT_CENTER_TO_CENTER_DISTANCE, WidthDuct=COOLING_DUCT_WIDTH):
    coefCLV = 0.5 if DistanceCoreLV > 3 else 0.318
    return HeatFluxLV / (2 * (coefCLV + (((CCDucts - WidthDuct) / CCDucts) * ((0.5 * MainGapEfficiency) + ((NumberDucts - 1) * EfficiencyDuct)))))


@njit(fastmath=True)
def CalculateGradientHeatLV(TotalTurnsLV, HeatFluxLV, MainGapEfficiency, EfficiencyDuct, NumberDucts):
    openDuctVal = AidingFormulaLVOpenDucts(HeatFluxLV, MainGapEfficiency, EfficiencyDuct, NumberDucts)
    closedDuctVal = openDuctVal if NumberDucts == 0 else AidingFormulaLVCloseDuct(HeatFluxLV, MainGapEfficiency, EfficiencyDuct, NumberDucts)
    val0_1 = (1.754 * openDuctVal**0.8 + (openDuctVal * CalculateTotalInsulationThicknessLV(TotalTurnsLV)) / (6 * 1.16 * (1 + NumberDucts)))
    val0_2 = 0.3 * (1.754 * closedDuctVal**0.8 + (closedDuctVal * CalculateTotalInsulationThicknessLV(TotalTurnsLV)) / (6 * 1.16 * (1 + NumberDucts)))
    return (val0_1 + val0_2) * oilToBeUsed_Factor * (50 / 47)


@njit(fastmath=True)
def CalculateGradientHeatHV(TotalTurnsHV, HeatFluxHV, MainGapEfficiency, EfficiencyDuct, NumberDucts):
    openDuctVal = AidingFormulaHVOpenDucts(HeatFluxHV, MainGapEfficiency, EfficiencyDuct, NumberDucts)
    closedDuctVal = openDuctVal if NumberDucts == 0 else AidingFormulaHVCloseDucts(HeatFluxHV, MainGapEfficiency, EfficiencyDuct, NumberDucts)
    val0_1 = (0.75 * (1.754 * openDuctVal**0.8 + (openDuctVal * CalculateTotalInsulationThicknessHV(TotalTurnsHV)) / (6 * 1.5 * (1 + NumberDucts))))
    val0_2 = (0.30 * (1.754 * closedDuctVal**0.8 + (closedDuctVal * CalculateTotalInsulationThicknessHV(TotalTurnsHV)) / (6 * 1.5 * (1 + NumberDucts))))
    return (val0_1 + val0_2) * oilToBeUsed_Factor


@njit(fastmath=True)
def CalculateNumberOfCoolingDucts(LVNumberOfTurns, LVFoilHeight, LVFoilThiccnes, HVWireThickness, HVWireLength, CoreDiameter, CoreLength, LVRATE=LVRATE, HVRATE=HVRATE, POWER=POWERRATING, FREQ=FREQUENCY, MaterialResistivity=materialToBeUsedWire_Resistivity, GUARANTEEDNLL=GUARANTEED_NO_LOAD_LOSS, GUARANTEEDLL=GUARANTEED_LOAD_LOSS, GUARANTEEDUCC=GUARANTEED_UCC, tolerance=1, isFinal=False):
    Ll, LlLv, LlHv = CalculateLoadLosses(LVNumberOfTurns, LVFoilThiccnes, CoreDiameter, LVFoilHeight, HVWireThickness, HVWireLength, MaterialResistivity, POWER, HVRATE, LVRATE, CoreLength, 0, 0)
    heatFluxLV = CalculateHeatFluxLV(LlLv, LVFoilHeight, LVNumberOfTurns, LVFoilThiccnes, CoreDiameter, CoreLength)
    MainGapEff = CalculateEfficencyOfMainGap(LVFoilHeight)
    DuctEff = CalculateEfficencyOfCoolingDuct(LVFoilHeight)
    gradientLV = CalculateGradientHeatLV(LVNumberOfTurns, heatFluxLV, MainGapEff, DuctEff, 0)
    heatFluxHV = CalculateHeatFluxHV(LlHv, LVFoilHeight - 50, LVFoilHeight, LVNumberOfTurns, LVFoilThiccnes, CoreDiameter, HVWireThickness, HVWireLength, CoreLength)
    HVNumberOfTurns = LVNumberOfTurns * HVRATE / LVRATE
    HVLayerHeight = LVFoilHeight - 50
    HVTurnsPerLayer = (HVLayerHeight / (HVWireLength + INSULATION_THICKNESS_WIRE)) - 1
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
def CalculateNumberOfCoolingDucts_WithLosses(LVNumberOfTurns, LVFoilHeight, LVFoilThiccnes, HVWireThickness, HVWireLength, CoreDiameter, CoreLength, MaterialResistivity, POWER, HVRATE_VAL, LVRATE_VAL, isFinal=False):
    """Returns (numberOfDuctsLV, numberOfDuctsHV, Ll_zero, LlLv_zero, LlHv_zero)"""
    Ll, LlLv, LlHv = CalculateLoadLosses(LVNumberOfTurns, LVFoilThiccnes, CoreDiameter, LVFoilHeight, HVWireThickness, HVWireLength, MaterialResistivity, POWER, HVRATE_VAL, LVRATE_VAL, CoreLength, 0, 0)
    heatFluxLV = CalculateHeatFluxLV(LlLv, LVFoilHeight, LVNumberOfTurns, LVFoilThiccnes, CoreDiameter, CoreLength)
    MainGapEff = CalculateEfficencyOfMainGap(LVFoilHeight)
    DuctEff = CalculateEfficencyOfCoolingDuct(LVFoilHeight)
    gradientLV = CalculateGradientHeatLV(LVNumberOfTurns, heatFluxLV, MainGapEff, DuctEff, 0)
    heatFluxHV = CalculateHeatFluxHV(LlHv, LVFoilHeight - 50, LVFoilHeight, LVNumberOfTurns, LVFoilThiccnes, CoreDiameter, HVWireThickness, HVWireLength, CoreLength)
    HVNumberOfTurns = LVNumberOfTurns * HVRATE_VAL / LVRATE_VAL
    HVLayerHeight = LVFoilHeight - 50
    HVTurnsPerLayer = (HVLayerHeight / (HVWireLength + INSULATION_THICKNESS_WIRE)) - 1
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
def CalculateFinalizedPrice(LVNumberOfTurns, LVFoilHeight, LVFoilThiccnes, HVWireThickness, HVWireLength, CoreDiameter, CoreLength=0, LVRATE=LVRATE, HVRATE=HVRATE, POWER=POWERRATING, FREQ=FREQUENCY, MaterialResistivity=materialToBeUsedWire_Resistivity, GUARANTEEDNLL=GUARANTEED_NO_LOAD_LOSS, GUARANTEEDLL=GUARANTEED_LOAD_LOSS, GUARANTEEDUCC=GUARANTEED_UCC, isFinal=False, PutCoolingDucts=True):
    LvCD = 0
    HvCD = 0
    if PutCoolingDucts:
        LvCD, HvCD = CalculateNumberOfCoolingDucts(LVNumberOfTurns, LVFoilHeight, LVFoilThiccnes, HVWireThickness, HVWireLength, CoreDiameter, CoreLength, LVRATE=LVRATE, HVRATE=HVRATE, POWER=POWERRATING, FREQ=FREQUENCY, MaterialResistivity=materialToBeUsedWire_Resistivity, GUARANTEEDNLL=GUARANTEED_NO_LOAD_LOSS, GUARANTEEDLL=GUARANTEED_LOAD_LOSS, GUARANTEEDUCC=GUARANTEED_UCC, tolerance=1, isFinal=isFinal)
    Nll = CalculateNoLoadLosses(LVNumberOfTurns, LVFoilHeight, LVFoilThiccnes, HVWireThickness, HVWireLength, CoreDiameter, LVRATE, CoreLength, LvCD, HvCD)
    Ll, LlHv, LlLv = CalculateLoadLosses(LVNumberOfTurns, LVFoilThiccnes, CoreDiameter, LVFoilHeight, HVWireThickness, HVWireLength, MaterialResistivity, POWER, HVRATE, LVRATE, CoreLength, LvCD, HvCD)
    strayDia = CalculateStrayDiameter(LVNumberOfTurns, LVFoilThiccnes, LVFoilHeight, HVWireThickness, HVWireLength, CoreDiameter, CoreLength, LvCD, HvCD)
    Ux = CalculateUx(POWER, strayDia, LVNumberOfTurns, LVFoilThiccnes, LVFoilHeight, HVWireThickness, HVWireLength, FREQ, LVRATE, LvCD, HvCD)
    Ur = CalculateUr(Ll, POWER)
    Ucc = CalculateImpedance(Ux, Ur)
    price = CalculatePrice(LVNumberOfTurns, LVFoilHeight, LVFoilThiccnes, HVWireThickness, HVWireLength, CoreDiameter, CoreLength, LvCD, HvCD, isFinal)
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
def CalculateFinalizedPriceIntolerant(LVNumberOfTurns, LVFoilHeight, LVFoilThiccnes, HVWireThickness, HVWireLength, CoreDiameter, CoreLength, LVRATE=LVRATE, HVRATE=HVRATE, POWER=POWERRATING, FREQ=FREQUENCY, MaterialResistivity=materialToBeUsedWire_Resistivity, GUARANTEEDNLL=GUARANTEED_NO_LOAD_LOSS, GUARANTEEDLL=GUARANTEED_LOAD_LOSS, GUARANTEEDUCC=GUARANTEED_UCC, tolerance=1, isFinal=False, PutCoolingDucts=True):
    LvCD = 0
    HvCD = 0
    if PutCoolingDucts:
        LvCD, HvCD = CalculateNumberOfCoolingDucts(LVNumberOfTurns, LVFoilHeight, LVFoilThiccnes, HVWireThickness, HVWireLength, CoreDiameter, CoreLength, LVRATE=LVRATE, HVRATE=HVRATE, POWER=POWERRATING, FREQ=FREQUENCY, MaterialResistivity=materialToBeUsedWire_Resistivity, GUARANTEEDNLL=GUARANTEED_NO_LOAD_LOSS, GUARANTEEDLL=GUARANTEED_LOAD_LOSS, GUARANTEEDUCC=GUARANTEED_UCC, tolerance=1, isFinal=False)
    Nll = CalculateNoLoadLosses(LVNumberOfTurns, LVFoilHeight, LVFoilThiccnes, HVWireThickness, HVWireLength, CoreDiameter, LVRATE, CoreLength, LvCD, HvCD)
    Ll, LlHv, LlLv = CalculateLoadLosses(LVNumberOfTurns, LVFoilThiccnes, CoreDiameter, LVFoilHeight, HVWireThickness, HVWireLength, MaterialResistivity, POWER, HVRATE, LVRATE, CoreLength, LvCD, HvCD)
    strayDia = CalculateStrayDiameter(LVNumberOfTurns, LVFoilThiccnes, LVFoilHeight, HVWireThickness, HVWireLength, CoreDiameter, CoreLength, LvCD, HvCD)
    Ux = CalculateUx(POWER, strayDia, LVNumberOfTurns, LVFoilThiccnes, LVFoilHeight, HVWireThickness, HVWireLength, FREQ, LVRATE, LvCD, HvCD)
    Ur = CalculateUr(Ll, POWER)
    Ucc = CalculateImpedance(Ux, Ur)
    price = CalculatePrice(LVNumberOfTurns, LVFoilHeight, LVFoilThiccnes, HVWireThickness, HVWireLength, CoreDiameter, CoreLength, LvCD, HvCD, printValues=isFinal)
    NllExtraLoss = max(0, Nll - GUARANTEEDNLL)
    LlExtraLoss = max(0, Ll - GUARANTEEDLL)
    UccExtraLoss = max(0, abs(Ucc - GUARANTEEDUCC) - abs(UCC_TOLERANCE))
    penaltyForNll = NllExtraLoss * PENALTY_NLL_FACTOR
    penaltyForLL = LlExtraLoss * PENALTY_LL_FACTOR
    penaltyforUcc = UccExtraLoss * PENALTY_UCC_FACTOR
    multiplier = 1
    if (NllExtraLoss > (GUARANTEEDNLL * tolerance / 100)) or (LlExtraLoss > (GUARANTEEDLL * tolerance / 100)) or (UccExtraLoss > (GUARANTEEDUCC * tolerance / 100)):
        multiplier = -1
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
    return (price + penaltyForNll + penaltyForLL + penaltyforUcc) * multiplier


@njit(fastmath=True)
def CalculateFinalizedPriceIntolerant_Optimized(LVNumberOfTurns, LVFoilHeight, LVFoilThiccnes, HVWireThickness, HVWireLength, CoreDiameter, CoreLength, LVRATE_VAL=LVRATE, HVRATE_VAL=HVRATE, POWER=POWERRATING, FREQ=FREQUENCY, MaterialResistivity=materialToBeUsedWire_Resistivity, GUARANTEEDNLL=GUARANTEED_NO_LOAD_LOSS, GUARANTEEDLL=GUARANTEED_LOAD_LOSS, GUARANTEEDUCC=GUARANTEED_UCC, tolerance=1, isFinal=False, PutCoolingDucts=True):
    """Optimized version that calculates load losses only once when possible."""
    LvCD = 0.0
    HvCD = 0.0
    if PutCoolingDucts:
        LvCD, HvCD, Ll_zero, LlLv_zero, LlHv_zero = CalculateNumberOfCoolingDucts_WithLosses(
            LVNumberOfTurns, LVFoilHeight, LVFoilThiccnes, HVWireThickness, HVWireLength,
            CoreDiameter, CoreLength, MaterialResistivity, POWER, HVRATE_VAL, LVRATE_VAL, isFinal
        )
        if LvCD == 0 and HvCD == 0:
            Ll, LlHv, LlLv = Ll_zero, LlHv_zero, LlLv_zero
        else:
            Ll, LlHv, LlLv = CalculateLoadLosses(LVNumberOfTurns, LVFoilThiccnes, CoreDiameter, LVFoilHeight, HVWireThickness, HVWireLength, MaterialResistivity, POWER, HVRATE_VAL, LVRATE_VAL, CoreLength, LvCD, HvCD)
    else:
        Ll, LlHv, LlLv = CalculateLoadLosses(LVNumberOfTurns, LVFoilThiccnes, CoreDiameter, LVFoilHeight, HVWireThickness, HVWireLength, MaterialResistivity, POWER, HVRATE_VAL, LVRATE_VAL, CoreLength, 0, 0)
    Nll = CalculateNoLoadLosses(LVNumberOfTurns, LVFoilHeight, LVFoilThiccnes, HVWireThickness, HVWireLength, CoreDiameter, LVRATE_VAL, CoreLength, LvCD, HvCD)
    strayDia = CalculateStrayDiameter(LVNumberOfTurns, LVFoilThiccnes, LVFoilHeight, HVWireThickness, HVWireLength, CoreDiameter, CoreLength, LvCD, HvCD)
    Ux = CalculateUx(POWER, strayDia, LVNumberOfTurns, LVFoilThiccnes, LVFoilHeight, HVWireThickness, HVWireLength, FREQ, LVRATE_VAL, LvCD, HvCD)
    Ur = CalculateUr(Ll, POWER)
    Ucc = CalculateImpedance(Ux, Ur)
    price = CalculatePrice(LVNumberOfTurns, LVFoilHeight, LVFoilThiccnes, HVWireThickness, HVWireLength, CoreDiameter, CoreLength, LvCD, HvCD, isFinal)
    NllExtraLoss = max(0, Nll - GUARANTEEDNLL)
    LlExtraLoss = max(0, Ll - GUARANTEEDLL)
    UccExtraLoss = max(0, abs(Ucc - GUARANTEEDUCC) - abs(UCC_TOLERANCE))
    penaltyForNll = NllExtraLoss * PENALTY_NLL_FACTOR
    penaltyForLL = LlExtraLoss * PENALTY_LL_FACTOR
    penaltyforUcc = UccExtraLoss * PENALTY_UCC_FACTOR
    multiplier = 1
    if (NllExtraLoss > (GUARANTEEDNLL * tolerance / 100)) or (LlExtraLoss > (GUARANTEEDLL * tolerance / 100)) or (UccExtraLoss > (GUARANTEEDUCC * tolerance / 100)):
        multiplier = -1
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
    return (price + penaltyForNll + penaltyForLL + penaltyforUcc) * multiplier


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
        CalculateFinalizedPrice(foundLVTurns, foundLVHeight, foundLVThickness, foundHVThickness, foundHVLength, foundCoreDiameter, foundCoreLength, isFinal=True, PutCoolingDucts=PutCoolingDuct)
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
                                 tolerance, obround, put_cooling_ducts,
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

                                # Basic validity checks
                                if hvlen < hvthick:
                                    continue

                                hv_layer_height = height - 50
                                hv_turns_per_layer = (hv_layer_height / (hvlen + InsulationThickness)) - 1
                                if hv_turns_per_layer <= 0:
                                    continue

                                # Use full calculation with cooling ducts and penalties
                                price = CalculateFinalizedPriceIntolerant_Optimized(
                                    turns, height, thick, hvthick, hvlen,
                                    core_dia, core_len,
                                    LVRATE_VAL, HVRATE_VAL, POWER, FREQ,
                                    MaterialResistivity,
                                    GUARANTEEDNLL, GUARANTEEDLL, GUARANTEEDUCC,
                                    tolerance, False, put_cooling_ducts
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
    def cuda_calculate_radial_thickness_hv(lv_height, lv_turns, hv_thick, hv_len, n_ducts, hv_rate, lv_rate, insulation_wire, hv_insulation_thick, duct_thick):
        hv_turns = lv_turns * (hv_rate / lv_rate)
        hv_layer_height = lv_height - 50.0
        hv_turns_per_layer = (hv_layer_height / (hv_len + insulation_wire)) - 1.0
        if hv_turns_per_layer <= 0:
            return 1e10  # Invalid
        hv_layer_number = math.ceil(hv_turns / hv_turns_per_layer)
        return hv_layer_number * hv_thick + (hv_layer_number - 1) * hv_insulation_thick + (n_ducts * (duct_thick + 0.5))

    @cuda.jit(device=True)
    def cuda_calculate_avg_diameter_hv(lv_turns, lv_thick, core_dia, lv_height, hv_thick, hv_len, core_len, n_ducts_lv, n_ducts_hv,
                                        dist_core_lv, main_gap, hv_rate, lv_rate, insulation_wire, lv_insulation_thick, hv_insulation_thick, duct_thick):
        radial_lv = cuda_calculate_radial_thickness_lv(lv_turns, lv_thick, n_ducts_lv, lv_insulation_thick, duct_thick)
        radial_hv = cuda_calculate_radial_thickness_hv(lv_height, lv_turns, hv_thick, hv_len, n_ducts_hv, hv_rate, lv_rate, insulation_wire, hv_insulation_thick, duct_thick)
        return core_dia + 2.0 * dist_core_lv + 2.0 * radial_lv + 2.0 * main_gap + radial_hv + (2.0 * core_len / math.pi)

    @cuda.jit(device=True)
    def cuda_calculate_total_length_hv(lv_turns, lv_thick, core_dia, lv_height, hv_thick, hv_len, core_len, n_ducts_lv, n_ducts_hv,
                                        dist_core_lv, main_gap, hv_rate, lv_rate, insulation_wire, lv_insulation_thick, hv_insulation_thick, duct_thick):
        hv_turns = lv_turns * hv_rate / lv_rate
        avg_dia = cuda_calculate_avg_diameter_hv(lv_turns, lv_thick, core_dia, lv_height, hv_thick, hv_len, core_len, n_ducts_lv, n_ducts_hv,
                                                  dist_core_lv, main_gap, hv_rate, lv_rate, insulation_wire, lv_insulation_thick, hv_insulation_thick, duct_thick)
        return avg_dia * math.pi * hv_turns

    @cuda.jit(device=True)
    def cuda_calculate_core_weight(lv_turns, lv_height, lv_thick, hv_thick, hv_len, core_dia, core_len, n_ducts_lv, n_ducts_hv,
                                    dist_core_lv, main_gap, phase_gap, hv_rate, lv_rate, insulation_wire,
                                    lv_insulation_thick, hv_insulation_thick, duct_thick, core_density, core_fill_round, core_fill_rect):
        window_height = lv_height + 40.0
        radial_lv = cuda_calculate_radial_thickness_lv(lv_turns, lv_thick, n_ducts_lv, lv_insulation_thick, duct_thick)
        radial_hv = cuda_calculate_radial_thickness_hv(lv_height, lv_turns, hv_thick, hv_len, n_ducts_hv, hv_rate, lv_rate, insulation_wire, hv_insulation_thick, duct_thick)
        radial_total = radial_lv + radial_hv + main_gap + dist_core_lv
        center_between_legs = (core_dia + radial_total * 2.0) + phase_gap
        rect_weight = (((3.0 * window_height) + 2.0 * (2.0 * center_between_legs + core_dia)) * (core_dia * core_len / 100.0) * core_density * core_fill_rect) / 1e6
        radius = core_dia / 2.0
        square_edge = radius * math.sqrt(math.pi)
        round_weight = (((3.0 * (window_height + 10.0)) + 2.0 * (2.0 * center_between_legs + core_dia)) * (square_edge * square_edge / 100.0) * core_density * core_fill_round) / 1e6
        return (rect_weight + round_weight) * 100.0

    @cuda.jit(device=True)
    def cuda_calculate_section_hv(thickness, length):
        return (thickness * length) - (thickness**2) + (((thickness / 2.0)**2) * math.pi)

    @cuda.jit(device=True)
    def cuda_calculate_section_lv(height, thickness):
        return height * thickness

    @cuda.jit(device=True)
    def cuda_calculate_load_losses(lv_turns, lv_thick, core_dia, lv_height, hv_thick, hv_len, resistivity, power, hv_rate, lv_rate, core_len, n_ducts_lv, n_ducts_hv,
                                    dist_core_lv, main_gap, insulation_wire, lv_insulation_thick, hv_insulation_thick, duct_thick, add_loss_lv, add_loss_hv):
        lv_length = cuda_calculate_total_length_lv(lv_turns, lv_thick, core_dia, core_len, n_ducts_lv, dist_core_lv, lv_insulation_thick, duct_thick)
        hv_length = cuda_calculate_total_length_hv(lv_turns, lv_thick, core_dia, lv_height, hv_thick, hv_len, core_len, n_ducts_lv, n_ducts_hv,
                                                    dist_core_lv, main_gap, hv_rate, lv_rate, insulation_wire, lv_insulation_thick, hv_insulation_thick, duct_thick)
        lv_section = cuda_calculate_section_lv(lv_height, lv_thick)
        hv_section = cuda_calculate_section_hv(hv_thick, hv_len)
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
                                       lv_insulation_thick, hv_insulation_thick, duct_thick, core_density, core_fill_round, core_fill_rect, freq):
        core_weight = cuda_calculate_core_weight(lv_turns, lv_height, lv_thick, hv_thick, hv_len, core_dia, core_len, n_ducts_lv, n_ducts_hv,
                                                  dist_core_lv, main_gap, phase_gap, hv_rate, lv_rate, insulation_wire,
                                                  lv_insulation_thick, hv_insulation_thick, duct_thick, core_density, core_fill_round, core_fill_rect)
        volts_per_turn = lv_rate / lv_turns
        induction = cuda_calculate_induction(volts_per_turn, core_dia, core_len, freq, core_fill_round, core_fill_rect)
        watts_per_kg = cuda_calculate_watts_per_kg(induction)
        return watts_per_kg * core_weight * 1.2

    @cuda.jit(device=True)
    def cuda_calculate_stray_diameter(lv_turns, lv_thick, lv_height, hv_thick, hv_len, core_dia, core_len, n_ducts_lv, n_ducts_hv,
                                       dist_core_lv, main_gap, hv_rate, lv_rate, insulation_wire, lv_insulation_thick, hv_insulation_thick, duct_thick):
        radial_lv = cuda_calculate_radial_thickness_lv(lv_turns, lv_thick, n_ducts_lv, lv_insulation_thick, duct_thick)
        main_gap_dia = core_dia + dist_core_lv * 2.0 + radial_lv * 2.0 + (2.0 * core_len / math.pi) + main_gap
        radial_hv = cuda_calculate_radial_thickness_hv(lv_height, lv_turns, hv_thick, hv_len, n_ducts_hv, hv_rate, lv_rate, insulation_wire, hv_insulation_thick, duct_thick)
        reduced_hv = radial_hv / 3.0
        reduced_lv = radial_lv / 3.0
        sd = main_gap_dia + reduced_hv - reduced_lv + ((reduced_hv**2 - reduced_lv**2) / (reduced_lv + reduced_hv + main_gap))
        return sd

    @cuda.jit(device=True)
    def cuda_calculate_ux(power, stray_dia, lv_turns, lv_thick, lv_height, hv_thick, hv_len, freq, lv_rate, n_ducts_lv, n_ducts_hv,
                          hv_rate, insulation_wire, lv_insulation_thick, hv_insulation_thick, duct_thick, main_gap):
        radial_hv = cuda_calculate_radial_thickness_hv(lv_height, lv_turns, hv_thick, hv_len, n_ducts_hv, hv_rate, lv_rate, insulation_wire, hv_insulation_thick, duct_thick)
        reduced_hv = radial_hv / 3.0
        radial_lv = cuda_calculate_radial_thickness_lv(lv_turns, lv_thick, n_ducts_lv, lv_insulation_thick, duct_thick)
        reduced_lv = radial_lv / 3.0
        volts_per_turn = lv_rate / lv_turns
        return (power * stray_dia * freq * (reduced_lv + reduced_hv + main_gap)) / (1210.0 * (volts_per_turn**2) * lv_height)

    @cuda.jit(device=True)
    def cuda_calculate_price(lv_turns, lv_height, lv_thick, hv_thick, hv_len, core_dia, core_len, n_ducts_lv, n_ducts_hv,
                              dist_core_lv, main_gap, phase_gap, hv_rate, lv_rate, insulation_wire,
                              lv_insulation_thick, hv_insulation_thick, duct_thick, core_density, core_fill_round, core_fill_rect,
                              core_price, foil_density, foil_price, wire_density, wire_price):
        # Core weight and price
        wc = cuda_calculate_core_weight(lv_turns, lv_height, lv_thick, hv_thick, hv_len, core_dia, core_len, n_ducts_lv, n_ducts_hv,
                                         dist_core_lv, main_gap, phase_gap, hv_rate, lv_rate, insulation_wire,
                                         lv_insulation_thick, hv_insulation_thick, duct_thick, core_density, core_fill_round, core_fill_rect)
        pc = wc * core_price

        # HV volume and price
        hv_length = cuda_calculate_total_length_hv(lv_turns, lv_thick, core_dia, lv_height, hv_thick, hv_len, core_len, n_ducts_lv, n_ducts_hv,
                                                    dist_core_lv, main_gap, hv_rate, lv_rate, insulation_wire, lv_insulation_thick, hv_insulation_thick, duct_thick)
        hv_section = cuda_calculate_section_hv(hv_thick, hv_len)
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
                                        add_loss_lv, add_loss_hv, penalty_nll, penalty_ll, penalty_ucc, max_gradient):
        """Calculate price with penalties - simplified version without cooling ducts for GPU."""
        n_ducts_lv = 0.0
        n_ducts_hv = 0.0

        # Calculate losses
        ll = cuda_calculate_load_losses(lv_turns, lv_thick, core_dia, lv_height, hv_thick, hv_len, resistivity, power, hv_rate, lv_rate, core_len, n_ducts_lv, n_ducts_hv,
                                         dist_core_lv, main_gap, insulation_wire, lv_insulation_thick, hv_insulation_thick, duct_thick, add_loss_lv, add_loss_hv)

        nll = cuda_calculate_no_load_losses(lv_turns, lv_height, lv_thick, hv_thick, hv_len, core_dia, lv_rate, core_len, n_ducts_lv, n_ducts_hv,
                                             dist_core_lv, main_gap, phase_gap, hv_rate, insulation_wire,
                                             lv_insulation_thick, hv_insulation_thick, duct_thick, core_density, core_fill_round, core_fill_rect, freq)

        # Calculate impedance
        stray_dia = cuda_calculate_stray_diameter(lv_turns, lv_thick, lv_height, hv_thick, hv_len, core_dia, core_len, n_ducts_lv, n_ducts_hv,
                                                   dist_core_lv, main_gap, hv_rate, lv_rate, insulation_wire, lv_insulation_thick, hv_insulation_thick, duct_thick)
        ux = cuda_calculate_ux(power, stray_dia, lv_turns, lv_thick, lv_height, hv_thick, hv_len, freq, lv_rate, n_ducts_lv, n_ducts_hv,
                               hv_rate, insulation_wire, lv_insulation_thick, hv_insulation_thick, duct_thick, main_gap)
        ur = ll / (10.0 * power)
        ucc = math.sqrt(ux**2 + ur**2)

        # Calculate price
        price = cuda_calculate_price(lv_turns, lv_height, lv_thick, hv_thick, hv_len, core_dia, core_len, n_ducts_lv, n_ducts_hv,
                                      dist_core_lv, main_gap, phase_gap, hv_rate, lv_rate, insulation_wire,
                                      lv_insulation_thick, hv_insulation_thick, duct_thick, core_density, core_fill_round, core_fill_rect,
                                      core_price, foil_density, foil_price, wire_density, wire_price)

        # Calculate penalties
        nll_extra = max(0.0, nll - guaranteed_nll)
        ll_extra = max(0.0, ll - guaranteed_ll)
        ucc_extra = max(0.0, abs(ucc - guaranteed_ucc) - abs(ucc_tol))

        penalty_for_nll = nll_extra * penalty_nll
        penalty_for_ll = ll_extra * penalty_ll
        penalty_for_ucc = ucc_extra * penalty_ucc

        # Check tolerance
        multiplier = 1.0
        if (nll_extra > (guaranteed_nll * tolerance / 100.0)) or \
           (ll_extra > (guaranteed_ll * tolerance / 100.0)) or \
           (ucc_extra > (guaranteed_ucc * tolerance / 100.0)):
            multiplier = -1.0

        return (price + penalty_for_nll + penalty_for_ll + penalty_for_ucc) * multiplier

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

        # Quick induction pre-filter
        core_section = cuda_calculate_core_section(core_dia, core_len, core_fill_round, core_fill_rect)
        volts_per_turn = lv_rate / turns
        induction = (volts_per_turn * 10000.0) / (math.sqrt(2.0) * math.pi * freq * core_section)

        # Skip invalid induction ranges
        if induction > 1.95 or induction < 0.8:
            results[idx, 0] = 1e18
            return

        # Basic validity checks
        if hvlen < hvthick:
            results[idx, 0] = 1e18
            return

        hv_layer_height = height - 50.0
        hv_turns_per_layer = (hv_layer_height / (hvlen + insulation_wire)) - 1.0
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
            add_loss_lv, add_loss_hv, penalty_nll, penalty_ll, penalty_ucc, max_gradient
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
    ], dtype=np.float64)

    # Allocate results array
    results = np.zeros((n_combinations, 8), dtype=np.float64)

    # Transfer to GPU
    print("Transferring data to GPU...")
    if progress_callback:
        try:
            progress_callback("CUDA GPU", 0.1, f"Transferring {n_combinations:,} combinations to GPU...", None)
        except (InterruptedError, KeyboardInterrupt):
            raise
        except:
            pass

    d_combinations = cuda.to_device(combinations)
    d_results = cuda.to_device(results)
    d_params = cuda.to_device(params)

    # Configure kernel launch
    threads_per_block = 256
    blocks_per_grid = (n_combinations + threads_per_block - 1) // threads_per_block

    print(f"Launching kernel: {blocks_per_grid} blocks x {threads_per_block} threads")

    if progress_callback:
        try:
            progress_callback("CUDA GPU", 0.3, f"Running CUDA kernel ({n_combinations:,} combinations)...", None)
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
            progress_callback("CUDA GPU", 0.9, f"Kernel complete in {kernel_time:.1f}s, processing results...", None)
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
                                best_core_dia, best_core_len, isFinal=True, PutCoolingDucts=put_cooling_ducts)
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
                       add_loss_lv, add_loss_hv, penalty_nll, penalty_ll, penalty_ucc):
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
    hv_turns_per_layer = torch.clamp((hv_layer_height / (hvlen + insulation_wire)) - 1.0, min=0.001)
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
    section_hv = (hvthick * hvlen) - (hvthick**2) + (((hvthick / 2.0)**2) * math.pi)
    resistance_lv = (total_len_lv / 1000.0) * resistivity / section_lv
    resistance_hv = (total_len_hv / 1000.0) * resistivity / section_hv

    # Currents and losses
    current_lv = (power * 1000.0) / (lv_rate * 3.0)
    current_hv = (power * 1000.0) / (hv_rate * 3.0)
    load_losses = resistance_lv * (current_lv**2) * 3.0 * add_loss_lv + resistance_hv * (current_hv**2) * 3.0 * add_loss_hv

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
    valid = (induction >= 0.8) & (induction <= 1.95) & (hvlen >= hvthick) & (hv_turns_per_layer > 0) & \
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
                       add_loss_lv, add_loss_hv, penalty_nll, penalty_ll, penalty_ucc):
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
    hv_turns_per_layer = mx.maximum((hv_layer_height / (hvlen + insulation_wire)) - 1.0, 0.001)
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
    section_hv = (hvthick * hvlen) - (hvthick**2) + (((hvthick / 2.0)**2) * math.pi)
    resistance_lv = (total_len_lv / 1000.0) * resistivity / section_lv
    resistance_hv = (total_len_hv / 1000.0) * resistivity / section_hv

    # Currents and losses
    current_lv = (power * 1000.0) / (lv_rate * 3.0)
    current_hv = (power * 1000.0) / (hv_rate * 3.0)
    load_losses = resistance_lv * (current_lv**2) * 3.0 * add_loss_lv + resistance_hv * (current_hv**2) * 3.0 * add_loss_hv

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
    valid = (induction >= 0.8) & (induction <= 1.95) & (hvlen >= hvthick) & (hv_turns_per_layer > 0) & \
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
        'penalty_ll': float(PENALTY_LL_FACTOR), 'penalty_ucc': float(PENALTY_UCC_FACTOR)
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

        # Progress
        pct = 100.0 * n_processed / n_total
        print(f"  Batch {batch_num}: {n_processed:,}/{n_total:,} ({pct:.1f}%)")

        # Call progress callback
        if progress_callback:
            try:
                progress_callback("MPS GPU", pct / 100.0, f"Batch {batch_num}: {n_processed:,}/{n_total:,}", None)
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
                                best_core_dia, best_core_len, isFinal=True, PutCoolingDucts=put_cooling_ducts)
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
        'penalty_ll': float(PENALTY_LL_FACTOR), 'penalty_ucc': float(PENALTY_UCC_FACTOR)
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

        # Progress
        pct = 100.0 * n_processed / n_total
        print(f"  Batch {batch_num}: {n_processed:,}/{n_total:,} ({pct:.1f}%)")

        # Call progress callback
        if progress_callback:
            try:
                progress_callback("MLX GPU", pct / 100.0, f"Batch {batch_num}: {n_processed:,}/{n_total:,}", None)
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
                                best_core_dia, best_core_len, isFinal=True, PutCoolingDucts=put_cooling_ducts)
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
        local_ranges = {'turns': 4, 'height': 40, 'thick': 0.4, 'hvthick': 0.4, 'hvlen': 1.5, 'core_dia': 20, 'core_len': 30}
        local_steps = {'height': 5, 'thick': 0.03, 'hvthick': 0.03, 'hvlen': 0.08, 'core_dia': 3, 'core_len': 5}
        n_regions = 5
        fine_core_len_steps = 12
    elif search_depth == 'exhaustive':
        # ~10-30min - maximum precision
        coarse_steps = {'core_dia': 15, 'core_len': 6, 'turns': 2, 'height': 25, 'thick': 0.25, 'hvthick': 0.25, 'hvlen': 0.6}
        fine_steps = {'core_dia': 2, 'turns': 1, 'height': 5, 'thick': 0.02, 'hvthick': 0.02, 'hvlen': 0.05}
        local_ranges = {'turns': 5, 'height': 50, 'thick': 0.5, 'hvthick': 0.5, 'hvlen': 2.0, 'core_dia': 25, 'core_len': 40}
        local_steps = {'height': 3, 'thick': 0.02, 'hvthick': 0.02, 'hvlen': 0.05, 'core_dia': 2, 'core_len': 3}
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
        'penalty_ll': float(PENALTY_LL_FACTOR), 'penalty_ucc': float(PENALTY_UCC_FACTOR)
    }

    # Collect all coarse results with prices
    all_prices = []
    all_params = []

    for batch in _generate_batches_streaming(core_dias, turns_arr, heights, thicks, hvthicks, hvlens, core_len_steps, obround, batch_size=2_000_000):
        combs = torch.tensor(batch, device=device)
        prices, _ = _compute_batch_mps(combs, device, **params)

        # Get valid results
        valid_mask = prices < 1e17
        valid_prices = prices[valid_mask].cpu().numpy()
        valid_combs = combs[valid_mask].cpu().numpy()

        all_prices.extend(valid_prices.tolist())
        all_params.extend(valid_combs.tolist())

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

        # Progress for this region
        region_progress = 0.34 + (0.33 * (i / len(top_candidates)))
        report_progress(2, region_progress, f"Stage 2: Region {i+1}/{len(top_candidates)}", None)

        region_best_price = 1e18
        region_best_params = None

        for batch in _generate_batches_streaming(fine_core_dias, fine_turns, fine_heights, fine_thicks, fine_hvthicks, fine_hvlens, fine_core_len_steps, obround, batch_size=2_000_000):
            combs = torch.tensor(batch, device=device)
            prices, _ = _compute_batch_mps(combs, device, **params)

            best_idx = torch.argmin(prices).item()
            best_price = prices[best_idx].item()

            if best_price < region_best_price:
                region_best_price = best_price
                region_best_params = combs[best_idx].cpu().numpy()

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

    eval_count = 0
    last_progress_report = 0
    stage3_start = time.time()

    for cd in core_dia_range:
        for cl in core_len_range:
            for t in turns_range:
                for h in height_range:
                    for th in thick_range:
                        for hvth in hvthick_range:
                            for hvl in hvlen_range:
                                if hvl < hvth:
                                    continue
                                eval_count += 1

                                # Report progress every 5000 evaluations
                                if eval_count - last_progress_report >= 5000:
                                    last_progress_report = eval_count
                                    stage3_progress = 0.68 + (0.30 * eval_count / max(n_local, 1))
                                    elapsed = time.time() - stage3_start
                                    if eval_count > 0:
                                        eta = (elapsed / eval_count) * (n_local - eval_count)
                                    else:
                                        eta = None
                                    report_progress(3, stage3_progress, f"Stage 3: {eval_count:,}/{n_local:,} ({100*eval_count/max(n_local,1):.1f}%)", eta)

                                # Evaluate WITH cooling ducts using CPU function
                                price = CalculateFinalizedPriceIntolerant_Optimized(
                                    t, h, th, hvth, hvl, cd, cl,
                                    LVRATE, HVRATE, POWERRATING, FREQUENCY,
                                    materialToBeUsedWire_Resistivity,
                                    GUARANTEED_NO_LOAD_LOSS, GUARANTEED_LOAD_LOSS, GUARANTEED_UCC,
                                    tolerance, False, put_cooling_ducts
                                )

                                if 0 < price < best_price:
                                    best_price = price
                                    best_turns = t
                                    best_height = h
                                    best_thick = th
                                    best_hvthick = hvth
                                    best_hvlen = hvl
                                    best_core_dia = cd
                                    best_core_len = cl

    stage3_time = time.time() - total_start - stage1_time - stage2_time

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
            tolerance, False, put_cooling_ducts
        )
        if best_price < 0:
            best_price = abs(best_price)
    else:
        print(f"  Evaluated {eval_count:,} designs, best price: {best_price:.2f}")

    total_time = time.time() - total_start
    report_progress(3, 1.0, f"Complete! Best price: {best_price:.2f}", None)

    if print_result:
        print(f"\n{'='*60}")
        print("HYBRID OPTIMIZATION RESULTS")
        print(f"{'='*60}")
        CalculateFinalizedPrice(best_turns, best_height, best_thick, best_hvthick, best_hvlen,
                                best_core_dia, best_core_len, isFinal=True, PutCoolingDucts=put_cooling_ducts)
        print(f"\nCore Diameter: {best_core_dia:.1f} mm")
        print(f"Core Length: {best_core_len:.1f} mm")
        print(f"LV Turns: {best_turns:.0f}")
        print(f"LV Foil Height: {best_height:.1f} mm")
        print(f"LV Foil Thickness: {best_thick:.2f} mm")
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
            float(tolerance), obround, put_cooling_ducts,
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
        CalculateFinalizedPrice(int(best[1]), best[2], best[3], best[4], best[5], best[6], best[7], isFinal=True, PutCoolingDucts=put_cooling_ducts)

    return _build_result_dict(
        lv_turns=best[1], lv_height=best[2], lv_thickness=best[3],
        hv_thickness=best[4], hv_length=best[5],
        core_diameter=best[6], core_length=best[7],
        total_price=best[0], elapsed_time=total_time, put_cooling_ducts=put_cooling_ducts
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

    if hv_length < hv_thickness or lv_turns < 5:
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
            tolerance=tolerance, PutCoolingDucts=put_cooling_ducts
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


def global_search_de(obround=True, put_cooling_ducts=True, tolerance=25, maxiter=50, popsize=10):
    """Use Differential Evolution for global optimization."""
    if not SCIPY_AVAILABLE:
        print("scipy not available, falling back to parallel grid search")
        return StartOptimized(tolerance, obround, put_cooling_ducts)

    print("Running Differential Evolution global search...")
    start_time = time.time()

    bounds = get_parameter_bounds(obround)

    result = differential_evolution(
        func=lambda x: objective_for_scipy(x, put_cooling_ducts, tolerance),
        bounds=bounds,
        maxiter=maxiter,
        popsize=popsize,
        workers=1,
        polish=True,
        seed=42,
        disp=True
    )

    elapsed = time.time() - start_time
    print(f"DE completed in {elapsed:.2f}s")

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


def StartSmartOptimized(tolerance=25, obround=True, put_cooling_ducts=True, use_de=False, n_lhs_samples=3000, print_result=True):
    """Combined smart optimization: LHS coarse search + L-BFGS-B fine search."""
    print("=" * 60)
    print("SMART OPTIMIZED TRANSFORMER SEARCH")
    print("=" * 60)
    start_time = time.time()

    if use_de:
        interim = global_search_de(obround, put_cooling_ducts, tolerance)
        best_price = interim["price"]
        lv_turns = interim["lvfoilturns"]
        lv_height = interim["lvfoilheight"]
        lv_thickness = interim["lvfoilthickness"]
        hv_thickness = interim["hvthickness"]
        hv_length = interim["hvlength"]
        core_diameter = interim["corediameter"]
        core_length = interim["corelength"]
    else:
        coarse_result = smart_coarse_search(n_lhs_samples, obround, put_cooling_ducts, tolerance)

        if coarse_result is None or coarse_result[0] is None:
            print("Coarse search failed, falling back to parallel grid search")
            return StartOptimized(tolerance, obround, put_cooling_ducts, print_result)

        initial_point, initial_price = coarse_result
        final_point, best_price = fine_search_scipy(initial_point, obround, put_cooling_ducts, tolerance)

        lv_turns = int(round(final_point[0]))
        lv_height = final_point[1]
        lv_thickness = final_point[2]
        hv_thickness = final_point[3]
        hv_length = final_point[4]
        core_diameter = final_point[5]
        core_length = final_point[6]

    total_time = time.time() - start_time

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
            core_diameter, core_length, isFinal=True, PutCoolingDucts=put_cooling_ducts
        )

    return _build_result_dict(
        lv_turns=lv_turns, lv_height=lv_height, lv_thickness=lv_thickness,
        hv_thickness=hv_thickness, hv_length=hv_length,
        core_diameter=core_diameter, core_length=core_length,
        total_price=best_price, elapsed_time=total_time, put_cooling_ducts=put_cooling_ducts
    )


# =============================================================================
# RESULT HELPER
# =============================================================================

def _build_result_dict(lv_turns, lv_height, lv_thickness, hv_thickness, hv_length,
                       core_diameter, core_length, total_price, elapsed_time, put_cooling_ducts=True):
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
            GUARANTEEDUCC=GUARANTEED_UCC
        )

    # Calculate losses - pass all parameters explicitly
    load_loss, _, _ = CalculateLoadLosses(
        lv_turns, lv_thickness, core_diameter, lv_height, hv_thickness, hv_length,
        _resistivity, _power, _hvrate, _lvrate, core_length,
        n_ducts_lv, n_ducts_hv
    )

    no_load_loss = CalculateNoLoadLosses(
        lv_turns, lv_height, lv_thickness, hv_thickness, hv_length,
        core_diameter, _lvrate, core_length, n_ducts_lv, n_ducts_hv
    )

    # Calculate impedance
    stray_dia = CalculateStrayDiameter(
        lv_turns, lv_thickness, lv_height, hv_thickness, hv_length,
        core_diameter, core_length, n_ducts_lv, n_ducts_hv
    )
    ux = CalculateUx(_power, stray_dia, lv_turns, lv_thickness, lv_height,
                     hv_thickness, hv_length, _freq, _lvrate, n_ducts_lv, n_ducts_hv)
    ur = CalculateUr(load_loss, _power)
    impedance = CalculateImpedance(ux, ur)

    # Calculate weights (multiply by 3 for 3-phase transformer)
    core_weight = CalculateCoreWeight(lv_turns, lv_height, lv_thickness, hv_thickness, hv_length,
                                      core_diameter, core_length, n_ducts_lv, n_ducts_hv)

    lv_volume = CalculateVolumeLV(lv_turns, lv_thickness, core_diameter, lv_height, core_length, n_ducts_lv) * 3
    lv_weight = CalculateWeightOfVolume(lv_volume, materialToBeUsedFoil_Density)

    hv_volume = CalculateVolumeHV(lv_turns, lv_thickness, core_diameter, lv_height, hv_thickness, hv_length,
                                  core_length, n_ducts_hv) * 3
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

def StartFast(tolerance=25, obround=True, put_cooling_ducts=True, method='de', print_result=True, grid_resolution='coarse', search_depth='normal', progress_callback=None):
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
            - 'smart': LHS + L-BFGS-B (CPU)
            - 'parallel': Numba parallel CPU grid search
        print_result: Print detailed results
        grid_resolution: For mps/mlx/gpu/parallel methods - 'coarse', 'medium', 'fine'
        search_depth: For hybrid method - 'fast', 'normal', 'thorough', 'exhaustive'
        progress_callback: Optional function(stage, progress, message, eta) for UI updates

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
    print(f"\n{'='*60}")
    print(f"TRANSFORMER OPTIMIZATION - Method: {method.upper()}")
    print(f"Available GPU backends: {', '.join(GPU_OPTIONS) if GPU_OPTIONS else 'None'}")
    print(f"{'='*60}\n")

    try:
        if method == 'hybrid':
            result = StartMPSHybrid(tolerance, obround, put_cooling_ducts, print_result, search_depth=search_depth, progress_callback=progress_callback)
        elif method == 'mps':
            result = StartMPS(tolerance, obround, put_cooling_ducts, print_result, grid_resolution=grid_resolution, progress_callback=progress_callback)
        elif method == 'mlx':
            result = StartMLX(tolerance, obround, put_cooling_ducts, print_result, grid_resolution=grid_resolution, progress_callback=progress_callback)
        elif method == 'gpu':
            result = StartGPU(tolerance, obround, put_cooling_ducts, print_result, grid_resolution=grid_resolution, progress_callback=progress_callback)
        elif method == 'parallel':
            result = StartOptimized(tolerance, obround, put_cooling_ducts, print_result, grid_resolution=grid_resolution, progress_callback=progress_callback)
        elif method == 'smart':
            result = StartSmartOptimized(tolerance, obround, put_cooling_ducts, use_de=False, print_result=print_result)
        elif method == 'de':
            result = StartSmartOptimized(tolerance, obround, put_cooling_ducts, use_de=True, print_result=print_result)
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
                put_cooling_ducts=put_cooling_ducts
            )

        return result

    except Exception as e:
        print(f"Optimization failed with error: {e}")
        print("Falling back to original BucketFillingSmart...")
        return BucketFillingSmart(printValuesFinal=print_result, tolerance=tolerance, obround=obround, PutCoolingDuct=put_cooling_ducts)


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
    CalculateRadialThiccnessHV(400.0, 20, 2.0, 5.0, 0)
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
        25.0, True, True,
        LVRATE, HVRATE, POWERRATING, FREQUENCY, materialToBeUsedWire_Resistivity,
        GUARANTEED_NO_LOAD_LOSS, GUARANTEED_LOAD_LOSS, GUARANTEED_UCC,
        CoreFillingFactorRound, CoreFillingFactorRectangular, INSULATION_THICKNESS_WIRE
    )


# Run warmup at module import
print("Pre-compiling JIT functions...", end=" ", flush=True)
_warmup_start = time.time()
_warmup_jit()
print(f"done ({time.time() - _warmup_start:.1f}s)")


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
