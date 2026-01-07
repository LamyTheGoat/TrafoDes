import math
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

from multiprocessing.pool import ThreadPool as Pool
from multiprocessing import Process,Queue
import multiprocessing

ConnectionType ={
    "D" : 1,
    "Y": 1/math.sqrt(3),
}

CORELENGTH_MINIMUM=0
CORE_MINIMUM = 70
CORE_MAXIMUM = 500
FOILHEIGHT_MINIMUM = 200
FOILHEIGHT_MAXIMUM = 1200
FOILTHICKNESS_MINIMUM = 0.3
FOILTHICKNESS_MAXIMUM = 4
FOILTURNS_MINIMUM = 5
FOILTURNS_MAXIMUM = 100
HVDIA_MINIMUM = 1
HVDIA_MAXIMUM = 5
##################################################################
#                           TDS SPECS                            #
##################################################################
HVCONNECTION = ConnectionType["D"]
LVCONNECTION = ConnectionType["Y"]

POWERRATING = 160 #400
HVRATE = 33000 * HVCONNECTION
LVRATE = 400 * LVCONNECTION
FREQUENCY = 50


GUARANTEED_NO_LOAD_LOSS =  242*0.98 #387
GUARANTEED_LOAD_LOSS =  1925*0.98 #325
GUARANTEED_UCC = 4.48

UCC_TOLERANCE_PERCENT = 3
UCC_TOLERANCE = GUARANTEED_UCC*(UCC_TOLERANCE_PERCENT/100)
#################################################################
#################################################################
#################################################################


PENALTY_NLL_FACTOR = 60
PENALTY_LL_FACTOR = 10
PENALTY_UCC_FACTOR = 10000
INSULATION_THICKNESS_WIRE=0.12

CORE_LENGTH = 122

#original main gap = 9
LVInsulationThickness = 0.125
HVInsulationThickness = 0.5
MainGap = 12
DistanceCoreLV = 2
PhaseGap = 12
CoreFillingFactorRound = 0.84
CoreFillingFactorRectangular = 0.97

CoreDensity = 7.65
CorePricePerKg = 3.6#5.5
AlPricePerKgFoil = 3.52 #3.5
AlPricePerKgWire = 4.88 #4.9
CuPricePerKg = 3.4
AlDensity = 2.7
CuDensity = 8.9
AlRestivity = 0.0336

class Material:
    materialDensity = 0
    materialPrice = 0
    materialResistivity = 0
    def __init__(self, density, price, resistance =0):
        self.materialDensity = density
        self.materialPrice = price
        self.materialResistivity = resistance

class ParametersCalculated: 
    price = 999999999999999999999999999999999999999999999999999999999999999999999999999999


AlMaterialFoil = Material(AlDensity,AlPricePerKgFoil, AlRestivity)
AlMaterialWire = Material(AlDensity, AlPricePerKgWire, AlRestivity)
CuMaterial = Material(CuDensity, CuPricePerKg)

#use this for aluminum
materialToBeUsedFoil = AlMaterialFoil
materialToBeUsedWire = AlMaterialWire
materialCore = Material(CoreDensity, CorePricePerKg)

#use this for copper
#materialToBeUsed = CuMaterial

def Clamp(Number, Minimum, Maximum):
    if Number < Minimum:
        return Minimum
    elif Number > Maximum:
        return Maximum
    else:
        return Number

def CalculateVoltsPerTurns(LVRate, LVTurns):
    return LVRate/LVTurns

def CalculateTransformer(LVNumberOfTurns, LVFoilHeight, LVFoilThiccnes, HVWireDiameter, CoreDiameter):
    #do calculations here
    return None

def CalculateCoreWeight(LVNumberOfTurns, LVFoilHeight, LVFoilThiccnes, HVWireDiameter, CoreDiameter, CoreLength):
    FoilHeight = LVFoilHeight
    WindowHeight = FoilHeight + 40
    HVNumberOfTurns = LVNumberOfTurns * HVRATE/ LVRATE
    HVLayerHeight = FoilHeight-50
    HVTurnsPerLayer = (HVLayerHeight/(HVWireDiameter+INSULATION_THICKNESS_WIRE)) -1
    HVLayerNumber = math.ceil(HVNumberOfTurns/HVTurnsPerLayer)
    RadialThickness = CalculateRadialThicknessLV(LVNumberOfTurns, LVFoilThiccnes) + CalculateRadialThiccnessHV(LVFoilHeight, LVNumberOfTurns, HVWireDiameter) + MainGap + DistanceCoreLV
    CenterBetweenLegs = (CoreDiameter +RadialThickness * 2) + PhaseGap
    SectionOfUpAndDownLegs= (CalculateCoreSection(CoreDiameter,0) * math.pi * 2 / 4) 
    #rectWeight =((3*WindowHeight*CoreLength*CoreDiameter) + (2*(CenterBetweenLegs+CoreDiameter)*(CoreLength*CoreDiameter)))#*(CoreDensity/math.pow(10,6))*(CoreFillingFactorRectangular)
    rectWeight = (((3*WindowHeight) + 2*(2*CenterBetweenLegs+CoreDiameter))*(CoreDiameter*CoreLength/100)*CoreDensity*CoreFillingFactorRectangular)/math.pow(10,6)
    section=CalculateCoreSection(CoreDiameter,CoreLength)
    radius =CoreDiameter/2
    squareEdge = radius*math.sqrt(math.pi)
    roundWeight = (((3*(WindowHeight+10)) + 2*(2*CenterBetweenLegs+CoreDiameter))*(squareEdge*squareEdge/100)*CoreDensity*CoreFillingFactorRound)/math.pow(10,6)
    #roundWeight = ( ( ( 3*WindowHeight*CalculateCoreSection( CoreDiameter,0 ) ) + ( 2* (2*CenterBetweenLegs+CoreDiameter) * (SectionOfUpAndDownLegs) ) -((8-2*math.pi)*math.pow(CoreDiameter/2,3)/100) ) * (CoreDensity/math.pow(10,6) ) * (CoreFillingFactorRound) )
    CW = rectWeight + roundWeight
    #basic version update later
    return CW * 100

def CalculateInduction(VolsPerTurn, CoreDiameter,CoreLength):
    return (VolsPerTurn * 10000)/(math.sqrt(2) * math.pi * FREQUENCY * CalculateCoreSection(CoreDiameter,CoreLength))

def CalculateWattsPerKG(Induction):
    return (1.3498 * math.pow(Induction,6)) + (-8.1737 * math.pow(Induction,5)) + (19.884 * math.pow(Induction,4)) + (-24.708 * math.pow(Induction,3)) + (16.689 * math.pow(Induction,2)) + (-5.5386 * Induction) + (0.7462)

def CalculateCoreSection(CoreDiameter, CoreLength ):
    return ((math.pow(CoreDiameter,2)*math.pi)/(4*100) ) *CoreFillingFactorRound + (CoreLength*CoreDiameter/100) * CoreFillingFactorRectangular

def CalculateRadialThicknessLV(LVNumberOfTurns, LVFoilThiccnes):
    return LVNumberOfTurns * LVFoilThiccnes + ((LVNumberOfTurns-1)*LVInsulationThickness)

def CalculateAverageDiameterLV(LVNumberOfTurns, LVFoilThiccnes, CoreDiameter, CoreLength):
    return CoreDiameter + (2 * DistanceCoreLV) + CalculateRadialThicknessLV(LVNumberOfTurns, LVFoilThiccnes) + (2*CoreLength/math.pi)

def CalculateTotalLengthCoilLV(LVNumberOfTurns, LVFoilThiccnes, CoreDiameter,CoreLength):

    return CalculateAverageDiameterLV(LVNumberOfTurns, LVFoilThiccnes, CoreDiameter,CoreLength) * math.pi * LVNumberOfTurns

def CalculateRadialThiccnessHV(LVFoilHeight, LVNumberOfTurns, HVWireDiameter):
    FoilHeight = LVFoilHeight
    HVNumberOfTurns = LVNumberOfTurns * (HVRATE/ LVRATE)
    HVLayerHeight = FoilHeight-50
    HVTurnsPerLayer = (HVLayerHeight/(HVWireDiameter+INSULATION_THICKNESS_WIRE)) -1
    HVLayerNumber = math.ceil(HVNumberOfTurns/HVTurnsPerLayer)
    return HVLayerNumber*HVWireDiameter + (HVLayerNumber-1) * HVInsulationThickness 

def CalculateAverageDiameterHV(LVNumberOfTurns, LVFoilThiccnes, CoreDiameter, LVFoilHeight, HVWireDiameter, CoreLength):
    return CoreDiameter + 2*DistanceCoreLV + 2*CalculateRadialThicknessLV(LVNumberOfTurns, LVFoilThiccnes) + 2 * MainGap + CalculateRadialThiccnessHV(LVFoilHeight, LVNumberOfTurns, HVWireDiameter) + (2*CoreLength/math.pi)

def CalculateTotalLengthCoilHV(LVNumberOfTurns, LVFoilThiccnes, CoreDiameter, LVFoilHeight, HVWireDiameter,CoreLength):
    FoilHeight = LVFoilHeight
    HVNumberOfTurns = LVNumberOfTurns * HVRATE/ LVRATE
    #Number of Turns = " + str(HVNumberOfTurns))
    HVLayerHeight = FoilHeight-20
    HVTurnsPerLayer = (HVLayerHeight/HVWireDiameter) -1
    HVLayerNumber = math.ceil(HVNumberOfTurns/HVTurnsPerLayer)
    return CalculateAverageDiameterHV(LVNumberOfTurns, LVFoilThiccnes, CoreDiameter, LVFoilHeight, HVWireDiameter,CoreLength) * math.pi * HVNumberOfTurns

def CalculateVolumeLV(LVNumberOfTurns, LVFoilThiccnes, CoreDiameter, LVFoilHeight,CoreLength):
    length = CalculateTotalLengthCoilLV(LVNumberOfTurns, LVFoilThiccnes, CoreDiameter,CoreLength)
    return (length * LVFoilHeight * LVFoilThiccnes) / (1000000)

def CalculateVolumeHV(LVNumberOfTurns, LVFoilThiccnes, CoreDiameter, LVFoilHeight, HVWireDiameter,CoreLength):
    length = CalculateTotalLengthCoilHV(LVNumberOfTurns, LVFoilThiccnes, CoreDiameter, LVFoilHeight, HVWireDiameter,CoreLength)
    return (length * math.pi * math.pow(HVWireDiameter,2)) / (4 * 1000000)

def CalculateWeightOfVolume(Volume, Material):
    return Volume * Material.materialDensity

def CalculatePriceOfWeight(Weight, Material):
    return Weight * Material.materialPrice

def CalculateResistanceLV(Material, Length, Section):
    lengthByMeters = Length/1000
    return lengthByMeters * Material.materialResistivity / Section

def CalculateResistanceHV(Material, Length, Section):
    lengthByMeters = Length/1000
    return lengthByMeters * Material.materialResistivity / Section

def CalculateCurrent(Power,Voltage):
    #triangle
    return (Power * 1000)/(Voltage*3)

def CalculateCurrentLV(Power, VoltageLV):
    return CalculateCurrent(Power, VoltageLV)

def CalculateCurrentHV(Power, VoltageHV):
    return CalculateCurrent(Power, VoltageHV)

def CalculateSectionHV(Diameter):
    return math.pi * math.pow(Diameter,2) / 4

def CalculateSectionLV(HeightLV, ThiccnessLV):
    return HeightLV * ThiccnessLV

def CalculateLoadLosses(LVNumberOfTurns, LVFoilThiccnes, CoreDiameter, LVFoilHeight, HVWireDiameter, Material, Power, HVRating, LVRating,CoreLength):
    lvLength = CalculateTotalLengthCoilLV(LVNumberOfTurns, LVFoilThiccnes, CoreDiameter,CoreLength)
    hvLength = CalculateTotalLengthCoilHV(LVNumberOfTurns, LVFoilThiccnes, CoreDiameter, LVFoilHeight, HVWireDiameter,CoreLength)
    lvSection = CalculateSectionLV(LVFoilHeight, LVFoilThiccnes)
    hvSection = CalculateSectionHV(HVWireDiameter)
    lvResistance = CalculateResistanceLV(Material, lvLength, lvSection)
    hvResistance = CalculateResistanceHV(Material, hvLength, hvSection)
    hvCurrent = CalculateCurrentHV(Power, HVRating)
    lvCurrent = CalculateCurrentLV(Power, LVRating)

    #print("RESISTANCE LV = " + str(lvResistance))
    #print("RESISTANCE HV = " + str(hvResistance))

    #print("CURRENT LV = " + str(lvCurrent))
    #print("CURRENT HV = " + str(hvCurrent))

    losses = (lvResistance * math.pow(lvCurrent,2) + hvResistance * math.pow(hvCurrent,2)) * 3 * 1.04 #1.16 # a constant change it carefully if you will
    return losses

def CalculateNoLoadLosses(LVNumberOfTurns, LVFoilHeight, LVFoilThiccnes, HVWireDiameter, CoreDiameter, LVRate,CoreLength):
    CoreWeight =  CalculateCoreWeight(LVNumberOfTurns, LVFoilHeight, LVFoilThiccnes, HVWireDiameter, CoreDiameter,CoreLength)
    Induction = CalculateInduction(CalculateVoltsPerTurns(LVRate, LVNumberOfTurns), CoreDiameter,CoreLength)
    WattsPerKG = CalculateWattsPerKG(Induction)
    return WattsPerKG * CoreWeight * 1.2 #some constant multiplier change it if you will

def CalculateStrayDiameter(LVNumberOfTurns,LVFoilThiccness, LVFoilHeight, HVWireDiameter, DiameterOfCore, CoreLength):
    MainGapDiameter = DiameterOfCore + DistanceCoreLV*2 + CalculateRadialThicknessLV(LVNumberOfTurns, LVFoilThiccness)*2 + (2*CoreLength/math.pi) + MainGap
    HVRadialThickness = CalculateRadialThiccnessHV(LVFoilHeight,LVNumberOfTurns, HVWireDiameter)
    ReducedWidthHV = HVRadialThickness/3
    LVRadialThickness = CalculateRadialThicknessLV(LVNumberOfTurns, LVFoilThiccness)
    ReducedWidthLV = LVRadialThickness/3
    SD = MainGapDiameter + ReducedWidthHV - ReducedWidthLV + (    (math.pow(ReducedWidthHV,2)-math.pow(ReducedWidthLV,2))/(ReducedWidthLV+ReducedWidthHV+MainGap)    )
    return SD

def CalculateUr(LoadLosses, Power):
    return (LoadLosses) /(10 * Power)

def CalculateUx(Power, StrayDiameter, LVNumberOfTurns,LVFoilThiccness, LVFoilHeight, HVWireDiameter, Frequency, LVRate):
    HVRadialThickness = CalculateRadialThiccnessHV(LVFoilHeight,LVNumberOfTurns, HVWireDiameter)
    ReducedWidthHV = HVRadialThickness/3
    LVRadialThickness = CalculateRadialThicknessLV(LVNumberOfTurns, LVFoilThiccness)
    ReducedWidthLV = LVRadialThickness/3
    return (Power * StrayDiameter * Frequency * (ReducedWidthLV + ReducedWidthHV + MainGap)) / (1210 * math.pow(CalculateVoltsPerTurns(LVRate,LVNumberOfTurns),2) * LVFoilHeight)


def CalculateImpedance(Ux, Ur):
    return math.sqrt(math.pow(Ux,2) + math.pow(Ur,2))

def CalculatePrice(LVNumberOfTurns, LVFoilHeight, LVFoilThiccnes, HVWireDiameter, CoreDiameter,CoreLength, printValues=False):
    wc = CalculateCoreWeight(LVNumberOfTurns, LVFoilHeight, LVFoilThiccnes, HVWireDiameter, CoreDiameter,CoreLength)
    pc = CalculatePriceOfWeight(wc, materialCore)

    volumeHV = CalculateVolumeHV(LVNumberOfTurns, LVFoilThiccnes, CoreDiameter, LVFoilHeight, HVWireDiameter,CoreLength) * 3
    whv =CalculateWeightOfVolume(volumeHV, materialToBeUsedWire)
    phv = CalculatePriceOfWeight(whv,materialToBeUsedWire)

    volumeLV = CalculateVolumeLV(LVNumberOfTurns, LVFoilThiccnes, CoreDiameter, LVFoilHeight,CoreLength) * 3
    wlv = CalculateWeightOfVolume(volumeLV, materialToBeUsedFoil)
    plv = CalculatePriceOfWeight(wlv, materialToBeUsedFoil)

    if(printValues):
        print("CORE WEIGHT = " + str(wc))
        print("FOIL WEIGHT = " + str(wlv))
        print("WIRE WEIGHT = " + str(whv))
    return pc + phv + plv

def CalculateFinalizedPrice(LVNumberOfTurns, LVFoilHeight, LVFoilThiccnes, HVWireDiameter, CoreDiameter,CoreLength, LVRATE=LVRATE, HVRATE = HVRATE, POWER = POWERRATING, FREQ = FREQUENCY, Material = materialToBeUsedWire, GUARANTEEDNLL = GUARANTEED_NO_LOAD_LOSS, GUARANTEEDLL = GUARANTEED_LOAD_LOSS, GUARANTEEDUCC=GUARANTEED_UCC, isFinal=False):
    #HVWireDiameter += INSULATION_THICKNESS
    Nll = CalculateNoLoadLosses(LVNumberOfTurns, LVFoilHeight, LVFoilThiccnes, HVWireDiameter, CoreDiameter, LVRATE,CoreLength)
    Ll = CalculateLoadLosses(LVNumberOfTurns, LVFoilThiccnes, CoreDiameter, LVFoilHeight, HVWireDiameter, Material, POWER, HVRATE, LVRATE,CoreLength)
    inductioncalculated = CalculateInduction(CalculateVoltsPerTurns(LVRATE, LVNumberOfTurns), CoreDiameter,CoreLength)
    strayDia = CalculateStrayDiameter(LVNumberOfTurns,LVFoilThiccnes,LVFoilHeight,HVWireDiameter,CoreDiameter,CoreLength)
    Ux = CalculateUx(POWER, strayDia, LVNumberOfTurns, LVFoilThiccnes, LVFoilHeight, HVWireDiameter, FREQ, LVRATE)
    Ur = CalculateUr(Ll, POWER)
    Ucc = CalculateImpedance(Ux, Ur)
    price = CalculatePrice(LVNumberOfTurns, LVFoilHeight, LVFoilThiccnes, HVWireDiameter, CoreDiameter,CoreLength,isFinal)

    penaltyForNll= max(0, Nll- GUARANTEEDNLL) * PENALTY_NLL_FACTOR
    penaltyForLL = max(0, Ll - GUARANTEEDLL) * PENALTY_LL_FACTOR
    penaltyforUcc = max(0, abs(Ucc- GUARANTEED_UCC) - abs(UCC_TOLERANCE)) * PENALTY_UCC_FACTOR

    if(isFinal):
        print("#######################################################")
        print("BARE PRICE = " + str(price))
        print("NLL penalty = " +str(penaltyForNll))
        print("LL penalty = " +str(penaltyForLL))
        print("UCC Penalty = " + str(penaltyforUcc))
        print("No load losses " + str(Nll))
        print("Load losses = " + str(Ll))
        print("Ucc  = " +str(Ucc))
        print("Price Is = " + str(price + penaltyForNll + penaltyForLL + penaltyforUcc))

    return price + penaltyForNll + penaltyForLL + penaltyforUcc

def CalculateFinalizedPriceIntolerant(LVNumberOfTurns, LVFoilHeight, LVFoilThiccnes, HVWireDiameter, CoreDiameter,CoreLength, LVRATE=LVRATE, HVRATE = HVRATE, POWER = POWERRATING, FREQ = FREQUENCY, Material = materialToBeUsedWire, GUARANTEEDNLL = GUARANTEED_NO_LOAD_LOSS, GUARANTEEDLL = GUARANTEED_LOAD_LOSS, GUARANTEEDUCC=GUARANTEED_UCC, tolerance =1, isFinal = False):
    #HVWireDiameter += INSULATION_THICKNESS
    Nll = CalculateNoLoadLosses(LVNumberOfTurns, LVFoilHeight, LVFoilThiccnes, HVWireDiameter, CoreDiameter, LVRATE,CoreLength)
    Ll = CalculateLoadLosses(LVNumberOfTurns, LVFoilThiccnes, CoreDiameter, LVFoilHeight, HVWireDiameter, Material, POWER, HVRATE, LVRATE,CoreLength)
    inductioncalculated = CalculateInduction(CalculateVoltsPerTurns(LVRATE, LVNumberOfTurns), CoreDiameter,CoreLength)
    strayDia = CalculateStrayDiameter(LVNumberOfTurns,LVFoilThiccnes,LVFoilHeight,HVWireDiameter,CoreDiameter, CoreLength)
    Ux = CalculateUx(POWER, strayDia, LVNumberOfTurns, LVFoilThiccnes, LVFoilHeight, HVWireDiameter, FREQ, LVRATE)
    Ur = CalculateUr(Ll, POWER)
    Ucc = CalculateImpedance(Ux, Ur)
    price = CalculatePrice(LVNumberOfTurns, LVFoilHeight, LVFoilThiccnes, HVWireDiameter, CoreDiameter,CoreLength,printValues=isFinal)

    NllExtraLoss=max(0,Nll-GUARANTEEDNLL)
    LlExtraLoss = max(0,Ll - GUARANTEEDLL)
    UccExtraLoss =max(0, abs(Ucc- GUARANTEEDUCC) - abs(UCC_TOLERANCE)) 
    penaltyForNll= NllExtraLoss * PENALTY_NLL_FACTOR
    penaltyForLL = LlExtraLoss * PENALTY_LL_FACTOR
    penaltyforUcc = UccExtraLoss * PENALTY_UCC_FACTOR

    multiplier = 1
    if((NllExtraLoss > (GUARANTEEDNLL*tolerance/100)) or (LlExtraLoss >(GUARANTEEDLL*tolerance/100)) or (UccExtraLoss>(GUARANTEEDUCC*tolerance/100))):
        multiplier = -1
    if(isFinal):
        print("#######################################################")
        print("BARE PRICE = " + str(price))
        print("NLL penalty = " +str(penaltyForNll))
        print("LL penalty = " +str(penaltyForLL))
        print("UCC Penalty = " + str(penaltyforUcc))
        print("No load losses " + str(Nll))
        print("Load losses = " + str(Ll))
        print("Ucc  = " +str(Ucc))
        print("Price Is = " + str(price + penaltyForNll + penaltyForLL + penaltyforUcc))


    return (price + penaltyForNll + penaltyForLL + penaltyforUcc) * multiplier















#inductioncalculated = CalculateInduction(CalculateVoltsPerTurns(LVRATE, lv_turns_t), core_diameter_t)
##strayDia = CalculateStrayDiameter(lv_turns_t,LV_THICCNESS_TEST,lv_foilheight_t,hv_wire_diameter_t,core_diameter_t)
#Ux = CalculateUx(POWERRATING, strayDia, lv_turns_t, LV_THICCNESS_TEST, lv_foilheight_t, hv_wire_diameter_t, FREQUENCY, LVRATE)
#print("UX = " + str(Ux))


#CalculateStrayDiameter(lv_turns_t, LV_THICCNESS_TEST, lv_foilheight_t,hv_wire_diameter_t,core_diameter_t)
#print("INDUCTION = " + str(inductioncalculated))
#print("WATTS/KG = " + str(CalculateWattsPerKG(inductioncalculated)))
#print("LV AVERAGE DIA = " + str(CalculateAverageDiameterLV(lv_turns_t, LV_THICCNESS_TEST, core_diameter_t)))
#print("LV TOTAL LENGTH = " + str(CalculateTotalLengthCoilLV(lv_turns_t,LV_THICCNESS_TEST,core_diameter_t)))
#print("HV AVERAGE DIA = " + str(CalculateAverageDiameterHV(lv_turns_t,LV_THICCNESS_TEST,core_diameter_t,lv_foilheight_t,hv_wire_diameter_t)))
#print("HV TOTAL LENGTH = " + str(CalculateTotalLengthCoilHV(lv_turns_t,LV_THICCNESS_TEST,core_diameter_t,lv_foilheight_t,hv_wire_diameter_t)))
#print("CORE WEIGHT = " + str(CalculateCoreWeight(lv_turns_t,lv_foilheight_t,LV_THICCNESS_TEST,hv_wire_diameter_t,core_diameter_t)))
#print("LOAD LOSSES = " + str(CalculateLoadLosses(lv_turns_t,LV_THICCNESS_TEST,core_diameter_t,lv_foilheight_t,hv_wire_diameter_t, materialToBeUsed, POWERRATING, HVRATE, LVRATE)))
#print("NO LOAD LOSSES = " + str(CalculateNoLoadLosses(lv_turns_t, lv_foilheight_t, LV_THICCNESS_TEST, hv_wire_diameter_t, core_diameter_t, LVRATE)))
#print("Total Price = " + str(CalculatePrice(lv_turns_t, lv_foilheight_t, LV_THICCNESS_TEST, hv_wire_diameter_t, core_diameter_t)))
#print("Total Price With Penalties = " + str(CalculateFinalizedPrice(lv_turns_t, lv_foilheight_t, LV_THICCNESS_TEST, hv_wire_diameter_t, core_diameter_t)))

########################################################################################################
##TEST AREA
########################################################################################################


#global bestTURNS
#global bestprice
#bestprice = 999999999999999999999999999999999999999
#global bestTHICC
#global bestFH
#global bestCD
#global besHVD

#bestTURNS=0
#bestTHICC=0
#bestFH=0
#bestCD=0
#besHVD=0


def RandomizedCalculations(ite,returnlist):
    for i in range(ite):
        global bestprice
        global bestTURNS
        global bestTHICC
        global bestCD
        global bestFH
        global besHVD
        lv_turns_tB = random.randrange(0,61)*1 + 10
        LV_THICCNESS_TESTB = random.randrange(0,19)*0.2 + 0.4
        lv_foilheight_tB = random.randrange(0,21)*50 + 200
        core_diameter_tB = random.randrange(0,81)*5 + 100
        hv_wire_diameter_tB = random.randrange(0,16)*0.2 + 1
        CoreLength =0

        inductioncalculatedB = CalculateInduction(CalculateVoltsPerTurns(LVRATE, lv_turns_tB), core_diameter_tB)
        strayDiaB = CalculateStrayDiameter(lv_turns_tB,LV_THICCNESS_TESTB,lv_foilheight_tB,hv_wire_diameter_tB,core_diameter_tB, CoreLength)
        loadLossB = CalculateLoadLosses(lv_turns_tB, LV_THICCNESS_TESTB, core_diameter_tB, lv_foilheight_tB, hv_wire_diameter_tB, materialToBeUsedFoil, POWERRATING, HVRATE, LVRATE)
        UxB = CalculateUx(POWERRATING, strayDiaB, lv_turns_tB, LV_THICCNESS_TESTB, lv_foilheight_tB, hv_wire_diameter_tB, FREQUENCY, LVRATE)
        UrB = CalculateUr(loadLossB,POWERRATING)
        wpkg = CalculateWattsPerKG(inductioncalculatedB)
        dlv = CalculateAverageDiameterLV(lv_turns_tB, LV_THICCNESS_TESTB, core_diameter_tB)
        tlclv = CalculateTotalLengthCoilLV(lv_turns_tB,LV_THICCNESS_TESTB,core_diameter_tB)
        avgdiahv = CalculateAverageDiameterHV(lv_turns_tB,LV_THICCNESS_TESTB,core_diameter_tB,lv_foilheight_tB,hv_wire_diameter_tB)
        tlchv = CalculateTotalLengthCoilHV(lv_turns_tB,LV_THICCNESS_TESTB,core_diameter_tB,lv_foilheight_tB,hv_wire_diameter_tB)
        cw =CalculateCoreWeight(lv_turns_tB,lv_foilheight_tB,LV_THICCNESS_TESTB,hv_wire_diameter_tB,core_diameter_tB)
        ll =CalculateLoadLosses(lv_turns_tB,LV_THICCNESS_TESTB,core_diameter_tB,lv_foilheight_tB,hv_wire_diameter_tB, materialToBeUsedWire, POWERRATING, HVRATE, LVRATE)
        nll =CalculateNoLoadLosses(lv_turns_tB, lv_foilheight_tB, LV_THICCNESS_TESTB, hv_wire_diameter_tB, core_diameter_tB, LVRATE)
        price = CalculatePrice(lv_turns_tB, lv_foilheight_tB, LV_THICCNESS_TESTB, hv_wire_diameter_tB, core_diameter_tB)
        if(price<bestprice):
            bestprice = price
            bestFH = lv_foilheight_tB
            bestCD = core_diameter_tB
            besHVD = hv_wire_diameter_tB
            bestTHICC = LV_THICCNESS_TESTB
            bestTURNS = lv_turns_tB
    returnlist.append(bestprice)

def RandomCalcThread():
    bestobj = ParametersCalculated()
    timestart = time.time()
    manager = multiprocessing.Manager()
    returnlist = manager.list()


    p1 = multiprocessing.Process(target=RandomizedCalculations,args=(125000, returnlist))
    p2 = multiprocessing.Process(target=RandomizedCalculations,args=(125000, returnlist))
    p3 = multiprocessing.Process(target=RandomizedCalculations,args=(125000, returnlist))
    p4 = multiprocessing.Process(target=RandomizedCalculations,args=(125000, returnlist))
    p5 = multiprocessing.Process(target=RandomizedCalculations,args=(125000, returnlist))
    p6 = multiprocessing.Process(target=RandomizedCalculations,args=(125000, returnlist))
    p7 = multiprocessing.Process(target=RandomizedCalculations,args=(125000, returnlist))
    p8 = multiprocessing.Process(target=RandomizedCalculations,args=(125000, returnlist))
    print ("Test has stated")

    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()
    p6.start()
    p7.start()
    p8.start()
    p1.join() 
    p2.join() 
    p3.join()
    p4.join() 
    p5.join() 
    p6.join() 
    p7.join()
    p8.join() 






    timeend = time.time()
    elapsed = timeend - timestart
    print("Test has been finished in time :" + str(elapsed))
    returnlist.sort()
    print(returnlist)
    print("best found price = " + str(returnlist[0]))
    print("best found TURNS = " + str(bestTURNS))
    print("best found THICKNESS = " + str(bestTHICC))
    print("best found FOIL HEIGHT = " + str(bestFH))
    print("best found core diameter = " + str(bestCD))
    print("best found HV diameter = " + str(besHVD))

def BucketFilling(TurnsStep = 1, ThicknessStep = 0.2, HeightStep = 50, CoreDiaStep = 30, HVWireDiaStep = 0.2):
    #lv_turns_t = 10
    #lv_thickness_t = 0.3
    #lv_foilheight_t = 200
    #core_diameter_t = 90
    #hv_wire_diameter_t = 1
    lv_turns_t = 10
    lv_thickness_t = 0.3
    lv_foilheight_t = 200
    core_diameter_t = 90
    hv_wire_diameter_t = 1

    value = -1
    iteration = 0

    while core_diameter_t < 500:
        print("Core Diameter is:" + str(int(core_diameter_t)))
        lv_thickness_t = 0.3
        lv_foilheight_t = 200
        lv_turns_t = 10
        hv_wire_diameter_t = 1

        while lv_turns_t < 70:
            lv_foilheight_t = 200
            lv_thickness_t = 0.3
            hv_wire_diameter_t = 1

            while lv_foilheight_t < 1200:
                lv_thickness_t = 0.3
                hv_wire_diameter_t = 1
                
                while  lv_thickness_t < 4:
                    hv_wire_diameter_t = 1

                    while hv_wire_diameter_t < 4:
                        value = CalculateFinalizedPriceIntolerant(lv_turns_t, lv_foilheight_t, lv_thickness_t, hv_wire_diameter_t, core_diameter_t)
                        iteration +=1
                        if((iteration % 1000000) == 0):
                            print("iteration has made" + str(iteration/1000000) + "M steps")
                        if(value > 0):
                            print(value)
                            print("LV TURNS IS = " + str(lv_turns_t))
                            print("LV THICKNESS IS = " + str(lv_thickness_t))
                            print("LV HEIGHT IS = " + str(lv_foilheight_t))
                            print("CORE DIAMETER IS = " + str(core_diameter_t))
                            print("HV DIAMETER IS = " + str(hv_wire_diameter_t))
                            return value
                        hv_wire_diameter_t += HVWireDiaStep
                    lv_thickness_t += ThicknessStep

                lv_foilheight_t += HeightStep
            lv_turns_t += TurnsStep

        core_diameter_t += CoreDiaStep




    
    return (value)


def BucketFillingSmart(TurnsStep = 1, ThicknessStep = 0.2, HeightStep = 50, CoreDiaStep = 30,CoreLengthStep=1, HVWireDiaStep = 0.2, TurnsStepMinimum = 1, FoilHeightStepMinimum = 5, FoilThicknessStepMinimum = 0.05, CoreDiameterStepMinimum =1, CoreLengthStepMinimum =1, HVDiameterStepMinimum = 0.05, BrakeDistance=5, tolerance=1, printValues = False,obround=True):
    #lv_turns_t = 10
    #lv_thickness_t = 0.3
    #lv_foilheight_t = 200
    #core_diameter_t = 90
    #hv_wire_diameter_t = 1
    if(obround):
        CoreDiameterStepMinimum=10



    core_length_t = CORELENGTH_MINIMUM
    lv_turns_t = FOILTURNS_MINIMUM
    lv_thickness_t = FOILTHICKNESS_MINIMUM
    lv_foilheight_t = FOILHEIGHT_MINIMUM
    core_diameter_t = CORE_MINIMUM
    hv_wire_diameter_t = HVDIA_MINIMUM
    startTime = time.time()
    value = -1
    iteration = 0
    loopEnded = False
    while (core_diameter_t < CORE_MAXIMUM) and (not loopEnded):
        if(printValues):
                print("Core Diameter is:" + str(int(core_diameter_t)))
        lv_thickness_t = FOILTHICKNESS_MINIMUM
        lv_foilheight_t = FOILHEIGHT_MINIMUM
        lv_turns_t = FOILTURNS_MINIMUM
        hv_wire_diameter_t = HVDIA_MINIMUM
        core_length_t = CORELENGTH_MINIMUM
        CoreLengthStep = core_diameter_t/5
        obroundStackMax=0
        if(obround):
            obroundStackMax=core_diameter_t
        while(core_length_t<=obroundStackMax) and (not loopEnded):
            if(printValues):
                print("Obround Stack is:" + str(int(core_length_t)))
            
            lv_thickness_t = FOILTHICKNESS_MINIMUM
            lv_foilheight_t = FOILHEIGHT_MINIMUM
            lv_turns_t = FOILTURNS_MINIMUM
            hv_wire_diameter_t = HVDIA_MINIMUM

            while (lv_turns_t < FOILTURNS_MAXIMUM) and (not loopEnded):
                lv_foilheight_t = FOILHEIGHT_MINIMUM
                lv_thickness_t = FOILTHICKNESS_MINIMUM
                hv_wire_diameter_t = HVDIA_MINIMUM

                while (lv_foilheight_t < FOILHEIGHT_MAXIMUM) and (not loopEnded):
                    lv_thickness_t = FOILTHICKNESS_MINIMUM
                    hv_wire_diameter_t = HVDIA_MINIMUM             
                    while  (lv_thickness_t < FOILTHICKNESS_MAXIMUM) and (not loopEnded):
                        hv_wire_diameter_t = HVDIA_MINIMUM
                        while (hv_wire_diameter_t < HVDIA_MAXIMUM) and (not loopEnded):
                            value = CalculateFinalizedPriceIntolerant(lv_turns_t, lv_foilheight_t, lv_thickness_t, hv_wire_diameter_t, core_diameter_t,tolerance=tolerance,CoreLength=core_length_t)
                            iteration +=1
                            if((iteration % 1000000) == 0):
                                if(printValues):
                                    print("iteration has made" + str(iteration/1000000) + "M steps")
                            if(value > 0):
                                if(printValues):
                                    print("################################################")
                                    print("First Loop Is Completed With Values")
                                    print("PRICE = " + str(value))
                                    print("LV TURNS IS = " + str(lv_turns_t))
                                    print("LV THICKNESS IS = " + str(lv_thickness_t))
                                    print("LV HEIGHT IS = " + str(lv_foilheight_t))
                                    print("CORE DIAMETER IS = " + str(core_diameter_t))
                                    print("HV DIAMETER IS = " + str(hv_wire_diameter_t))
                                    print("################################################")
                                loopEnded = True
                            hv_wire_diameter_t += HVWireDiaStep
                        lv_thickness_t += ThicknessStep
                    lv_foilheight_t += HeightStep
                lv_turns_t += TurnsStep
            core_length_t += CoreLengthStep
        core_diameter_t += CoreDiaStep


    coreLengthNewStartPoint = CORELENGTH_MINIMUM
    coreLengthNewEndPoint =0
    coreDiameterNewStartPoint = max(core_diameter_t - CoreDiaStep*2, CORE_MINIMUM)
    foilTurnsNewStartPoint = max(lv_turns_t - 30*TurnsStep,FOILTURNS_MINIMUM)
    foilHeightNewStartPoint = max(lv_foilheight_t - 10*HeightStep, FOILHEIGHT_MINIMUM)
    foilThicknessNewStartPoint = max(lv_thickness_t - 6* ThicknessStep, FOILTHICKNESS_MINIMUM)
    hvWireDiameterNewStartPoint = max(hv_wire_diameter_t -6*HVWireDiaStep, HVDIA_MINIMUM)
    coreDiameterNewEndPoint = CORE_MAXIMUM
    foilTurnsNewEndPoint = min(lv_turns_t + TurnsStep, FOILTURNS_MAXIMUM)
    foilHeightNewEndPoint = min(lv_foilheight_t + HeightStep,FOILHEIGHT_MAXIMUM)
    foilThicknessNewEndPoint = min(lv_thickness_t + ThicknessStep, FOILTHICKNESS_MAXIMUM)
    hvWireDiameterNewEndPoint = min(hv_wire_diameter_t + HVWireDiaStep,HVDIA_MAXIMUM)
    loopEnded = False
    CountDown = BrakeDistance +1

    transformerFound = {

        "price" : 9999999999999999999999999999999999999999999999999999999,
        "corediameter" : 0,
        "hvdiameter" : 0,
        "lvfoilheight" : 0,
        "lvfoilturns" : 0,
        "lvfoilthickness" : 0,
        "corelength":0
    }
    core_diameter_t = coreDiameterNewStartPoint
    while (core_diameter_t < coreDiameterNewEndPoint) and (CountDown>0):
        if(printValues):
            print("Core Diameter is:" + str(int(core_diameter_t)))
        lv_thickness_t = foilThicknessNewStartPoint
        lv_foilheight_t = foilHeightNewStartPoint
        lv_turns_t = foilTurnsNewStartPoint
        hv_wire_diameter_t = hvWireDiameterNewStartPoint
        core_length_t = coreLengthNewStartPoint
        CoreLengthStepMinimum = core_diameter_t/10
        if(obround):
            coreLengthNewEndPoint=core_diameter_t
        while(core_length_t<=coreLengthNewEndPoint):
            lv_thickness_t = foilThicknessNewStartPoint
            lv_foilheight_t = foilHeightNewStartPoint
            lv_turns_t = foilTurnsNewStartPoint
            hv_wire_diameter_t = hvWireDiameterNewStartPoint

            while (lv_turns_t < foilTurnsNewEndPoint):
                lv_foilheight_t = foilHeightNewStartPoint
                lv_thickness_t = foilThicknessNewStartPoint
                hv_wire_diameter_t = hvWireDiameterNewStartPoint

                while (lv_foilheight_t < foilHeightNewEndPoint):
                    lv_thickness_t = foilThicknessNewStartPoint
                    hv_wire_diameter_t = hvWireDiameterNewStartPoint
                    
                    while  (lv_thickness_t < foilThicknessNewEndPoint):
                        hv_wire_diameter_t = hvWireDiameterNewStartPoint

                        while (hv_wire_diameter_t < hvWireDiameterNewEndPoint):

                            value = CalculateFinalizedPriceIntolerant(lv_turns_t, lv_foilheight_t, lv_thickness_t, hv_wire_diameter_t, core_diameter_t,CoreLength=core_length_t,tolerance=tolerance)

                            iteration +=1
                            if((iteration % 1000000) == 0):
                                print("iteration has made" + str(iteration/1000000) + "M steps")
                            if(value > 0):
                                if(not loopEnded):
                                    print("FIRST TRANSFORMER FOUND SOON THE PROCESS WILL FINISH")
                                if(value < transformerFound["price"]):
                                    transformerFound["price"] = value
                                    transformerFound["corediameter"] = core_diameter_t
                                    transformerFound["hvdiameter"] = hv_wire_diameter_t
                                    transformerFound["lvfoilheight"] = lv_foilheight_t
                                    transformerFound["lvfoilthickness"] = lv_thickness_t
                                    transformerFound["lvfoilturns"] = lv_turns_t
                                    transformerFound["corelength"] = core_length_t
                                loopEnded = True
                            hv_wire_diameter_t += HVDiameterStepMinimum
                        lv_thickness_t += FoilThicknessStepMinimum
                    lv_foilheight_t += FoilHeightStepMinimum
                lv_turns_t += TurnsStepMinimum
            core_length_t+=CoreLengthStepMinimum
        if(loopEnded):
            CountDown = CountDown -1
        core_diameter_t += CoreDiameterStepMinimum

    endTime = time.time()
    totalTime = endTime-startTime
    if(printValues):
        CalculateFinalizedPrice(transformerFound["lvfoilturns"], transformerFound["lvfoilheight"], transformerFound["lvfoilthickness"], transformerFound["hvdiameter"], transformerFound["corediameter"],transformerFound["corelength"],isFinal=True)
        print("####################################################")
        print("Core Diameter = " + str(transformerFound["corediameter"]))
        print("Core Length = " + str(transformerFound["corelength"]))  
        print("HV Diameter = " + str(transformerFound["hvdiameter"]))
        print("Foil Height = " + str(transformerFound["lvfoilheight"]))
        print("Foil Turns = " + str(transformerFound["lvfoilturns"]))
        print("Foil Thickness = " + str(transformerFound["lvfoilthickness"]))
        print("THE PRICE IS =" + str(transformerFound["price"]))
        print("The process took : " + str(totalTime) + " seconds")
        print("####################################################")
    
    
    return (transformerFound)

def BucketFillingFast(TurnsStep = 1, ThicknessStep = 0.2, HeightStep = 50, CoreDiaStep = 30,CoreLengthStep=1, HVWireDiaStep = 0.2, TurnsStepMinimum = 1, FoilHeightStepMinimum = 5, FoilThicknessStepMinimum = 0.05, CoreDiameterStepMinimum =1, CoreLengthStepMinimum =1, HVDiameterStepMinimum = 0.05, BrakeDistance=5, tolerance=1, printValues = False,obround=True):
    if obround:
        CoreDiameterStepMinimum = 10


    coreLengthNewStartPoint = CORELENGTH_MINIMUM
    coreLengthNewEndPoint =0
    coreDiameterNewStartPoint = CORE_MINIMUM
    foilTurnsNewStartPoint = FOILTURNS_MINIMUM
    foilHeightNewStartPoint = FOILHEIGHT_MINIMUM
    foilThicknessNewStartPoint = FOILTHICKNESS_MINIMUM
    hvWireDiameterNewStartPoint = HVDIA_MINIMUM
    coreDiameterNewEndPoint = CORE_MAXIMUM
    foilTurnsNewEndPoint = FOILTURNS_MAXIMUM
    foilHeightNewEndPoint = FOILHEIGHT_MAXIMUM
    foilThicknessNewEndPoint = FOILTHICKNESS_MAXIMUM
    hvWireDiameterNewEndPoint = HVDIA_MAXIMUM

    CoreDiaStep = CoreDiameterStepMinimum
    CoreLengthStep = CoreLengthStepMinimum
    HeightStep = FoilHeightStepMinimum
    ThicknessStep = FoilThicknessStepMinimum
    TurnsStep = TurnsStepMinimum
    HVWireDiaStep = HVDiameterStepMinimum

    core_diameter_t = coreDiameterNewStartPoint
    core_length_t = coreLengthNewStartPoint
    lv_turns_t = foilTurnsNewStartPoint
    lv_foilheight_t = foilHeightNewStartPoint
    lv_thickness_t = foilThicknessNewStartPoint
    hv_wire_diameter_t = hvWireDiameterNewStartPoint

    iteration=0
    valueBefore = 0

    CoreDiaStepIncrement=0
    CoreLengthStepIncrement =0
    HeightStepIncrement =0
    ThicknessStepIncrement =0
    TurnsStepIncrement =0
    HVWireDiaStepIncrement =0
    
    

    loopEnded = False
    CountDown = BrakeDistance +1

    foundOne = False

    startTime =time.time()

    transformerFound = {
        "price" : 9999999999999999999999999999999999999999999999999999999,
        "corediameter" : 0,
        "hvdiameter" : 0,
        "lvfoilheight" : 0,
        "lvfoilturns" : 0,
        "lvfoilthickness" : 0,
        "corelength":0
    }
    core_diameter_t = coreDiameterNewStartPoint
    while (core_diameter_t <= coreDiameterNewEndPoint) and (not loopEnded):
        if(printValues):
            print("Core Diameter is:" + str(int(core_diameter_t)))


        lv_thickness_t = foilThicknessNewStartPoint
        lv_foilheight_t = foilHeightNewStartPoint
        lv_turns_t = foilTurnsNewStartPoint
        hv_wire_diameter_t = hvWireDiameterNewStartPoint
        core_length_t = coreLengthNewStartPoint

        CoreLengthStep = CoreLengthStepMinimum
        HeightStep = FoilHeightStepMinimum
        ThicknessStep = FoilThicknessStepMinimum
        TurnsStep = TurnsStepMinimum
        HVWireDiaStep = HVDiameterStepMinimum




        if(obround):
            coreLengthNewEndPoint=core_diameter_t
        while(core_length_t<=coreLengthNewEndPoint):
            lv_thickness_t = foilThicknessNewStartPoint
            lv_foilheight_t = foilHeightNewStartPoint
            lv_turns_t = foilTurnsNewStartPoint
            hv_wire_diameter_t = hvWireDiameterNewStartPoint

            HeightStep = FoilHeightStepMinimum
            ThicknessStep = FoilThicknessStepMinimum
            TurnsStep = TurnsStepMinimum
            HVWireDiaStep = HVDiameterStepMinimum

            while (lv_turns_t <= foilTurnsNewEndPoint):
                lv_foilheight_t = foilHeightNewStartPoint
                lv_thickness_t = foilThicknessNewStartPoint
                hv_wire_diameter_t = hvWireDiameterNewStartPoint

                HeightStep = FoilHeightStepMinimum
                ThicknessStep = FoilThicknessStepMinimum
                HVWireDiaStep = HVDiameterStepMinimum

                while (lv_foilheight_t <= foilHeightNewEndPoint):
                    lv_thickness_t = foilThicknessNewStartPoint
                    hv_wire_diameter_t = hvWireDiameterNewStartPoint

                    ThicknessStep = FoilThicknessStepMinimum
                    HVWireDiaStep = HVDiameterStepMinimum
                    
                    while  (lv_thickness_t <= foilThicknessNewEndPoint):
                        hv_wire_diameter_t = hvWireDiameterNewStartPoint
                        HVWireDiaStep = HVDiameterStepMinimum

                        while (hv_wire_diameter_t <= hvWireDiameterNewEndPoint):



                            value = CalculateFinalizedPriceIntolerant(lv_turns_t, lv_foilheight_t, lv_thickness_t, hv_wire_diameter_t, core_diameter_t,CoreLength=core_length_t,tolerance=tolerance)



                            iteration +=1
                            if((iteration % 1000000) == 0):
                                print("iteration has made" + str(iteration/1000000) + "M steps")


                            if(value > 0):

                                foilTurnsNewEndPoint = lv_turns_t+TurnsStep
                                foilHeightNewEndPoint = lv_foilheight_t+HeightStep
                                foilThicknessNewEndPoint = lv_thickness_t+ThicknessStep
                                hvWireDiameterNewEndPoint = hv_wire_diameter_t+HVWireDiaStep
                                coreDiameterNewEndPoint = core_diameter_t+CoreDiaStep
                                coreLengthNewEndPoint = core_length_t+CoreLengthStep

                                foilTurnsNewStartPoint = max(lv_turns_t - 3*TurnsStep,FOILTURNS_MINIMUM)
                                foilHeightNewStartPoint = max(lv_foilheight_t -3*HeightStep,FOILHEIGHT_MINIMUM)
                                foilThicknessNewStartPoint = max(lv_thickness_t - 3*ThicknessStep,FOILTHICKNESS_MINIMUM)
                                hvWireDiameterNewStartPoint = max(hv_wire_diameter_t - 3*HVWireDiaStep,HVDIA_MINIMUM)
                                coreDiameterNewStartPoint = max(core_diameter_t - 3*CoreDiaStep,CORE_MINIMUM)
                                coreLengthNewStartPoint = max(core_length_t - 3*CoreLengthStep,CORELENGTH_MINIMUM)

                                CoreDiaStep = CoreDiameterStepMinimum
                                CoreLengthStep = CoreLengthStepMinimum
                                HeightStep = FoilHeightStepMinimum
                                ThicknessStep = FoilThicknessStepMinimum
                                TurnsStep = TurnsStepMinimum
                                HVWireDiaStep = HVDiameterStepMinimum


                                if(value==valueBefore):
                                    loopEnded=True
                                valueBefore=value
                                if(value < transformerFound["price"]):
                                    transformerFound["price"] = value
                                    transformerFound["corediameter"] = core_diameter_t
                                    transformerFound["hvdiameter"] = hv_wire_diameter_t
                                    transformerFound["lvfoilheight"] = lv_foilheight_t
                                    transformerFound["lvfoilthickness"] = lv_thickness_t
                                    transformerFound["lvfoilturns"] = lv_turns_t
                                    transformerFound["corelength"] = core_length_t
                                #sloopEnded = True

                                core_diameter_t = coreDiameterNewStartPoint
                                core_length_t = coreLengthNewStartPoint
                                lv_turns_t = foilTurnsNewStartPoint
                                lv_foilheight_t = foilHeightNewStartPoint
                                lv_thickness_t = foilThicknessNewStartPoint
                                hv_wire_diameter_t = hvWireDiameterNewStartPoint

                                foundOne=True

                            if(foundOne):
                                break
                            else:
                                hv_wire_diameter_t += HVWireDiaStep
                                HVWireDiaStep += HVDiameterStepMinimum
                        if(foundOne):
                            break
                        else:
                            lv_thickness_t += ThicknessStep
                            ThicknessStep += FoilThicknessStepMinimum
                    if(foundOne):
                        break
                    else:
                        lv_foilheight_t += HeightStep
                        HeightStep += FoilHeightStepMinimum
                if(foundOne):
                    break
                else:
                    lv_turns_t += TurnsStep
                    TurnsStep += TurnsStepMinimum
            if(foundOne):
                break
            else:
                core_length_t+=CoreLengthStep
                CoreLengthStep += CoreLengthStepMinimum
        if(foundOne):
            foundOne=False
        else:
            core_diameter_t += CoreDiaStep
            CoreDiaStep += CoreDiameterStepMinimum

    endTime = time.time()
    totalTime = endTime-startTime
    if(printValues):
        CalculateFinalizedPrice(transformerFound["lvfoilturns"], transformerFound["lvfoilheight"], transformerFound["lvfoilthickness"], transformerFound["hvdiameter"], transformerFound["corediameter"],transformerFound["corelength"],isFinal=True)
        print("####################################################")
        print("Core Diameter = " + str(transformerFound["corediameter"]))
        print("Core Length = " + str(transformerFound["corelength"]))  
        print("HV Diameter = " + str(transformerFound["hvdiameter"]))
        print("Foil Height = " + str(transformerFound["lvfoilheight"]))
        print("Foil Turns = " + str(transformerFound["lvfoilturns"]))
        print("Foil Thickness = " + str(transformerFound["lvfoilthickness"]))
        print("THE PRICE IS =" + str(transformerFound["price"]))
        print("The process took : " + str(totalTime) + " seconds")
        print("####################################################")
    
    
    return (transformerFound)



def DoEverythingDumb(returns=None, TurnsStep = 1, ThicknessStep = 0.2, HeightStep = 50, CoreDiaStep = 50, HVWireDiaStep = 0.2,
 CoreDiaStart = 90, CoreDiaEnd= 500,
 LvTurnsStart = 10, LvTurnsEnd= 70,
 LVThicknessStart =0.3, LvThicknessEnd =4,
 LVHeightStart = 200, LvHeightEnd = 1200,
 HvDiaStart =1, HvDiaEnd=4,
 printValues = False):
    #lv_turns_t = 10
    #lv_thickness_t = 0.3
    #lv_foilheight_t = 200
    #core_diameter_t = 90
    #hv_wire_diameter_t = 1
    lv_turns_t = LvTurnsStart
    lv_thickness_t = LVThicknessStart
    lv_foilheight_t = LVHeightStart
    core_diameter_t = CoreDiaStart
    hv_wire_diameter_t = HvDiaStart

    value = -1
    minimumValue=9999999999999999999999999999999999
    mv_turns =0
    mv_thick =0
    mv_height =0
    mv_core = 0
    mv_hv = 0
    iteration = 0

    while core_diameter_t < CoreDiaEnd:
        if(printValues):
            print("Core Diameter is:" + str(int(core_diameter_t)) + "/" + str(int(CoreDiaEnd)))
        lv_thickness_t = 0.3
        lv_foilheight_t = 200
        lv_turns_t = 10
        hv_wire_diameter_t = 1

        while lv_thickness_t < LvThicknessEnd:
            lv_foilheight_t = 200
            lv_turns_t = 10
            hv_wire_diameter_t = 1

            while lv_foilheight_t < LvHeightEnd:
                lv_turns_t = 10
                hv_wire_diameter_t = 1
                
                while lv_turns_t < LvTurnsEnd:
                    hv_wire_diameter_t = 1

                    while hv_wire_diameter_t < HvDiaEnd:
                        value = CalculateFinalizedPrice(lv_turns_t, lv_foilheight_t, lv_thickness_t, hv_wire_diameter_t, core_diameter_t)
                        iteration +=1
                        if((iteration % 1000000) == 0):
                            if(printValues):
                                print("iteration has made" + str(iteration/1000000) + "M steps")
                        if(value < minimumValue):
                            minimumValue = value
                            mv_height=lv_foilheight_t
                            mv_core= core_diameter_t
                            mv_hv = hv_wire_diameter_t
                            mv_thick = lv_thickness_t
                            mv_turns =lv_turns_t

                        hv_wire_diameter_t += HVWireDiaStep
                    lv_turns_t += TurnsStep
                lv_foilheight_t += HeightStep
            lv_thickness_t += ThicknessStep
        core_diameter_t += CoreDiaStep

    CalculateFinalizedPrice(mv_turns, mv_height, mv_thick, mv_hv, mv_core,isFinal=printValues)
    if(printValues):
        print("####################################################")
        print("Core Diameter = " + str(mv_core))
        print("HV Diameter = " + str(mv_hv))
        print("Foil Height = " + str(mv_height))
        print("Foil Turns = " + str(mv_turns))
        print("Foil Thickness = " + str(mv_thick))
        print("THE PRICE IS =" + str(minimumValue))
        print("####################################################")
    transformerFound = {
        "price" : minimumValue,
        "corediameter" : mv_core,
        "hvdiameter" : mv_hv,
        "lvfoilheight" : mv_height,
        "lvfoilturns" : mv_turns,
        "lvfoilthickness" : mv_thick

    }
    if(returns != None):
        returns.append(transformerFound)
    return (transformerFound)

def DoEverythingThread(TurnsStep = 1, ThicknessStep = 0.2, HeightStep = 50, CoreDiaStep = 50, HVWireDiaStep = 0.2,

 CoreDiaStartT = 90, CoreDiaEndT= 500,
 LvTurnsStart = 10, LvTurnsEnd= 70,
 LVThicknessStart =0.3, LvThicknessEnd =4,
 LVHeightStart = 200, LvHeightEnd = 1200,
 HvDiaStart =1, HvDiaEnd=4, 
 printValues = True):


    manager = multiprocessing.Manager()
    returnlist = manager.list()

    ThreadStep = (round((round((CoreDiaEndT-CoreDiaStartT)/CoreDiaStep))/8))*CoreDiaStep
    p1Start = CoreDiaStartT
    p1End = CoreDiaStartT + ThreadStep
    p2Start = p1End
    p2End = p2Start + ThreadStep
    p3Start = p2End
    p3End = p3Start + ThreadStep
    p4Start = p3End
    p4End = p4Start + ThreadStep
    p5Start = p4End
    p5End = p5Start + ThreadStep
    p6Start = p5End
    p6End = p6Start+ThreadStep
    p7Start = p6End
    p7End = p7Start + ThreadStep
    p8Start = p7End
    p8End = max(CoreDiaEndT,(p8Start+ThreadStep))

    p1 = multiprocessing.Process(target=DoEverythingDumb,args=(returnlist, TurnsStep, ThicknessStep, HeightStep, CoreDiaStep, HVWireDiaStep, p1Start, p1End,LvTurnsStart,LvTurnsEnd,LVThicknessStart,LvThicknessEnd,LVHeightStart,LvHeightEnd,HvDiaStart,HvDiaEnd,False))
    p2 = multiprocessing.Process(target=DoEverythingDumb,args=(returnlist, TurnsStep, ThicknessStep, HeightStep, CoreDiaStep, HVWireDiaStep, p2Start, p2End,LvTurnsStart,LvTurnsEnd,LVThicknessStart,LvThicknessEnd,LVHeightStart,LvHeightEnd,HvDiaStart,HvDiaEnd,False))
    p3 = multiprocessing.Process(target=DoEverythingDumb,args=(returnlist, TurnsStep, ThicknessStep, HeightStep, CoreDiaStep, HVWireDiaStep, p3Start, p3End,LvTurnsStart,LvTurnsEnd,LVThicknessStart,LvThicknessEnd,LVHeightStart,LvHeightEnd,HvDiaStart,HvDiaEnd,False))
    p4 = multiprocessing.Process(target=DoEverythingDumb,args=(returnlist, TurnsStep, ThicknessStep, HeightStep, CoreDiaStep, HVWireDiaStep, p4Start, p4End,LvTurnsStart,LvTurnsEnd,LVThicknessStart,LvThicknessEnd,LVHeightStart,LvHeightEnd,HvDiaStart,HvDiaEnd,False))
    p5 = multiprocessing.Process(target=DoEverythingDumb,args=(returnlist, TurnsStep, ThicknessStep, HeightStep, CoreDiaStep, HVWireDiaStep, p5Start, p5End,LvTurnsStart,LvTurnsEnd,LVThicknessStart,LvThicknessEnd,LVHeightStart,LvHeightEnd,HvDiaStart,HvDiaEnd,False))
    p6 = multiprocessing.Process(target=DoEverythingDumb,args=(returnlist, TurnsStep, ThicknessStep, HeightStep, CoreDiaStep, HVWireDiaStep, p6Start, p6End,LvTurnsStart,LvTurnsEnd,LVThicknessStart,LvThicknessEnd,LVHeightStart,LvHeightEnd,HvDiaStart,HvDiaEnd,False))
    p7 = multiprocessing.Process(target=DoEverythingDumb,args=(returnlist, TurnsStep, ThicknessStep, HeightStep, CoreDiaStep, HVWireDiaStep, p7Start, p7End,LvTurnsStart,LvTurnsEnd,LVThicknessStart,LvThicknessEnd,LVHeightStart,LvHeightEnd,HvDiaStart,HvDiaEnd,False))
    p8 = multiprocessing.Process(target=DoEverythingDumb,args=(returnlist, TurnsStep, ThicknessStep, HeightStep, CoreDiaStep, HVWireDiaStep, p8Start, p8End,LvTurnsStart,LvTurnsEnd,LVThicknessStart,LvThicknessEnd,LVHeightStart,LvHeightEnd,HvDiaStart,HvDiaEnd,False))
    TimeStart = time.time()
    if(printValues):
        print ("Test has started")
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()
    p6.start()
    p7.start()
    p8.start()

    p1.join() 
    p2.join() 
    p3.join()
    p4.join()
    p5.join() 
    p6.join() 
    p7.join()
    p8.join()
    TimeEnd = time.time()
    if(printValues):
        print("Test has ended in = " + str(TimeEnd-TimeStart) + " seconds")
        print(returnlist[0])
        print(returnlist[1])
        print(returnlist[2])
        print(returnlist[3])

    cheapestTransformer = returnlist[0]
    for transformer in returnlist:
        if(transformer["price"] < cheapestTransformer["price"]):
            cheapestTransformer = transformer
    if(printValues):
        CalculateFinalizedPrice(cheapestTransformer["lvfoilturns"], cheapestTransformer["lvfoilheight"], cheapestTransformer["lvfoilthickness"], cheapestTransformer["hvdiameter"], cheapestTransformer["corediameter"],isFinal=True)
        print("####################################################")
        print("Core Diameter = " + str(cheapestTransformer["corediameter"]))
        print("HV Diameter = " + str(cheapestTransformer["hvdiameter"]))
        print("Foil Height = " + str(cheapestTransformer["lvfoilheight"]))
        print("Foil Turns = " + str(cheapestTransformer["lvfoilturns"]))
        print("Foil Thickness = " + str(cheapestTransformer["lvfoilthickness"]))
        print("THE PRICE IS =" + str(cheapestTransformer["price"]))
        print("####################################################")

    return cheapestTransformer

def SearchWindow(TurnsStep = 1, ThicknessStep = 0.2, HeightStep = 50, CoreDiaStep = 50, HVWireDiaStep = 0.2,
                 

 CoreDiaStartT = 90, CoreDiaEndT= 500, CoreDiaStepReduction =5,
 LvTurnsStart = 10, LvTurnsEnd= 70, LVTurnsStepReduction =1,
 LVThicknessStart =0.3, LvThicknessEnd =4, LVThicknessStepReduction =0.05,
 LVHeightStart = 200, LvHeightEnd = 1200, LVHeightStepReduction = 5,
 HvDiaStart =1, HvDiaEnd=4, HVDiaStepReduction =0.05, 
 printValues = True,):



    StartTime = time.time()
    HighIteration = 1
    cheapestTransformer=DoEverythingThread(TurnsStep, ThicknessStep, HeightStep, CoreDiaStep, HVWireDiaStep, CoreDiaStartT, CoreDiaEndT, LvTurnsStart, LvTurnsEnd, LVThicknessStart, LvThicknessEnd, LVHeightStart, LvHeightEnd, HvDiaStart, HvDiaEnd, False)
    CoreDiaMinimumStep =1 
    FoilHeightMinimumStep = 5
    FoilThicknessMinimumStep = 0.05
    FoilTurnsMinimumStep = 1
    HVDiaMinimumStep = 0.05

    TTRCore =((math.ceil((CoreDiaStep-CoreDiaMinimumStep)/CoreDiaStepReduction))+1)
    TTRTurns =((math.ceil((TurnsStep-FoilTurnsMinimumStep)/LVTurnsStepReduction))+1)
    TTRThickness = ((math.ceil((ThicknessStep-FoilThicknessMinimumStep)/LVThicknessStepReduction))+1)
    TTRHeight = ((math.ceil((HeightStep-FoilHeightMinimumStep)/LVHeightStepReduction))+1)
    TTRWireDia = ((math.ceil((HVWireDiaStep-HVDiaMinimumStep)/HVDiaStepReduction))+1)

    totalHighIterations = max(TTRCore,TTRTurns,TTRThickness,TTRHeight,TTRWireDia)

    print("Now is High Iteration = " + str(HighIteration) + "/" + str(totalHighIterations))
    
    while((CoreDiaStep > CoreDiaMinimumStep) or (HeightStep> FoilHeightMinimumStep) or (ThicknessStep > FoilThicknessMinimumStep) or (HVWireDiaStep > HVDiaMinimumStep) or(TurnsStep > FoilTurnsMinimumStep)):
        #CORE ADJUSTMENTS
        CoreDiaStartT = cheapestTransformer["corediameter"] - (5*CoreDiaStep)
        CoreDiaEndT = cheapestTransformer["corediameter"] + (5*CoreDiaStep)
        if((CoreDiaStep-CoreDiaStepReduction) > CoreDiaMinimumStep):
            CoreDiaStep = CoreDiaStep-CoreDiaStepReduction
        else:
            CoreDiaStep = CoreDiaMinimumStep
        if(CoreDiaStartT <CORE_MINIMUM):
            CoreDiaStartT = CORE_MINIMUM
        if(CoreDiaEndT > CORE_MAXIMUM):
            CoreDiaEndT= CORE_MAXIMUM


        #FOILHEIGHT ADJUSTMENTS
        LVHeightStart = cheapestTransformer["lvfoilheight"] - (5*HeightStep)
        LvHeightEnd = cheapestTransformer["lvfoilheight"] + (5*HeightStep)
        if((HeightStep - LVHeightStepReduction) > FoilHeightMinimumStep):
            HeightStep = HeightStep - LVHeightStepReduction
        else:
            HeightStep = FoilHeightMinimumStep

        if(LVHeightStart < FOILHEIGHT_MINIMUM):
            LVHeightStart = FOILHEIGHT_MINIMUM
        if(LvHeightEnd > FOILHEIGHT_MAXIMUM):
            LvHeightEnd = FOILHEIGHT_MAXIMUM

        #FOILTHICKNESS ADJUSTMENTS
        LVThicknessStart = cheapestTransformer["lvfoilthickness"] - (5*ThicknessStep)
        LvThicknessEnd = cheapestTransformer["lvfoilthickness"] + (5*ThicknessStep)
        if((ThicknessStep - LVThicknessStepReduction) > FoilThicknessMinimumStep):
            ThicknessStep = ThicknessStep - LVThicknessStepReduction
        else:
            ThicknessStep = FoilThicknessMinimumStep
        if(LVThicknessStart < FOILTHICKNESS_MINIMUM):
            LVThicknessStart = FOILTHICKNESS_MINIMUM
        if(LvThicknessEnd>FOILTHICKNESS_MAXIMUM):
            LvThicknessEnd = FOILTHICKNESS_MAXIMUM
        

        
        #FOILTURNS ADJUSTMENTS
        LvTurnsStart = cheapestTransformer["lvfoilturns"] - (5*TurnsStep)
        LvTurnsEnd = cheapestTransformer["lvfoilturns"] + (5*TurnsStep)
        if((TurnsStep-LVTurnsStepReduction)>FoilTurnsMinimumStep):
            TurnsStep = TurnsStep-LVTurnsStepReduction
        else:
            TurnsStep = FoilTurnsMinimumStep

        if(LvTurnsStart<FOILTURNS_MINIMUM):
            LvTurnsStart = FOILTURNS_MINIMUM
        if(LvTurnsEnd>FOILTURNS_MAXIMUM):
            LvTurnsEnd = FOILTURNS_MAXIMUM

        #HVDIAMETER ADJUSTMENTS
        HvDiaStart = cheapestTransformer["hvdiameter"] - (5*HVWireDiaStep)
        HvDiaEnd = cheapestTransformer["hvdiameter"] + (5*HVWireDiaStep)
        if((HVWireDiaStep-HVDiaStepReduction)>HVDiaMinimumStep):
            HVWireDiaStep = HVWireDiaStep-HVDiaStepReduction
        else:
            HVWireDiaStep = HVDiaMinimumStep


        if(HvDiaStart <HVDIA_MINIMUM):
            HvDiaStart = HVDIA_MINIMUM
        if(HvDiaEnd > HVDIA_MAXIMUM):
            HvDiaEnd = HVDIA_MAXIMUM

        #CALCULATE THE NEXT TRANSFORMER
        HighIterationTimeStart = time.time()
        transformerCalculated =DoEverythingThread(TurnsStep, ThicknessStep, HeightStep, CoreDiaStep, HVWireDiaStep, CoreDiaStartT, CoreDiaEndT, LvTurnsStart, LvTurnsEnd, LVThicknessStart, LvThicknessEnd, LVHeightStart, LvHeightEnd, HvDiaStart, HvDiaEnd, False)
        HighIterationTimeEnd = time.time()
        thisHighIterationTime = HighIterationTimeEnd-HighIterationTimeStart

        HighIteration +=1
        print("Now is High Iteration = " + str(HighIteration) + "/" + str(totalHighIterations))
        print("Estimated Remaining Time Is : " + str((totalHighIterations-HighIteration+1)*thisHighIterationTime))

        #COMPARE THE CALCULATED TRANSFORMER TO OUR BEST
        if(transformerCalculated["price"]< cheapestTransformer["price"]):
            cheapestTransformer = transformerCalculated
        #else:
        #    break
        


    endTime = time.time()

    if(printValues):
        CalculateFinalizedPrice(cheapestTransformer["lvfoilturns"], cheapestTransformer["lvfoilheight"], cheapestTransformer["lvfoilthickness"], cheapestTransformer["hvdiameter"], cheapestTransformer["corediameter"],isFinal=True)
        print("####################################################")
        print("Core Diameter = " + str(cheapestTransformer["corediameter"]))
        print("HV Diameter = " + str(cheapestTransformer["hvdiameter"]))
        print("Foil Height = " + str(cheapestTransformer["lvfoilheight"]))
        print("Foil Turns = " + str(cheapestTransformer["lvfoilturns"]))
        print("Foil Thickness = " + str(cheapestTransformer["lvfoilthickness"]))
        print("THE PRICE IS =" + str(cheapestTransformer["price"]))
        print("####################################################")
        print("PROCESS FINISHED IN : " + str(int(endTime-StartTime)) + " SECONDS")

    return cheapestTransformer

def CalculateGradient(LVTurns, LVThickness, LVHeight, CoreDiameter,CoreLength, HVWireDiameter, TurnsStepMinimum = 1, FoilHeightStepMinimum = 5, FoilThicknessStepMinimum = 0.05, CoreDiameterStepMinimum =1, CoreLengthStepMinimum =1, HVDiameterStepMinimum = 0.05,  printValues = False,obround=True):

    InitialPoint = CalculateFinalizedPrice(LVTurns,LVHeight,LVThickness,HVWireDiameter,CoreDiameter,CoreLength)
    gradients = {
        "Slope":0,
        "CoreDiameter":0,
        "CoreLength":0,
        "LVTurns":0,
        "LVHeight":0,
        "LVThickness":0,
        "HVDiameter":0
    }

    for CoreDiaGradient in range(-1,1,1):
        for CoreLengthGradient in range(-1,1,1):
            for TurnsGradient in range(-1,1,1):
                for LVHeightGradient in range(-1,1,1):
                    for LVThicknessGradient in range(-1,1,1):
                        for HVDiameterGradient  in range(-1,1,1):

                            calculatedPoint =CalculateFinalizedPrice(LVTurns+(TurnsGradient*TurnsStepMinimum),
                                                                     LVHeight+(LVHeightGradient*FoilHeightStepMinimum),
                                                                     LVThickness+(LVThicknessGradient*FoilThicknessStepMinimum),
                                                                     HVWireDiameter+(HVDiameterGradient*HVDiameterStepMinimum),
                                                                     CoreDiameter+(CoreDiaGradient*CoreDiameterStepMinimum),
                                                                     CoreLength+(CoreLengthGradient*CoreLengthStepMinimum)
                                                                    )
                            difference = calculatedPoint-InitialPoint
                            if(difference<=gradients["Slope"]):
                                gradients["Slope"]=difference
                                gradients["CoreDiameter"] = CoreDiaGradient
                                gradients["CoreLength"] = CoreLengthGradient
                                gradients["HVDiameter"]=HVDiameterGradient
                                gradients["LVHeight"]=LVHeightGradient
                                gradients["LVThickness"] = LVThicknessGradient
                                gradients["LVTurns"] = TurnsGradient
                                        


    gradients["Slope"] = abs(gradients["Slope"])    
    return gradients

def CalculateStochasticGradient(LVTurns, LVThickness, LVHeight, CoreDiameter,CoreLength, HVWireDiameter, TurnsStepMinimum = 1, FoilHeightStepMinimum = 5, FoilThicknessStepMinimum = 0.05, CoreDiameterStepMinimum =1, CoreLengthStepMinimum =1, HVDiameterStepMinimum = 0.05, sampleSize=3, ErrorPow = 1,  printValues = False,obround=True):


    indexes = [0,1,2,3,4,5]
    samples = random.sample(indexes,sampleSize)
    CoreDiameterR = 1 if 0 in samples else 0
    CoreLengthR = 1 if 1 in samples else 0
    LVTurnsR = 1 if 2 in samples else 0
    LVHeightR = 1 if 3 in samples else 0
    LVThicknessR = 1 if 4 in samples else 0
    HVDiaR = 1 if 5 in samples else 0
    
    InitialPoint = CalculateFinalizedPrice(LVTurns,LVHeight,LVThickness,HVWireDiameter,CoreDiameter,CoreLength)
    gradients = {
        "Slope":0,
        "CoreDiameter":0,
        "CoreLength":0,
        "LVTurns":0,
        "LVHeight":0,
        "LVThickness":0,
        "HVDiameter":0
    }

    for CoreDiaGradient in range(-1*CoreDiameterR,(1*CoreDiameterR)+1,1):
        for CoreLengthGradient in range(-1*CoreLengthR,(1*CoreLengthR)+1,1):
            for TurnsGradient in range(-1*LVTurnsR,(1*LVTurnsR)+1,1):
                for LVHeightGradient in range(-1*LVHeightR,(1*LVHeightR)+1,1):
                    for LVThicknessGradient in range(-1*LVThicknessR,(1*LVThicknessR)+1,1):
                        for HVDiameterGradient  in range(-1*HVDiaR,(1*HVDiaR)+1,1):

                            calculatedPoint =CalculateFinalizedPrice(LVTurns+(TurnsGradient*TurnsStepMinimum),
                                                                     LVHeight+(LVHeightGradient*FoilHeightStepMinimum),
                                                                     LVThickness+(LVThicknessGradient*FoilThicknessStepMinimum),
                                                                     HVWireDiameter+(HVDiameterGradient*HVDiameterStepMinimum),
                                                                     CoreDiameter+(CoreDiaGradient*CoreDiameterStepMinimum),
                                                                     CoreLength+(CoreLengthGradient*CoreLengthStepMinimum)
                                                                    )
                            difference = calculatedPoint-InitialPoint
                            if(difference<=gradients["Slope"]):
                                gradients["Slope"]=difference
                                gradients["CoreDiameter"] = CoreDiaGradient
                                gradients["CoreLength"] = CoreLengthGradient
                                gradients["HVDiameter"]=HVDiameterGradient
                                gradients["LVHeight"]=LVHeightGradient
                                gradients["LVThickness"] = LVThicknessGradient
                                gradients["LVTurns"] = TurnsGradient
                                        


    gradients["Slope"] = math.pow(round(abs(gradients["Slope"])),ErrorPow)
    return gradients

def CalculateStochasticGradientFast(LVTurns, LVThickness, LVHeight, CoreDiameter,CoreLength, HVWireDiameter, TurnsStepMinimum = 1, FoilHeightStepMinimum = 5, FoilThicknessStepMinimum = 0.05, CoreDiameterStepMinimum =1, CoreLengthStepMinimum =1, HVDiameterStepMinimum = 0.05, sampleSize=3, ErrorPow = 1,  printValues = False,obround=True):


    indexes = [0,1,2,3,4,5]
    samples = random.sample(indexes,sampleSize)
    CoreDiameterR = 1 if 0 in samples else 0
    CoreLengthR = 1 if 1 in samples else 0
    LVTurnsR = 1 if 2 in samples else 0
    LVHeightR = 1 if 3 in samples else 0
    LVThicknessR = 1 if 4 in samples else 0
    HVDiaR = 1 if 5 in samples else 0
    ilist =[]    


    if CoreDiameterR == 1:
        ilist.append("CD")
    if CoreLengthR == 1:
        ilist.append("CL")
    if LVTurnsR == 1:
        ilist.append("LVT")
    if LVThicknessR == 1:
        ilist.append("LVTH")
    if LVHeightR == 1:
        ilist.append("LVH")
    if HVDiaR == 1:
        ilist.append("HVD")
    
    paramList = [LVTurns, LVHeight, LVThickness, HVWireDiameter, CoreDiameter, CoreLength]
    paramDict = {
        0: "LVTurns",
        1: "LVHeight",
        2: "LVThickness",
        3: "HVDiameter",
        4: "CoreDiameter",
        5: "CoreLength"
    }



    InitialPoint = CalculateFinalizedPrice(LVTurns,LVHeight,LVThickness,HVWireDiameter,CoreDiameter,CoreLength)
    bestVal = InitialPoint
    gradients = {
        "Slope":0,
        "CoreDiameter":0,
        "CoreLength":0,
        "LVTurns":0,
        "LVHeight":0,
        "LVThickness":0,
        "HVDiameter":0
    }
    i0 = 0
    i1 = 0
    i2=  0
    i3 = 0
    i4= 0
    i5 = 0

    i0ms = 0
    i1ms = 0
    i2ms = 0
    i3ms = 0
    i4ms = 0
    i5ms = 0

    i0p = 0
    i1p = 0
    i2p = 0
    i3p = 0
    i4p = 0
    i5p = 0

    if len(ilist)==1:
        val = ilist[0]
        if val == "CD":
            i0 =CoreDiameter
            i0ms = CoreDiameterStepMinimum
            i0p= 4
        elif val == "CL":
            i0 = CoreLength
            i0ms = CoreLengthStepMinimum
            i0p= 5
        elif val == "LVT":
            i0 = LVTurns
            i0ms = TurnsStepMinimum
            i0p= 0
        elif val == "LVTH":
            i0 = LVThickness
            i0ms = FoilThicknessStepMinimum
            i0p = 2
        elif val == "LVH":
            i0 = LVHeight
            i0ms = FoilHeightStepMinimum
            i0p = 1
        elif val == "HVD":
            i0=HVWireDiameter
            i0ms = HVDiameterStepMinimum
            i0p = 3
        
        for iGradient in range(-1,2,1):
            paramList[i0p] = i0 + i0ms*iGradient
            calculatedPoint = CalculateFinalizedPrice(paramList[0],paramList[1],paramList[2],paramList[3],paramList[4],paramList[5])
            difference = calculatedPoint-InitialPoint
            if(difference<=gradients["Slope"]):
                                gradients["Slope"]=difference
                                gradients[paramDict[i0p]] = iGradient
    
    if len(ilist)==2:
        val = ilist[0]
        if val == "CD":
            i0 =CoreDiameter
            i0ms = CoreDiameterStepMinimum
            i0p= 4
        elif val == "CL":
            i0 = CoreLength
            i0ms = CoreLengthStepMinimum
            i0p= 5
        elif val == "LVT":
            i0 = LVTurns
            i0ms = TurnsStepMinimum
            i0p= 0
        elif val == "LVTH":
            i0 = LVThickness
            i0ms = FoilThicknessStepMinimum
            i0p = 2
        elif val == "LVH":
            i0 = LVHeight
            i0ms = FoilHeightStepMinimum
            i0p = 1
        elif val == "HVD":
            i0=HVWireDiameter
            i0ms = HVDiameterStepMinimum
            i0p = 3
        
        val = ilist[1]
        if val == "CD":
            i1 =CoreDiameter
            i1ms = CoreDiameterStepMinimum
            i1p= 4
        elif val == "CL":
            i1 = CoreLength
            i1ms = CoreLengthStepMinimum
            i1p= 5
        elif val == "LVT":
            i1 = LVTurns
            i1ms = TurnsStepMinimum
            i1p= 0
        elif val == "LVTH":
            i1 = LVThickness
            i1ms = FoilThicknessStepMinimum
            i1p = 2
        elif val == "LVH":
            i1 = LVHeight
            i1ms = FoilHeightStepMinimum
            i1p = 1
        elif val == "HVD":
            i1=HVWireDiameter
            i1ms = HVDiameterStepMinimum
            i1p = 3
        
        for jGradient in range(-1,2,1):
            for iGradient in range(-1,2,1):
                paramList[i1p] = i1 + i1ms*jGradient
                paramList[i0p] = i0 + i0ms*iGradient
                calculatedPoint = CalculateFinalizedPrice(paramList[0],paramList[1],paramList[2],paramList[3],paramList[4],paramList[5])
                difference = calculatedPoint-InitialPoint
                if(difference<=gradients["Slope"]):
                                    gradients["Slope"]=difference
                                    gradients[paramDict[i0p]] = iGradient
                                    gradients[paramDict[i1p]] = jGradient
    
    if len(ilist)==3:
        val = ilist[0]
        if val == "CD":
            i0 =CoreDiameter
            i0ms = CoreDiameterStepMinimum
            i0p= 4
        elif val == "CL":
            i0 = CoreLength
            i0ms = CoreLengthStepMinimum
            i0p= 5
        elif val == "LVT":
            i0 = LVTurns
            i0ms = TurnsStepMinimum
            i0p= 0
        elif val == "LVTH":
            i0 = LVThickness
            i0ms = FoilThicknessStepMinimum
            i0p = 2
        elif val == "LVH":
            i0 = LVHeight
            i0ms = FoilHeightStepMinimum
            i0p = 1
        elif val == "HVD":
            i0=HVWireDiameter
            i0ms = HVDiameterStepMinimum
            i0p = 3
        
        val = ilist[1]
        if val == "CD":
            i1 =CoreDiameter
            i1ms = CoreDiameterStepMinimum
            i1p= 4
        elif val == "CL":
            i1 = CoreLength
            i1ms = CoreLengthStepMinimum
            i1p= 5
        elif val == "LVT":
            i1 = LVTurns
            i1ms = TurnsStepMinimum
            i1p= 0
        elif val == "LVTH":
            i1 = LVThickness
            i1ms = FoilThicknessStepMinimum
            i1p = 2
        elif val == "LVH":
            i1 = LVHeight
            i1ms = FoilHeightStepMinimum
            i1p = 1
        elif val == "HVD":
            i1=HVWireDiameter
            i1ms = HVDiameterStepMinimum
            i1p = 3
        
        val = ilist[2]
        if val == "CD":
            i2 =CoreDiameter
            i2ms = CoreDiameterStepMinimum
            i2p= 4
        elif val == "CL":
            i2 = CoreLength
            i2ms = CoreLengthStepMinimum
            i2p= 5
        elif val == "LVT":
            i2 = LVTurns
            i2ms = TurnsStepMinimum
            i2p= 0
        elif val == "LVTH":
            i2 = LVThickness
            i2ms = FoilThicknessStepMinimum
            i2p = 2
        elif val == "LVH":
            i2 = LVHeight
            i2ms = FoilHeightStepMinimum
            i2p = 1
        elif val == "HVD":
            i2=HVWireDiameter
            i2ms = HVDiameterStepMinimum
            i2p = 3

        for kGradient in range(-1,2,1):
            for jGradient in range(-1,2,1):
                for iGradient in range(-1,2,1):
                    paramList[i2p] = i2 + i2ms*kGradient
                    paramList[i1p] = i1 + i1ms*jGradient
                    paramList[i0p] = i0 + i0ms*iGradient
                    calculatedPoint = CalculateFinalizedPrice(paramList[0],paramList[1],paramList[2],paramList[3],paramList[4],paramList[5])
                    difference = calculatedPoint-InitialPoint
                    if(difference<=gradients["Slope"]):
                                        gradients["Slope"]=difference
                                        gradients[paramDict[i0p]] = iGradient
                                        gradients[paramDict[i1p]] = jGradient
                                        gradients[paramDict[i2p]] = kGradient
    
    if len(ilist)==4:
        val = ilist[0]
        if val == "CD":
            i0 =CoreDiameter
            i0ms = CoreDiameterStepMinimum
            i0p= 4
        elif val == "CL":
            i0 = CoreLength
            i0ms = CoreLengthStepMinimum
            i0p= 5
        elif val == "LVT":
            i0 = LVTurns
            i0ms = TurnsStepMinimum
            i0p= 0
        elif val == "LVTH":
            i0 = LVThickness
            i0ms = FoilThicknessStepMinimum
            i0p = 2
        elif val == "LVH":
            i0 = LVHeight
            i0ms = FoilHeightStepMinimum
            i0p = 1
        elif val == "HVD":
            i0=HVWireDiameter
            i0ms = HVDiameterStepMinimum
            i0p = 3
        
        val = ilist[1]
        if val == "CD":
            i1 =CoreDiameter
            i1ms = CoreDiameterStepMinimum
            i1p= 4
        elif val == "CL":
            i1 = CoreLength
            i1ms = CoreLengthStepMinimum
            i1p= 5
        elif val == "LVT":
            i1 = LVTurns
            i1ms = TurnsStepMinimum
            i1p= 0
        elif val == "LVTH":
            i1 = LVThickness
            i1ms = FoilThicknessStepMinimum
            i1p = 2
        elif val == "LVH":
            i1 = LVHeight
            i1ms = FoilHeightStepMinimum
            i1p = 1
        elif val == "HVD":
            i1=HVWireDiameter
            i1ms = HVDiameterStepMinimum
            i1p = 3
        
        val = ilist[2]
        if val == "CD":
            i2 =CoreDiameter
            i2ms = CoreDiameterStepMinimum
            i2p= 4
        elif val == "CL":
            i2 = CoreLength
            i2ms = CoreLengthStepMinimum
            i2p= 5
        elif val == "LVT":
            i2 = LVTurns
            i2ms = TurnsStepMinimum
            i2p= 0
        elif val == "LVTH":
            i2 = LVThickness
            i2ms = FoilThicknessStepMinimum
            i2p = 2
        elif val == "LVH":
            i2 = LVHeight
            i2ms = FoilHeightStepMinimum
            i2p = 1
        elif val == "HVD":
            i2=HVWireDiameter
            i2ms = HVDiameterStepMinimum
            i2p = 3

        val = ilist[3]
        if val == "CD":
            i3 =CoreDiameter
            i3ms = CoreDiameterStepMinimum
            i3p= 4
        elif val == "CL":
            i3 = CoreLength
            i3ms = CoreLengthStepMinimum
            i3p= 5
        elif val == "LVT":
            i3 = LVTurns
            i3ms = TurnsStepMinimum
            i3p= 0
        elif val == "LVTH":
            i3 = LVThickness
            i3ms = FoilThicknessStepMinimum
            i3p = 2
        elif val == "LVH":
            i3 = LVHeight
            i3ms = FoilHeightStepMinimum
            i3p = 1
        elif val == "HVD":
            i3=HVWireDiameter
            i3ms = HVDiameterStepMinimum
            i3p = 3

        for lGradient in range(-1,2,1):
            for kGradient in range(-1,2,1):
                for jGradient in range(-1,2,1):
                    for iGradient in range(-1,2,1):
                        paramList[i3p] = i3 + i3ms*lGradient
                        paramList[i2p] = i2 + i2ms*kGradient
                        paramList[i1p] = i1 + i1ms*jGradient
                        paramList[i0p] = i0 + i0ms*iGradient
                        calculatedPoint = CalculateFinalizedPrice(paramList[0],paramList[1],paramList[2],paramList[3],paramList[4],paramList[5])
                        difference = calculatedPoint-InitialPoint
                        if(difference<=gradients["Slope"]):
                                            gradients["Slope"]=difference
                                            gradients[paramDict[i0p]] = iGradient
                                            gradients[paramDict[i1p]] = jGradient
                                            gradients[paramDict[i2p]] = kGradient
                                            gradients[paramDict[i3p]] = lGradient
    
    if len(ilist)==5:
        val = ilist[0]
        if val == "CD":
            i0 =CoreDiameter
            i0ms = CoreDiameterStepMinimum
            i0p= 4
        elif val == "CL":
            i0 = CoreLength
            i0ms = CoreLengthStepMinimum
            i0p= 5
        elif val == "LVT":
            i0 = LVTurns
            i0ms = TurnsStepMinimum
            i0p= 0
        elif val == "LVTH":
            i0 = LVThickness
            i0ms = FoilThicknessStepMinimum
            i0p = 2
        elif val == "LVH":
            i0 = LVHeight
            i0ms = FoilHeightStepMinimum
            i0p = 1
        elif val == "HVD":
            i0=HVWireDiameter
            i0ms = HVDiameterStepMinimum
            i0p = 3
        
        val = ilist[1]
        if val == "CD":
            i1 =CoreDiameter
            i1ms = CoreDiameterStepMinimum
            i1p= 4
        elif val == "CL":
            i1 = CoreLength
            i1ms = CoreLengthStepMinimum
            i1p= 5
        elif val == "LVT":
            i1 = LVTurns
            i1ms = TurnsStepMinimum
            i1p= 0
        elif val == "LVTH":
            i1 = LVThickness
            i1ms = FoilThicknessStepMinimum
            i1p = 2
        elif val == "LVH":
            i1 = LVHeight
            i1ms = FoilHeightStepMinimum
            i1p = 1
        elif val == "HVD":
            i1=HVWireDiameter
            i1ms = HVDiameterStepMinimum
            i1p = 3
        
        val = ilist[2]
        if val == "CD":
            i2 =CoreDiameter
            i2ms = CoreDiameterStepMinimum
            i2p= 4
        elif val == "CL":
            i2 = CoreLength
            i2ms = CoreLengthStepMinimum
            i2p= 5
        elif val == "LVT":
            i2 = LVTurns
            i2ms = TurnsStepMinimum
            i2p= 0
        elif val == "LVTH":
            i2 = LVThickness
            i2ms = FoilThicknessStepMinimum
            i2p = 2
        elif val == "LVH":
            i2 = LVHeight
            i2ms = FoilHeightStepMinimum
            i2p = 1
        elif val == "HVD":
            i2=HVWireDiameter
            i2ms = HVDiameterStepMinimum
            i2p = 3

        val = ilist[3]
        if val == "CD":
            i3 =CoreDiameter
            i3ms = CoreDiameterStepMinimum
            i3p= 4
        elif val == "CL":
            i3 = CoreLength
            i3ms = CoreLengthStepMinimum
            i3p= 5
        elif val == "LVT":
            i3 = LVTurns
            i3ms = TurnsStepMinimum
            i3p= 0
        elif val == "LVTH":
            i3 = LVThickness
            i3ms = FoilThicknessStepMinimum
            i3p = 2
        elif val == "LVH":
            i3 = LVHeight
            i3ms = FoilHeightStepMinimum
            i3p = 1
        elif val == "HVD":
            i3=HVWireDiameter
            i3ms = HVDiameterStepMinimum
            i3p = 3
        
        val = ilist[4]
        if val == "CD":
            i4 =CoreDiameter
            i4ms = CoreDiameterStepMinimum
            i4p= 4
        elif val == "CL":
            i4 = CoreLength
            i4ms = CoreLengthStepMinimum
            i4p= 5
        elif val == "LVT":
            i4 = LVTurns
            i4ms = TurnsStepMinimum
            i4p= 0
        elif val == "LVTH":
            i4 = LVThickness
            i4ms = FoilThicknessStepMinimum
            i4p = 2
        elif val == "LVH":
            i4 = LVHeight
            i4ms = FoilHeightStepMinimum
            i4p = 1
        elif val == "HVD":
            i4=HVWireDiameter
            i4ms = HVDiameterStepMinimum
            i4p = 3


        for mGradient in range(-1,2,1):
            for lGradient in range(-1,2,1):
                for kGradient in range(-1,2,1):
                    for jGradient in range(-1,2,1):
                        for iGradient in range(-1,2,1):
                            paramList[i4p] = i4 + i4ms*mGradient
                            paramList[i3p] = i3 + i3ms*lGradient
                            paramList[i2p] = i2 + i2ms*kGradient
                            paramList[i1p] = i1 + i1ms*jGradient
                            paramList[i0p] = i0 + i0ms*iGradient
                            calculatedPoint = CalculateFinalizedPrice(paramList[0],paramList[1],paramList[2],paramList[3],paramList[4],paramList[5])
                            difference = calculatedPoint-InitialPoint
                            if(difference<=gradients["Slope"]):
                                                gradients["Slope"]=difference
                                                gradients[paramDict[i0p]] = iGradient
                                                gradients[paramDict[i1p]] = jGradient
                                                gradients[paramDict[i2p]] = kGradient
                                                gradients[paramDict[i3p]] = lGradient
    gradients["Slope"] = math.pow(round(abs(gradients["Slope"])),ErrorPow)
    
    return gradients
            



def CalculateGradientDescent(alpha = 0.2, MaxIter = 10000000,InitLVTurns=53, InitLVThickness=2.2, InitLVHeight=700, InitCoreDiameter=280,InitCoreLength=140, InitHVWireDiameter=3, TurnsStepMinimum = 1, FoilHeightStepMinimum = 5, FoilThicknessStepMinimum = 0.05, CoreDiameterStepMinimum =1, CoreLengthStepMinimum =1, HVDiameterStepMinimum = 0.05,  printValues = False,obround=True, Stochastic = True,Power =1, Samples =3):
    gradients = {
        "Slope":0,
        "CoreDiameter":0,
        "CoreLength":0,
        "LVTurns":0,
        "LVHeight":0,
        "LVThickness":0,
        "HVDiameter":0
    }

    Value = 0
    LvTurnsI= InitLVTurns
    LvHeightI = InitLVHeight
    LvThicknessI=InitLVThickness
    CoreDiameterI=InitCoreDiameter
    CoreLengthI=InitCoreLength
    HvDiameterI=InitHVWireDiameter
    Value = CalculateFinalizedPrice(LvTurnsI, LvHeightI, LvThicknessI, HvDiameterI, CoreDiameterI, CoreLengthI)
    transformerFound = {

        "price" : Value,
        "corediameter" : CoreDiameterI,
        "hvdiameter" : HvDiameterI,
        "lvfoilheight" : LvHeightI,
        "lvfoilturns" : LvTurnsI,
        "lvfoilthickness" : LvThicknessI,
        "corelength":CoreLengthI
    }
    startTime = time.time()
    printedGradTime = False
    printedCheckValuesTime = False

    for i in range(0,MaxIter):
        if(printValues):
            if((i%100000) ==0):
                print("Program Has Made " + str(i/1000000)+"/"+str(MaxIter/1000000)+ "M Iterations")
        gradStart=time.time()
        if(Stochastic):
            if(Samples <=5):
                gradients = CalculateStochasticGradientFast(LvTurnsI,LvThicknessI,LvHeightI,CoreDiameterI,CoreLengthI,HvDiameterI,sampleSize=Samples,ErrorPow=Power)

            else:
                gradients = CalculateStochasticGradient(LvTurnsI,LvThicknessI,LvHeightI,CoreDiameterI,CoreLengthI,HvDiameterI,sampleSize=Samples,ErrorPow=Power)
        else:
            gradients = CalculateGradient(LvTurnsI,LvThicknessI,LvHeightI,CoreDiameterI,CoreLengthI,HvDiameterI)
        gradEnd = time.time()
        if( not printedGradTime and printValues):
            print("Calculating gradients took : " + str(gradEnd-gradStart) + " seconds")
            printedGradTime = True
        
        chechValuesStart = time.time()
        LvTurnsI += gradients["LVTurns"] * TurnsStepMinimum * alpha * gradients["Slope"]
        LvHeightI+= gradients["LVHeight"] * FoilHeightStepMinimum * alpha * gradients["Slope"]
        LvThicknessI+= gradients["LVThickness"] * FoilThicknessStepMinimum * alpha * gradients["Slope"]
        CoreDiameterI += gradients["CoreDiameter"] * CoreDiameterStepMinimum * alpha * gradients["Slope"]
        CoreLengthI += gradients["CoreLength"] * CoreLengthStepMinimum * alpha * gradients["Slope"]
        HvDiameterI += gradients["HVDiameter"] * HVDiameterStepMinimum * alpha * gradients["Slope"]

        LvTurnsI = Clamp(LvTurnsI,FOILTURNS_MINIMUM,FOILTURNS_MAXIMUM)
        LvHeightI = Clamp(LvHeightI,FOILHEIGHT_MINIMUM,FOILHEIGHT_MAXIMUM)
        LvThicknessI = Clamp(LvThicknessI, FOILTHICKNESS_MINIMUM,FOILTHICKNESS_MAXIMUM)
        CoreDiameterI = Clamp(CoreDiameterI, CORE_MINIMUM,CORE_MAXIMUM)
        CoreLengthI = Clamp(CoreLengthI, CORELENGTH_MINIMUM,CoreDiameterI)
        HvDiameterI = Clamp(HvDiameterI,HVDIA_MINIMUM,HVDIA_MAXIMUM)

        ValueIter = CalculateFinalizedPrice(LvTurnsI, LvHeightI, LvThicknessI, HvDiameterI, CoreDiameterI, CoreLengthI)
        if ValueIter<=Value:
                Value = ValueIter
                transformerFound = {
                "price" : Value,
                "corediameter" : CoreDiameterI,
                "hvdiameter" : HvDiameterI,
                "lvfoilheight" : LvHeightI,
                "lvfoilturns" : LvTurnsI,
                "lvfoilthickness" : LvThicknessI,
                "corelength":CoreLengthI
            }
        checkValuesEnd = time.time()
        if(not printedCheckValuesTime and printValues):
            print("Checking and comparing values took : " + str(checkValuesEnd-chechValuesStart) + " seconds")
            printedCheckValuesTime = True
    if(printValues):
        endTime = time.time()
        PrintTransformer(transformerFound)
        print("PROCESS HAS TOOK " + str(endTime-startTime) + " SECONDS")
    return transformerFound


def PrintTransformer(transformerFound):
        CalculateFinalizedPrice(transformerFound["lvfoilturns"], transformerFound["lvfoilheight"], transformerFound["lvfoilthickness"], transformerFound["hvdiameter"], transformerFound["corediameter"],transformerFound["corelength"],isFinal=True)
        print("####################################################")
        print("Core Diameter = " + str(transformerFound["corediameter"]))
        print("Core Length = " + str(transformerFound["corelength"]))  
        print("HV Diameter = " + str(transformerFound["hvdiameter"]))
        print("Foil Height = " + str(transformerFound["lvfoilheight"]))
        print("Foil Turns = " + str(transformerFound["lvfoilturns"]))
        print("Foil Thickness = " + str(transformerFound["lvfoilthickness"]))
        print("THE PRICE IS =" + str(transformerFound["price"]))
        print("####################################################")
    
def PlotCalculateCore(x,y):
    return CalculateFinalizedPrice(y,700,2.2,2.2,x,140)

if __name__ == '__main__':
    BucketFillingSmart(printValues=True, BrakeDistance=5, tolerance=25, obround=True)
    #SearchWindow()
    #CalculateGradientDescent(printValues=True,alpha=0.001,MaxIter=12000000,Stochastic=True,Samples=1,Power=1)
    #BucketFillingFast(printValues=True, BrakeDistance=5, tolerance=0, obround=True)
    #DoEverythingThread(printValues=True)
    #startTime = time.time()
    #DoEverythingThread(TurnsStep=1, HeightStep=5, CoreDiaStep=1, ThicknessStep=0.05, HVWireDiaStep=0.05, printValues = True)
    #CalculateFinalizedPrice(56,815,0.5,3.25,130,isFinal=True)
    #CalculateFinalizedPriceIntolerant(47,500,0.4,1.35,100,83,isFinal=True)
    #endTime = time.time()




#CalculateFinalizedPrice(56,815,0.5,3.25,198,isFinal=True)




#startTime = time.time()
#DoEverythingDumb(returns=None, TurnsStep = 1, ThicknessStep = 0.05, HeightStep = 5, CoreDiaStep = 1, HVWireDiaStep = 0.05,
# CoreDiaStart = 175, CoreDiaEnd= 215,
# LvTurnsStart = 50, LvTurnsEnd= 60,
# LVThicknessStart =0.4, LvThicknessEnd =0.7,
# LVHeightStart = 750, LvHeightEnd = 900,
# HvDiaStart =3, HvDiaEnd=3.6,
# printValues = True)
#finishTime = time.time()
#print("Time consumed is:" + str(finishTime-startTime)

