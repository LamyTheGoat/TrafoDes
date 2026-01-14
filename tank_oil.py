"""
Tank, Finwall, and Oil Calculations Module for Transformer Design

This module provides post-optimization calculations for:
- Tank dimensions based on active part + clearances
- Corrugated finwall (auto-optimized or manual)
- Oil volume and pricing

These calculations are performed AFTER optimization completes,
using the optimized transformer dimensions as input.
"""

import math

# =============================================================================
# MATERIAL CONSTANTS
# =============================================================================

# Steel properties
STEEL_DENSITY = 7.85  # g/cm3 = 7850 kg/m3
STEEL_PRICE_DEFAULT = 2.50  # $/kg

# Tank default clearances (mm)
DEFAULT_CLEARANCE_X = 40.0  # Side clearance (width direction)
DEFAULT_CLEARANCE_Y = 40.0  # Front/back clearance (depth direction)
DEFAULT_CLEARANCE_Z = 40.0  # Top/bottom clearance (height direction)

# Tank wall thickness (mm)
DEFAULT_TANK_WALL_THICKNESS = 3.0  # mm

# Finwall specifications
FINWALL_PITCH = 45  # mm (center-to-center distance between fins)
FINWALL_FIN_DEPTH_MIN = 40   # mm
FINWALL_FIN_DEPTH_MAX = 400  # mm
FINWALL_THICKNESS_OPTIONS = [0.8, 1.0, 1.2, 1.5]  # mm

# Heat dissipation coefficients (W/m2/K)
HEAT_DISSIPATION_PLAIN = 12.5
FINWALL_EFFICIENCY_FACTOR = 2.0  # Corrugated ~2x vs plain surface

# Default target temperature rise for ONAN transformers (K)
DEFAULT_TARGET_TEMP_RISE = 65.0

# Oil properties (researched defaults for 2024-2025)
OIL_TYPES = {
    'mineral': {
        'density': 0.87,      # kg/L
        'price': 1.90,        # $/L
        'name': 'Mineral Oil'
    },
    'natural_ester': {
        'density': 0.92,      # kg/L
        'price': 5.00,        # $/L
        'name': 'Natural Ester (FR3)'
    },
    'silicone': {
        'density': 0.96,      # kg/L
        'price': 7.00,        # $/L
        'name': 'Silicone Oil'
    }
}

DEFAULT_OIL_TYPE = 'mineral'
DEFAULT_OIL_FILL_LEVEL = 85.0  # % (allows for thermal expansion)


# =============================================================================
# ACTIVE PART DIMENSION CALCULATIONS
# =============================================================================

def calculate_radial_thickness_lv(lv_turns, lv_thickness, n_ducts_lv,
                                   lv_insulation=0.125, duct_thickness=4.0):
    """
    Calculate LV winding radial thickness.

    Based on mainRect.py CalculateRadialThicknessLV logic.
    """
    radial = lv_turns * lv_thickness
    radial += (lv_turns - 1) * lv_insulation
    radial += n_ducts_lv * (duct_thickness + 0.5)
    return radial


def calculate_radial_thickness_hv(lv_height, lv_turns, hv_thickness, hv_length,
                                   n_ducts_hv, hv_rate=20000, lv_rate=400,
                                   hv_insulation=0.5, duct_thickness=4.0,
                                   circular_hv=False):
    """
    Calculate HV winding radial thickness.

    Based on mainRect.py CalculateRadialThicknessHV logic.
    """
    # Calculate HV turns based on voltage ratio
    hv_connection_factor = 1.0  # Delta
    lv_connection_factor = 1 / math.sqrt(3)  # Star
    hv_turns = lv_turns * (hv_rate * hv_connection_factor) / (lv_rate * lv_connection_factor)

    # Layer height
    hv_layer_height = lv_height - 50
    if hv_layer_height <= 0:
        hv_layer_height = lv_height * 0.9

    # Wire axial dimension
    wire_axial = hv_thickness if circular_hv else hv_length
    insulation_wire = 0.12

    # Turns per layer
    turns_per_layer = (hv_layer_height / (wire_axial + insulation_wire)) - 1
    if turns_per_layer <= 0:
        turns_per_layer = 1

    # Number of layers
    n_layers = math.ceil(hv_turns / turns_per_layer)

    # Radial thickness
    radial = n_layers * hv_thickness
    radial += (n_layers - 1) * hv_insulation
    radial += n_ducts_hv * (duct_thickness + 0.5)

    return radial


def calculate_active_part_dimensions(core_diameter, core_length, lv_turns,
                                      lv_height, lv_thickness, hv_thickness,
                                      hv_length, n_ducts_lv=0, n_ducts_hv=0,
                                      circular_hv=False,
                                      phase_gap=12.0, distance_core_lv=2.0,
                                      main_gap=12.0):
    """
    Calculate overall active part envelope dimensions.

    Args:
        core_diameter: Core leg diameter (mm)
        core_length: Core stack depth/length (mm)
        lv_turns: Number of LV turns
        lv_height: LV foil height (mm)
        lv_thickness: LV foil thickness (mm)
        hv_thickness: HV wire thickness (mm)
        hv_length: HV wire length for rectangular (mm)
        n_ducts_lv: Number of LV cooling ducts
        n_ducts_hv: Number of HV cooling ducts
        circular_hv: Whether HV wire is circular
        phase_gap: Gap between phases (mm)
        distance_core_lv: Core to LV distance (mm)
        main_gap: LV to HV gap (mm)

    Returns:
        dict: {'width': mm, 'depth': mm, 'height': mm}
    """
    # Calculate radial thicknesses
    lv_radial = calculate_radial_thickness_lv(lv_turns, lv_thickness, n_ducts_lv)
    hv_radial = calculate_radial_thickness_hv(lv_height, lv_turns, hv_thickness,
                                               hv_length, n_ducts_hv,
                                               circular_hv=circular_hv)

    # Total radial build-up per phase (one side)
    total_radial = lv_radial + hv_radial + main_gap + distance_core_lv

    # Center between adjacent legs
    center_between_legs = (core_diameter + total_radial * 2) + phase_gap

    # Window height (from mainRect.py)
    window_height = lv_height + 40

    # Yoke height (typically equals core diameter)
    yoke_height = core_diameter

    # ACTIVE PART DIMENSIONS
    # Width: 3 legs arrangement (2 * leg spacing + core diameter)
    width = 2 * center_between_legs + core_diameter

    # Depth: core diameter + obround extension (core_length)
    depth = core_diameter + core_length

    # Height: window + 2 yokes
    height = window_height + 2 * yoke_height

    return {
        'width': width,
        'depth': depth,
        'height': height,
        'window_height': window_height,
        'center_between_legs': center_between_legs,
        'total_radial': total_radial
    }


# =============================================================================
# TANK CALCULATIONS
# =============================================================================

def calculate_tank_dimensions(active_width, active_depth, active_height,
                               clearance_x=DEFAULT_CLEARANCE_X,
                               clearance_y=DEFAULT_CLEARANCE_Y,
                               clearance_z=DEFAULT_CLEARANCE_Z):
    """
    Calculate internal tank dimensions based on active part + clearances.

    Args:
        active_width, active_depth, active_height: Active part dimensions (mm)
        clearance_x: Side clearance (mm)
        clearance_y: Front/back clearance (mm)
        clearance_z: Top/bottom clearance (mm)

    Returns:
        dict: {'width': mm, 'depth': mm, 'height': mm}
    """
    return {
        'width': active_width + 2 * clearance_x,
        'depth': active_depth + 2 * clearance_y,
        'height': active_height + 2 * clearance_z
    }


def calculate_tank_shell_weight(tank_width, tank_depth, tank_height,
                                  wall_thickness=DEFAULT_TANK_WALL_THICKNESS,
                                  steel_density=STEEL_DENSITY):
    """
    Calculate tank shell weight (rectangular box without finwalls).

    Includes: 4 side walls + top cover + bottom plate

    Args:
        tank_width, tank_depth, tank_height: Internal dimensions (mm)
        wall_thickness: Wall thickness (mm)
        steel_density: Steel density (g/cm3)

    Returns:
        float: Tank shell weight in kg
    """
    # Convert mm to cm for volume calculation
    t = wall_thickness / 10.0
    w = tank_width / 10.0
    d = tank_depth / 10.0
    h = tank_height / 10.0

    # Surface areas (cm2) - 6 faces of rectangular box
    front_back_area = 2 * (w * h)
    left_right_area = 2 * (d * h)
    top_bottom_area = 2 * (w * d)

    total_area = front_back_area + left_right_area + top_bottom_area

    # Volume = area * thickness (cm3)
    volume_cm3 = total_area * t

    # Weight = volume * density (g) -> convert to kg
    weight_kg = (volume_cm3 * steel_density) / 1000.0

    return weight_kg


# =============================================================================
# FINWALL CALCULATIONS
# =============================================================================

def calculate_finwall_area_multiplier(fin_depth, pitch=FINWALL_PITCH):
    """
    Calculate effective surface area multiplier for corrugated finwall.

    Approximation: Semi-circular corrugation profile
    Area_multiplier = (2 * pi * radius) / pitch

    Args:
        fin_depth: Depth of fin from wall surface (mm)
        pitch: Center-to-center distance between fins (mm)

    Returns:
        float: Surface area multiplier (typically 1.5-3.0)
    """
    radius = fin_depth / 2.0
    arc_length = math.pi * radius

    # One pitch contains two half-circles
    effective_length = 2 * arc_length

    multiplier = effective_length / pitch

    # Clamp to reasonable range
    return max(1.0, min(4.0, multiplier))


def calculate_optimal_fin_depth(total_losses, target_temp_rise,
                                 tank_width, tank_depth, tank_height,
                                 finwall_surfaces, pitch=FINWALL_PITCH):
    """
    Auto-calculate optimal fin depth to achieve target temperature rise.

    Formula: Required_Area = Total_Losses / (Heat_Coeff * Temp_Rise)

    Args:
        total_losses: Total transformer losses (W) - NLL + LL
        target_temp_rise: Target temperature rise (K), typically 65K
        tank_width, tank_depth, tank_height: Tank dimensions (mm)
        finwall_surfaces: List of surfaces with finwalls ['front', 'back', 'left', 'right']
        pitch: Fin pitch (mm)

    Returns:
        float: Optimal fin depth in mm (clamped to 40-400mm range)
    """
    # Required dissipation area (m2)
    required_area_m2 = total_losses / (HEAT_DISSIPATION_PLAIN * target_temp_rise)

    # Calculate plain tank surface area for selected sides (m2)
    plain_area_m2 = 0.0
    for surface in finwall_surfaces:
        if surface in ['front', 'back']:
            area_mm2 = tank_width * tank_height
        elif surface in ['left', 'right']:
            area_mm2 = tank_depth * tank_height
        else:
            continue
        plain_area_m2 += area_mm2 / 1_000_000.0

    if plain_area_m2 <= 0:
        return FINWALL_FIN_DEPTH_MIN

    # Enhancement ratio needed
    enhancement_ratio = required_area_m2 / plain_area_m2

    # If plain tank is sufficient, use minimum fin depth
    if enhancement_ratio <= 1.0:
        return FINWALL_FIN_DEPTH_MIN

    # Calculate fin depth that achieves this enhancement
    # From area_multiplier formula: multiplier = pi * fin_depth / pitch
    # Solving: fin_depth = multiplier * pitch / pi
    optimal_depth = enhancement_ratio * pitch / math.pi

    # Clamp to valid range
    return max(FINWALL_FIN_DEPTH_MIN, min(FINWALL_FIN_DEPTH_MAX, optimal_depth))


def calculate_finwall_panel_weight(panel_width, panel_height, fin_depth,
                                    sheet_thickness, pitch=FINWALL_PITCH,
                                    steel_density=STEEL_DENSITY):
    """
    Calculate weight of a single finwall panel.

    Args:
        panel_width: Panel width (mm)
        panel_height: Panel height (mm)
        fin_depth: Fin depth (mm)
        sheet_thickness: Steel sheet thickness (mm)
        pitch: Fin pitch (mm)
        steel_density: Steel density (g/cm3)

    Returns:
        float: Panel weight in kg
    """
    # Number of corrugations
    n_fins = panel_width / pitch

    # Developed (unfolded) length per pitch
    radius = fin_depth / 2.0
    arc_length = math.pi * radius
    developed_length = 2 * arc_length

    # Total developed width
    total_developed_width = n_fins * developed_length

    # Area in cm2
    area_cm2 = (total_developed_width / 10.0) * (panel_height / 10.0)

    # Volume and weight
    volume_cm3 = area_cm2 * (sheet_thickness / 10.0)
    weight_kg = (volume_cm3 * steel_density) / 1000.0

    return weight_kg


def calculate_total_finwall_weight(tank_width, tank_depth, tank_height,
                                    fin_depth, sheet_thickness,
                                    finwall_surfaces,
                                    pitch=FINWALL_PITCH,
                                    steel_density=STEEL_DENSITY):
    """
    Calculate total finwall weight for all selected surfaces.

    Args:
        tank_width, tank_depth, tank_height: Tank dimensions (mm)
        fin_depth: Fin depth (mm)
        sheet_thickness: Steel thickness (mm)
        finwall_surfaces: List of surfaces ['front', 'back', 'left', 'right']
        pitch: Fin pitch (mm)
        steel_density: Steel density (g/cm3)

    Returns:
        float: Total finwall weight in kg
    """
    total_weight = 0.0

    for surface in finwall_surfaces:
        if surface in ['front', 'back']:
            panel_width = tank_width
            panel_height = tank_height
        elif surface in ['left', 'right']:
            panel_width = tank_depth
            panel_height = tank_height
        else:
            continue

        weight = calculate_finwall_panel_weight(
            panel_width, panel_height, fin_depth,
            sheet_thickness, pitch, steel_density
        )
        total_weight += weight

    return total_weight


def calculate_finwall_details(tank_width, tank_depth, tank_height,
                               finwall_surfaces, pitch=FINWALL_PITCH):
    """
    Calculate number of fins on each selected surface.

    Args:
        tank_width, tank_depth, tank_height: Tank dimensions (mm)
        finwall_surfaces: List of surfaces ['front', 'back', 'left', 'right']
        pitch: Fin pitch (mm)

    Returns:
        dict: Details per surface {'front': {'width': mm, 'height': mm, 'n_fins': int}, ...}
    """
    details = {}

    for surface in finwall_surfaces:
        if surface in ['front', 'back']:
            panel_width = tank_width
            panel_height = tank_height
        elif surface in ['left', 'right']:
            panel_width = tank_depth
            panel_height = tank_height
        else:
            continue

        n_fins = int(panel_width / pitch)

        details[surface] = {
            'width': panel_width,
            'height': panel_height,
            'n_fins': n_fins
        }

    return details


# =============================================================================
# OIL CALCULATIONS
# =============================================================================

# Material densities for volume calculation (g/cm³)
CORE_DENSITY = 7.65      # Silicon steel
COPPER_DENSITY = 8.9     # Copper conductors
ALUMINUM_DENSITY = 2.7   # Aluminum conductors


def calculate_oil_volume_accurate(tank_width, tank_depth, tank_height,
                                   core_weight, lv_weight, hv_weight,
                                   core_density=CORE_DENSITY,
                                   lv_density=COPPER_DENSITY,
                                   hv_density=COPPER_DENSITY,
                                   fill_level_percent=DEFAULT_OIL_FILL_LEVEL):
    """
    Calculate oil volume using actual component volumes from optimization.

    Oil fills the space between tank walls and solid components:
    Oil_Volume = Tank_Volume - (Core_Volume + LV_Volume + HV_Volume)

    Args:
        tank_width, tank_depth, tank_height: Tank internal dimensions (mm)
        core_weight: Core weight from optimization (kg)
        lv_weight: LV winding weight from optimization (kg)
        hv_weight: HV winding weight from optimization (kg)
        core_density: Core material density (g/cm³)
        lv_density: LV conductor density (g/cm³)
        hv_density: HV conductor density (g/cm³)
        fill_level_percent: Fill level percentage (default 85%)

    Returns:
        float: Oil volume in liters
    """
    # Tank internal volume in liters
    # mm³ to liters: divide by 1,000,000
    tank_volume_liters = (tank_width * tank_depth * tank_height) / 1_000_000.0

    # Convert component weights to volumes
    # Weight (kg) / Density (g/cm³) = Volume (liters)
    # Because: kg / (g/cm³) = 1000g / (g/cm³) = 1000 cm³ = 1 liter
    core_volume_liters = core_weight / core_density
    lv_volume_liters = lv_weight / lv_density
    hv_volume_liters = hv_weight / hv_density

    # Total solid volume
    solid_volume_liters = core_volume_liters + lv_volume_liters + hv_volume_liters

    # Available space for oil
    available_volume = tank_volume_liters - solid_volume_liters

    # Apply fill level (typically 85% to allow for thermal expansion)
    oil_volume_liters = available_volume * (fill_level_percent / 100.0)

    return max(0.0, oil_volume_liters)


def calculate_oil_weight_and_price(oil_volume_liters, oil_type='mineral',
                                    custom_price_per_liter=None):
    """
    Calculate oil weight and price.

    Args:
        oil_volume_liters: Oil volume (L)
        oil_type: Oil type key ('mineral', 'natural_ester', 'silicone')
        custom_price_per_liter: Override default price (optional)

    Returns:
        dict: {'weight_kg': float, 'price': float, 'type_name': str}
    """
    oil_props = OIL_TYPES.get(oil_type, OIL_TYPES['mineral'])

    weight_kg = oil_volume_liters * oil_props['density']

    if custom_price_per_liter is not None:
        price = oil_volume_liters * custom_price_per_liter
    else:
        price = oil_volume_liters * oil_props['price']

    return {
        'weight_kg': weight_kg,
        'price': price,
        'type_name': oil_props['name']
    }


# =============================================================================
# MASTER CALCULATION FUNCTION
# =============================================================================

def calculate_tank_and_oil(
    # Optimization result parameters
    core_diameter,
    core_length,
    lv_turns,
    lv_height,
    lv_thickness,
    hv_thickness,
    hv_length,
    no_load_loss,
    load_loss,
    # Weights from optimization (for accurate oil volume)
    core_weight,
    lv_weight,
    hv_weight,
    n_ducts_lv=0,
    n_ducts_hv=0,
    circular_hv=False,
    # Tank configuration
    clearance_x=DEFAULT_CLEARANCE_X,
    clearance_y=DEFAULT_CLEARANCE_Y,
    clearance_z=DEFAULT_CLEARANCE_Z,
    tank_wall_thickness=DEFAULT_TANK_WALL_THICKNESS,
    steel_price_per_kg=STEEL_PRICE_DEFAULT,
    # Finwall configuration
    enable_finwall=True,
    finwall_surfaces=None,
    finwall_auto_optimize=True,
    target_temp_rise=DEFAULT_TARGET_TEMP_RISE,
    fin_depth_manual=80.0,
    finwall_thickness=1.0,
    # Oil configuration
    oil_type='mineral',
    oil_price_per_liter=None,
    oil_fill_level=DEFAULT_OIL_FILL_LEVEL,
    # Material densities (for oil volume calculation)
    core_density=CORE_DENSITY,
    lv_density=COPPER_DENSITY,
    hv_density=COPPER_DENSITY
):
    """
    Complete tank and oil calculation (POST-OPTIMIZATION).

    This function is called AFTER optimization completes, using the
    optimized transformer parameters to calculate tank/oil costs.

    Args:
        # From optimization result:
        core_diameter, core_length, lv_turns, lv_height, lv_thickness,
        hv_thickness, hv_length: Optimized design parameters
        no_load_loss, load_loss: Calculated losses (W)
        n_ducts_lv, n_ducts_hv: Cooling duct counts
        circular_hv: Whether HV wire is circular

        # Tank configuration:
        clearance_x, y, z: Clearances in mm (default 40)
        tank_wall_thickness: mm (default 3.0)
        steel_price_per_kg: $/kg (default 2.50)

        # Finwall configuration:
        enable_finwall: Whether to add finwalls
        finwall_surfaces: List of surfaces (default all 4 sides)
        finwall_auto_optimize: Auto-calculate fin depth
        target_temp_rise: Target temp rise for auto-optimization (K)
        fin_depth_manual: Manual fin depth if auto=False (mm)
        finwall_thickness: Steel sheet thickness (mm)

        # Oil configuration:
        oil_type: 'mineral', 'natural_ester', or 'silicone'
        oil_price_per_liter: Custom price override
        oil_fill_level: Fill percentage

    Returns:
        dict: Complete tank and oil calculation results
    """
    if finwall_surfaces is None:
        finwall_surfaces = ['front', 'back', 'left', 'right']

    # Step 1: Calculate active part dimensions
    active_dims = calculate_active_part_dimensions(
        core_diameter, core_length, lv_turns, lv_height, lv_thickness,
        hv_thickness, hv_length, n_ducts_lv, n_ducts_hv, circular_hv
    )

    # Step 2: Calculate tank dimensions
    tank_dims = calculate_tank_dimensions(
        active_dims['width'], active_dims['depth'], active_dims['height'],
        clearance_x, clearance_y, clearance_z
    )

    # Step 3: Calculate tank shell weight
    tank_shell_weight = calculate_tank_shell_weight(
        tank_dims['width'], tank_dims['depth'], tank_dims['height'],
        tank_wall_thickness
    )

    # Step 4: Calculate finwall
    finwall_weight = 0.0
    fin_depth_used = 0.0
    finwall_details = {}

    if enable_finwall and finwall_surfaces:
        # Total losses for finwall optimization
        total_losses = no_load_loss + load_loss

        if finwall_auto_optimize:
            # Auto-calculate optimal fin depth
            fin_depth_used = calculate_optimal_fin_depth(
                total_losses, target_temp_rise,
                tank_dims['width'], tank_dims['depth'], tank_dims['height'],
                finwall_surfaces
            )
        else:
            fin_depth_used = fin_depth_manual

        finwall_weight = calculate_total_finwall_weight(
            tank_dims['width'], tank_dims['depth'], tank_dims['height'],
            fin_depth_used, finwall_thickness, finwall_surfaces
        )

        # Calculate fin count per surface
        finwall_details = calculate_finwall_details(
            tank_dims['width'], tank_dims['depth'], tank_dims['height'],
            finwall_surfaces
        )

    # Step 5: Total tank weight and price
    total_tank_weight = tank_shell_weight + finwall_weight
    tank_price = total_tank_weight * steel_price_per_kg

    # Step 6: Calculate oil volume using actual component weights
    oil_volume = calculate_oil_volume_accurate(
        tank_dims['width'], tank_dims['depth'], tank_dims['height'],
        core_weight, lv_weight, hv_weight,
        core_density, lv_density, hv_density,
        oil_fill_level
    )

    # Step 7: Calculate oil weight and price
    oil_data = calculate_oil_weight_and_price(
        oil_volume, oil_type, oil_price_per_liter
    )

    return {
        # Active part dimensions
        'active_part_width': active_dims['width'],
        'active_part_depth': active_dims['depth'],
        'active_part_height': active_dims['height'],

        # Tank dimensions
        'tank_width': tank_dims['width'],
        'tank_depth': tank_dims['depth'],
        'tank_height': tank_dims['height'],

        # Tank weights and price
        'tank_shell_weight': tank_shell_weight,
        'finwall_weight': finwall_weight,
        'total_tank_weight': total_tank_weight,
        'tank_price': tank_price,

        # Finwall details
        'finwall_enabled': enable_finwall,
        'finwall_surfaces': finwall_surfaces if enable_finwall else [],
        'finwall_details': finwall_details,  # {surface: {'width', 'height', 'n_fins'}}
        'fin_depth': fin_depth_used,
        'finwall_thickness': finwall_thickness,
        'finwall_auto_optimized': finwall_auto_optimize and enable_finwall,

        # Oil data
        'oil_type': oil_type,
        'oil_type_name': oil_data['type_name'],
        'oil_volume_liters': oil_volume,
        'oil_weight': oil_data['weight_kg'],
        'oil_price': oil_data['price'],

        # Configuration used
        'clearances': {
            'x': clearance_x,
            'y': clearance_y,
            'z': clearance_z
        },
        'target_temp_rise': target_temp_rise if finwall_auto_optimize else None
    }
