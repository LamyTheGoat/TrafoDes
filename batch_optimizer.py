"""
Batch optimization for multiple transformer specifications.

Supports:
- Manual spec input
- CSV file upload
- Parametric sweep
- Parallel execution with progress tracking
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import pandas as pd
from io import StringIO


# =============================================================================
# MATERIAL PRESETS
# =============================================================================

@dataclass
class MaterialPreset:
    """Material properties for transformer components."""
    name: str
    density: float          # kg/dm³
    resistivity: float      # ohm·mm²/m
    price_foil: float       # $/kg (for LV foil)
    price_wire: float       # $/kg (for HV wire)


# Material constants
MATERIAL_COPPER = MaterialPreset(
    name="Copper",
    density=8.9,
    resistivity=0.021,
    price_foil=11.55,
    price_wire=11.05,
)

MATERIAL_ALUMINUM = MaterialPreset(
    name="Aluminum",
    density=2.7,
    resistivity=0.0336,
    price_foil=6.00,       # Estimated Al foil price $/kg
    price_wire=5.80,       # Estimated Al wire price $/kg
)

# Core steel options
CORE_STEEL_STANDARD = {
    "name": "Standard",
    "density": 7.65,
    "price_per_kg": 3.6,
}

CORE_STEEL_HIPERCO = {
    "name": "HIPERCO",
    "density": 8.12,
    "price_per_kg": 45.0,   # Premium steel
}

CORE_STEEL_AMORPHOUS = {
    "name": "Amorphous",
    "density": 7.18,
    "price_per_kg": 12.0,   # Low loss steel
}

# Available presets for UI
WINDING_MATERIALS = {
    "Copper": MATERIAL_COPPER,
    "Aluminum": MATERIAL_ALUMINUM,
}

CORE_MATERIALS = {
    "Standard": CORE_STEEL_STANDARD,
    "HIPERCO": CORE_STEEL_HIPERCO,
    "Amorphous": CORE_STEEL_AMORPHOUS,
}


@dataclass
class TransformerSpec:
    """Specification for a single transformer to optimize."""
    name: str
    power: float                    # kVA
    hv_voltage: float               # V
    lv_voltage: float               # V
    nll_limit: float                # W (No-Load Loss guarantee)
    ll_limit: float                 # W (Load Loss guarantee)
    ucc_target: float = 6.0         # % (Impedance target)
    ucc_tolerance: float = 10.0     # % tolerance on impedance
    obround: bool = True            # Stadium/obround core shape
    put_cooling_ducts: bool = True  # Calculate cooling ducts


@dataclass
class BatchResult:
    """Result from a single optimization in a batch."""
    spec: TransformerSpec
    result: Dict                    # Full optimization result
    elapsed_time: float
    success: bool
    error_message: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for export, merging spec and result."""
        d = {
            'name': self.spec.name,
            'power': self.spec.power,
            'hv_voltage': self.spec.hv_voltage,
            'lv_voltage': self.spec.lv_voltage,
            'nll_limit': self.spec.nll_limit,
            'll_limit': self.spec.ll_limit,
            'ucc_target': self.spec.ucc_target,
            'success': self.success,
            'error_message': self.error_message,
            'batch_time': self.elapsed_time,
        }
        if self.result:
            d.update(self.result)
        return d


def parse_specs_from_csv(csv_content: str) -> List[TransformerSpec]:
    """
    Parse transformer specifications from CSV content.

    Expected columns (case-insensitive):
        - name: Transformer name/ID
        - power: Power rating in kVA
        - hv_voltage: HV voltage in V
        - lv_voltage: LV voltage in V
        - nll_limit: No-load loss limit in W
        - ll_limit: Load loss limit in W
        - ucc_target (optional): Impedance target in %
        - obround (optional): True/False for core shape

    Args:
        csv_content: CSV file content as string

    Returns:
        List of TransformerSpec objects
    """
    df = pd.read_csv(StringIO(csv_content))

    # Normalize column names to lowercase
    df.columns = [c.lower().strip().replace(' ', '_') for c in df.columns]

    specs = []
    for _, row in df.iterrows():
        spec = TransformerSpec(
            name=str(row.get('name', f'Trafo-{len(specs)+1}')),
            power=float(row['power']),
            hv_voltage=float(row['hv_voltage']),
            lv_voltage=float(row['lv_voltage']),
            nll_limit=float(row['nll_limit']),
            ll_limit=float(row['ll_limit']),
            ucc_target=float(row.get('ucc_target', 6.0)),
            ucc_tolerance=float(row.get('ucc_tolerance', 10.0)),
            obround=bool(row.get('obround', True)),
            put_cooling_ducts=bool(row.get('put_cooling_ducts', True)),
        )
        specs.append(spec)

    return specs


def parse_specs_from_dataframe(df: pd.DataFrame) -> List[TransformerSpec]:
    """
    Parse transformer specifications from pandas DataFrame.

    Args:
        df: DataFrame with transformer specifications

    Returns:
        List of TransformerSpec objects
    """
    # Normalize column names
    df = df.copy()
    df.columns = [c.lower().strip().replace(' ', '_') for c in df.columns]

    specs = []
    for idx, row in df.iterrows():
        spec = TransformerSpec(
            name=str(row.get('name', f'Trafo-{idx+1}')),
            power=float(row['power']),
            hv_voltage=float(row['hv_voltage']),
            lv_voltage=float(row['lv_voltage']),
            nll_limit=float(row['nll_limit']),
            ll_limit=float(row['ll_limit']),
            ucc_target=float(row.get('ucc_target', 6.0)),
            ucc_tolerance=float(row.get('ucc_tolerance', 10.0)),
            obround=bool(row.get('obround', True)) if 'obround' in row else True,
            put_cooling_ducts=bool(row.get('put_cooling_ducts', True)) if 'put_cooling_ducts' in row else True,
        )
        specs.append(spec)

    return specs


def generate_parametric_sweep(
    power_values: List[float],
    hv_voltage: float = 33000,
    lv_voltage: float = 400,
    nll_scale: float = 2.0,      # NLL limit = power * nll_scale
    ll_scale: float = 10.0,      # LL limit = power * ll_scale
    ucc_target: float = 6.0,
    obround: bool = True
) -> List[TransformerSpec]:
    """
    Generate transformer specifications for parametric sweep.

    Args:
        power_values: List of power ratings to sweep
        hv_voltage: Fixed HV voltage
        lv_voltage: Fixed LV voltage
        nll_scale: NLL limit = power * nll_scale
        ll_scale: LL limit = power * ll_scale
        ucc_target: Impedance target
        obround: Core shape

    Returns:
        List of TransformerSpec objects
    """
    specs = []
    for power in power_values:
        spec = TransformerSpec(
            name=f'Trafo-{int(power)}kVA',
            power=power,
            hv_voltage=hv_voltage,
            lv_voltage=lv_voltage,
            nll_limit=power * nll_scale,
            ll_limit=power * ll_scale,
            ucc_target=ucc_target,
            obround=obround,
        )
        specs.append(spec)

    return specs


def run_batch_optimization(
    specs: List[TransformerSpec],
    parallel: bool = True,
    max_workers: int = 4,
    method: str = 'hybrid',
    depth: str = 'fast',
    tolerance: float = 25,
    multi_runs: int = 1,
    loss_target_factor: float = 1.0,
    progress_callback: Optional[Callable[[int, int, str, float], None]] = None,
    lv_material: Optional[MaterialPreset] = None,
    hv_material: Optional[MaterialPreset] = None,
    core_material: Optional[Dict] = None,
) -> List[BatchResult]:
    """
    Run optimization for multiple transformer specifications.

    Args:
        specs: List of transformer specifications
        parallel: Run in parallel (True) or sequential (False)
        max_workers: Max parallel threads (only if parallel=True)
        method: Optimization method ('hybrid', 'de', 'smart', etc.)
        depth: Search depth for hybrid method ('fast', 'normal', 'thorough')
        tolerance: Constraint tolerance percentage
        multi_runs: Number of DE runs per transformer, picks best result (1=single run)
        loss_target_factor: Target losses at this fraction of limit (0.95 = target 95% of limit for cheaper designs)
        progress_callback: Optional function(completed, total, current_name, elapsed)
        lv_material: Material preset for LV winding (foil), default Copper
        hv_material: Material preset for HV winding (wire), default Copper
        core_material: Core steel properties dict, default Standard

    Returns:
        List of BatchResult objects
    """
    # Import mainRect here to avoid circular imports
    import mainRect

    results = []
    total = len(specs)
    batch_start = time.time()

    def optimize_single(spec: TransformerSpec) -> BatchResult:
        """Optimize a single transformer specification."""
        start = time.time()
        try:
            # Save original module values
            orig_power = mainRect.POWERRATING
            orig_hv = mainRect.HVRATE
            orig_lv = mainRect.LVRATE
            orig_nll = mainRect.GUARANTEED_NO_LOAD_LOSS
            orig_ll = mainRect.GUARANTEED_LOAD_LOSS
            orig_ucc = mainRect.GUARANTEED_UCC

            # Save original material values
            orig_foil_density = mainRect.materialToBeUsedFoil_Density
            orig_foil_price = mainRect.materialToBeUsedFoil_Price
            orig_foil_resistivity = mainRect.materialToBeUsedFoil_Resistivity
            orig_wire_density = mainRect.materialToBeUsedWire_Density
            orig_wire_price = mainRect.materialToBeUsedWire_Price
            orig_wire_resistivity = mainRect.materialToBeUsedWire_Resistivity
            orig_core_density = mainRect.materialCore_Density
            orig_core_price = mainRect.materialCore_Price

            try:
                # Set module values for this spec
                # Note: LVRATE expects phase voltage, user inputs line voltage
                # For Y connection: phase = line / sqrt(3)
                mainRect.POWERRATING = spec.power
                mainRect.HVRATE = spec.hv_voltage
                mainRect.LVRATE = spec.lv_voltage / (3 ** 0.5)  # Convert line to phase voltage
                # Apply loss_target_factor to push design closer to limits (cheaper)
                # e.g., factor=0.95 means target 95% of limit
                mainRect.GUARANTEED_NO_LOAD_LOSS = spec.nll_limit * loss_target_factor
                mainRect.GUARANTEED_LOAD_LOSS = spec.ll_limit * loss_target_factor
                mainRect.GUARANTEED_UCC = spec.ucc_target

                # Set LV material (foil)
                if lv_material is not None:
                    mainRect.materialToBeUsedFoil_Density = lv_material.density
                    mainRect.materialToBeUsedFoil_Price = lv_material.price_foil
                    mainRect.materialToBeUsedFoil_Resistivity = lv_material.resistivity

                # Set HV material (wire)
                if hv_material is not None:
                    mainRect.materialToBeUsedWire_Density = hv_material.density
                    mainRect.materialToBeUsedWire_Price = hv_material.price_wire
                    mainRect.materialToBeUsedWire_Resistivity = hv_material.resistivity

                # Set core material
                if core_material is not None:
                    mainRect.materialCore_Density = core_material["density"]
                    mainRect.materialCore_Price = core_material["price_per_kg"]

                # Run optimization (multiple runs if multi_runs > 1)
                best_result = None
                best_price = float('inf')

                for run_idx in range(multi_runs):
                    result = mainRect.StartFast(
                        tolerance=tolerance,
                        obround=spec.obround,
                        put_cooling_ducts=spec.put_cooling_ducts,
                        method=method,
                        search_depth=depth,
                        print_result=False
                    )

                    if result is not None:
                        price = result.get('total_price', float('inf'))
                        if price < best_price:
                            best_price = price
                            best_result = result

                if best_result is None:
                    return BatchResult(
                        spec=spec,
                        result={},
                        elapsed_time=time.time() - start,
                        success=False,
                        error_message="No valid design found"
                    )

                # Add spec info to result
                best_result['power'] = spec.power
                best_result['hv_voltage'] = spec.hv_voltage
                best_result['lv_voltage'] = spec.lv_voltage
                best_result['name'] = spec.name
                best_result['runs_completed'] = multi_runs

                return BatchResult(
                    spec=spec,
                    result=best_result,
                    elapsed_time=time.time() - start,
                    success=True
                )

            finally:
                # Restore original values
                mainRect.POWERRATING = orig_power
                mainRect.HVRATE = orig_hv
                mainRect.LVRATE = orig_lv
                mainRect.GUARANTEED_NO_LOAD_LOSS = orig_nll
                mainRect.GUARANTEED_LOAD_LOSS = orig_ll
                mainRect.GUARANTEED_UCC = orig_ucc

                # Restore original material values
                mainRect.materialToBeUsedFoil_Density = orig_foil_density
                mainRect.materialToBeUsedFoil_Price = orig_foil_price
                mainRect.materialToBeUsedFoil_Resistivity = orig_foil_resistivity
                mainRect.materialToBeUsedWire_Density = orig_wire_density
                mainRect.materialToBeUsedWire_Price = orig_wire_price
                mainRect.materialToBeUsedWire_Resistivity = orig_wire_resistivity
                mainRect.materialCore_Density = orig_core_density
                mainRect.materialCore_Price = orig_core_price

        except Exception as e:
            return BatchResult(
                spec=spec,
                result={},
                elapsed_time=time.time() - start,
                success=False,
                error_message=str(e)
            )

    if parallel and max_workers > 1:
        # Parallel execution
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(optimize_single, spec): spec for spec in specs}

            for i, future in enumerate(as_completed(futures)):
                result = future.result()
                results.append(result)

                if progress_callback:
                    elapsed = time.time() - batch_start
                    progress_callback(i + 1, total, result.spec.name, elapsed)
    else:
        # Sequential execution
        for i, spec in enumerate(specs):
            result = optimize_single(spec)
            results.append(result)

            if progress_callback:
                elapsed = time.time() - batch_start
                progress_callback(i + 1, total, spec.name, elapsed)

    # Sort results by original spec order
    spec_order = {spec.name: i for i, spec in enumerate(specs)}
    results.sort(key=lambda r: spec_order.get(r.spec.name, 999))

    return results


def batch_results_to_dataframe(results: List[BatchResult]) -> pd.DataFrame:
    """
    Convert batch results to pandas DataFrame for display/export.

    Args:
        results: List of BatchResult objects

    Returns:
        DataFrame with all results
    """
    rows = []
    for r in results:
        row = r.to_dict()
        rows.append(row)

    return pd.DataFrame(rows)


def get_default_specs_dataframe() -> pd.DataFrame:
    """
    Get a default/template DataFrame for batch input.

    Returns:
        DataFrame with example transformer specifications
    """
    data = {
        'name': ['Trafo-400', 'Trafo-630', 'Trafo-1000'],
        'power': [400, 630, 1000],
        'hv_voltage': [33000, 33000, 33000],
        'lv_voltage': [400, 400, 400],
        'nll_limit': [600, 800, 1100],
        'll_limit': [4500, 6500, 9500],
        'ucc_target': [6.0, 6.0, 6.0],
        'obround': [True, True, True],
    }
    return pd.DataFrame(data)


# Template CSV content for download
TEMPLATE_CSV = """name,power,hv_voltage,lv_voltage,nll_limit,ll_limit,ucc_target,obround
Trafo-250,250,33000,400,450,3500,6.0,True
Trafo-400,400,33000,400,600,4500,6.0,True
Trafo-630,630,33000,400,800,6500,6.0,True
Trafo-1000,1000,20000,400,1100,9500,6.0,True
"""


@dataclass
class InverseResult:
    """Result from inverse optimization (margin maximization within budget)."""
    power: float                    # Optimal power rating found
    price: float                    # Actual price achieved
    design: Dict                    # Full design parameters
    nll: float                      # Achieved no-load loss (W)
    ll: float                       # Achieved load loss (W)
    nll_margin: float               # NLL margin (fraction below limit)
    ll_margin: float                # LL margin (fraction below limit)
    total_margin: float             # Combined NLL + LL margin (higher = better)
    ucc: float                      # Achieved impedance (%)
    success: bool
    message: str
    candidates_evaluated: int       # Number of optimization runs


def _run_single_optimization(power: float, hv_voltage: float, lv_voltage: float,
                              nll_limit: float, ll_limit: float, ucc_target: float,
                              method: str, depth: str,
                              lv_material: Optional[MaterialPreset] = None,
                              hv_material: Optional[MaterialPreset] = None,
                              core_material: Optional[Dict] = None) -> Optional[Dict]:
    """
    Run a single optimization with given parameters.

    Args:
        power: Power rating (kVA)
        hv_voltage: HV voltage (V)
        lv_voltage: LV voltage (V)
        nll_limit: No-load loss limit (W)
        ll_limit: Load loss limit (W)
        ucc_target: Impedance target (%)
        method: Optimization method
        depth: Search depth
        lv_material: Material preset for LV winding (foil), default Copper
        hv_material: Material preset for HV winding (wire), default Copper
        core_material: Core steel properties dict, default Standard

    Returns:
        Result dict or None if optimization failed
    """
    import mainRect

    # Save original module values
    orig_power = mainRect.POWERRATING
    orig_hv = mainRect.HVRATE
    orig_lv = mainRect.LVRATE
    orig_nll = mainRect.GUARANTEED_NO_LOAD_LOSS
    orig_ll = mainRect.GUARANTEED_LOAD_LOSS
    orig_ucc = mainRect.GUARANTEED_UCC

    # Save original material values
    orig_foil_density = mainRect.materialToBeUsedFoil_Density
    orig_foil_price = mainRect.materialToBeUsedFoil_Price
    orig_foil_resistivity = mainRect.materialToBeUsedFoil_Resistivity
    orig_wire_density = mainRect.materialToBeUsedWire_Density
    orig_wire_price = mainRect.materialToBeUsedWire_Price
    orig_wire_resistivity = mainRect.materialToBeUsedWire_Resistivity
    orig_core_density = mainRect.materialCore_Density
    orig_core_price = mainRect.materialCore_Price

    try:
        # Set module values for this optimization
        # Note: LVRATE expects phase voltage, user inputs line voltage
        # For Y connection: phase = line / sqrt(3)
        mainRect.POWERRATING = power
        mainRect.HVRATE = hv_voltage
        mainRect.LVRATE = lv_voltage / (3 ** 0.5)  # Convert line to phase voltage
        mainRect.GUARANTEED_NO_LOAD_LOSS = nll_limit
        mainRect.GUARANTEED_LOAD_LOSS = ll_limit
        mainRect.GUARANTEED_UCC = ucc_target

        # Set LV material (foil)
        if lv_material is not None:
            mainRect.materialToBeUsedFoil_Density = lv_material.density
            mainRect.materialToBeUsedFoil_Price = lv_material.price_foil
            mainRect.materialToBeUsedFoil_Resistivity = lv_material.resistivity

        # Set HV material (wire)
        if hv_material is not None:
            mainRect.materialToBeUsedWire_Density = hv_material.density
            mainRect.materialToBeUsedWire_Price = hv_material.price_wire
            mainRect.materialToBeUsedWire_Resistivity = hv_material.resistivity

        # Set core material
        if core_material is not None:
            mainRect.materialCore_Density = core_material["density"]
            mainRect.materialCore_Price = core_material["price_per_kg"]

        # Run optimization
        result = mainRect.StartFast(
            tolerance=25,
            obround=True,
            put_cooling_ducts=True,
            method=method,
            search_depth=depth,
            print_result=False
        )
        return result

    except Exception:
        return None

    finally:
        # Restore original values
        mainRect.POWERRATING = orig_power
        mainRect.HVRATE = orig_hv
        mainRect.LVRATE = orig_lv
        mainRect.GUARANTEED_NO_LOAD_LOSS = orig_nll
        mainRect.GUARANTEED_LOAD_LOSS = orig_ll
        mainRect.GUARANTEED_UCC = orig_ucc

        # Restore original material values
        mainRect.materialToBeUsedFoil_Density = orig_foil_density
        mainRect.materialToBeUsedFoil_Price = orig_foil_price
        mainRect.materialToBeUsedFoil_Resistivity = orig_foil_resistivity
        mainRect.materialToBeUsedWire_Density = orig_wire_density
        mainRect.materialToBeUsedWire_Price = orig_wire_price
        mainRect.materialToBeUsedWire_Resistivity = orig_wire_resistivity
        mainRect.materialCore_Density = orig_core_density
        mainRect.materialCore_Price = orig_core_price


def _binary_search_margin(
    power: float,
    hv_voltage: float,
    lv_voltage: float,
    guaranteed_nll: float,
    guaranteed_ll: float,
    ucc_target: float,
    max_price: float,
    method: str,
    depth: str,
    num_iterations: int = 6,
    lv_material: Optional[MaterialPreset] = None,
    hv_material: Optional[MaterialPreset] = None,
    core_material: Optional[Dict] = None,
    loss_tolerance: float = 0.85,
) -> Optional[Dict]:
    """
    Binary search to find the tightest loss limits that fit within budget.
    Returns the best result found, or None if nothing fits budget.

    Args:
        loss_tolerance: How much losses can exceed limits (0.85 = 85% over allowed)
    """
    # Search range for multiplier: 0.3 (tight) to 8.0 (very loose)
    # Very wide range needed for cheap/small transformers where losses scale differently
    mult_low = 0.3
    mult_high = 8.0
    best_result = None
    best_margin = -float('inf')

    # First, try the loosest limits to see if ANY solution fits budget
    result = _run_single_optimization(
        power, hv_voltage, lv_voltage,
        guaranteed_nll * mult_high, guaranteed_ll * mult_high, ucc_target,
        method, depth,
        lv_material, hv_material, core_material
    )

    if result is None:
        return None  # Can't even find a solution with very loose limits

    price = result.get('total_price', float('inf'))
    if price > max_price:
        return None  # Even loosest limits don't fit budget

    # We have at least one solution that fits budget, but check if it meets loss limits
    nll = result.get('no_load_loss', 0)
    ll = result.get('load_loss', 0)
    ucc = result.get('impedance', ucc_target)
    nll_margin = (guaranteed_nll - nll) / guaranteed_nll if guaranteed_nll > 0 else 0
    ll_margin = (guaranteed_ll - ll) / guaranteed_ll if guaranteed_ll > 0 else 0
    total_margin = 0.5 * nll_margin + 0.5 * ll_margin

    # Accept if margins are within tolerance (negative margin = over limit)
    # loss_tolerance=0.10 means allow up to 10% over limit (margin >= -0.10)
    if nll_margin >= -loss_tolerance and ll_margin >= -loss_tolerance:
        best_result = {
            'power': power,
            'price': price,
            'design': result,
            'nll': nll,
            'll': ll,
            'nll_margin': nll_margin,
            'll_margin': ll_margin,
            'total_margin': total_margin,
            'ucc': ucc,
            'multiplier': mult_high,
        }
        best_margin = total_margin
    else:
        # Solution exceeds tolerance - not valid
        best_result = None
        best_margin = -float('inf')

    # Now binary search to find tighter limits that still fit budget
    for _ in range(num_iterations):
        mult_mid = (mult_low + mult_high) / 2
        nll_limit = guaranteed_nll * mult_mid
        ll_limit = guaranteed_ll * mult_mid

        result = _run_single_optimization(
            power, hv_voltage, lv_voltage,
            nll_limit, ll_limit, ucc_target,
            method, depth,
            lv_material, hv_material, core_material
        )

        if result is None:
            # Couldn't find solution with these limits, try looser
            mult_low = mult_mid
            continue

        price = result.get('total_price', float('inf'))

        if price <= max_price:
            # Fits budget! Calculate margin and check if it meets loss limits
            nll = result.get('no_load_loss', 0)
            ll = result.get('load_loss', 0)
            ucc = result.get('impedance', ucc_target)

            nll_margin = (guaranteed_nll - nll) / guaranteed_nll if guaranteed_nll > 0 else 0
            ll_margin = (guaranteed_ll - ll) / guaranteed_ll if guaranteed_ll > 0 else 0
            total_margin = 0.5 * nll_margin + 0.5 * ll_margin

            # Accept if margins are within tolerance and better than previous
            if nll_margin >= -loss_tolerance and ll_margin >= -loss_tolerance and total_margin > best_margin:
                best_margin = total_margin
                best_result = {
                    'power': power,
                    'price': price,
                    'design': result,
                    'nll': nll,
                    'll': ll,
                    'nll_margin': nll_margin,
                    'll_margin': ll_margin,
                    'total_margin': total_margin,
                    'ucc': ucc,
                    'multiplier': mult_mid,
                }

            # Try even tighter limits
            mult_high = mult_mid
        else:
            # Too expensive, try looser limits
            mult_low = mult_mid

    return best_result


def run_inverse_optimization(
    target_price: float,
    power_min: float,
    power_max: float,
    hv_voltage: float = 33000,
    lv_voltage: float = 400,
    nll_limit_scale: float = 10.0,   # Very loose limits for cheapest designs
    ll_limit_scale: float = 50.0,   # Very loose limits for cheapest designs
    ucc_target: float = 6.0,
    price_tolerance: float = 0.05,
    power_step: float = 25,
    method: str = 'de',
    depth: str = 'fast',
    loss_tolerance: float = 0.85,
    progress_callback: Optional[Callable[[int, int, float, str], None]] = None,
    lv_material: Optional[MaterialPreset] = None,
    hv_material: Optional[MaterialPreset] = None,
    core_material: Optional[Dict] = None,
) -> InverseResult:
    """
    Inverse optimization: Given a price budget, find the HIGHEST POWER transformer.

    Priority order:
    1. Maximize power rating (biggest transformer that fits budget)
    2. Then maximize margin (best reliability at that power)

    Uses binary search to efficiently find the maximum achievable power,
    then optimizes margin at that power level.

    Args:
        target_price: Maximum acceptable price ($)
        power_min: Minimum power rating (kVA)
        power_max: Maximum power rating (kVA)
        hv_voltage: HV voltage (V)
        lv_voltage: LV voltage (V)
        nll_limit_scale: Guaranteed NLL = power * scale (W/kVA)
        ll_limit_scale: Guaranteed LL = power * scale (W/kVA)
        ucc_target: Impedance target (%)
        price_tolerance: Acceptable price overshoot (e.g., 0.05 = 5%)
        power_step: Power increment for sweep (kVA)
        method: Optimization method
        depth: Search depth
        progress_callback: Optional fn(step, total_steps, power, status)
        lv_material: Material preset for LV winding (foil), default Copper
        hv_material: Material preset for HV winding (wire), default Copper
        core_material: Core steel properties dict, default Standard

    Returns:
        InverseResult with highest power rating that fits budget
    """
    max_price = target_price * (1 + price_tolerance)
    candidates_evaluated = 0

    # Full scan approach: Cost is NOT monotonic with power at high voltages
    # (small transformers can be more expensive than larger ones due to insulation costs)
    # So we scan ALL power levels and keep track of the HIGHEST power that fits budget
    if progress_callback:
        progress_callback(1, 4, power_min, "Scanning all power levels...")

    best_power_result = None
    best_power = 0

    # Scan ALL power levels from min to max
    power_levels = list(range(int(power_min), int(power_max) + 1, int(power_step)))
    total_levels = len(power_levels)

    for i, test_power in enumerate(power_levels):
        if progress_callback:
            progress_callback(1, 4, test_power, f"Testing {test_power:.0f} kVA ({i+1}/{total_levels})...")

        guaranteed_nll = test_power * nll_limit_scale
        guaranteed_ll = test_power * ll_limit_scale

        result = _binary_search_margin(
            power=test_power,
            hv_voltage=hv_voltage,
            lv_voltage=lv_voltage,
            guaranteed_nll=guaranteed_nll,
            guaranteed_ll=guaranteed_ll,
            ucc_target=ucc_target,
            max_price=max_price,
            method=method,
            depth=depth,
            num_iterations=4,  # Fewer iterations since we're scanning more levels
            lv_material=lv_material,
            hv_material=hv_material,
            core_material=core_material,
            loss_tolerance=loss_tolerance,
        )
        candidates_evaluated += 4

        if result is not None and test_power > best_power:
            # Found a working power level that's higher than previous best!
            best_power = test_power
            best_power_result = result

    if best_power_result is None:
        # Couldn't find any power level that fits budget
        return InverseResult(
            power=0, price=0, design={}, nll=0, ll=0,
            nll_margin=0, ll_margin=0, total_margin=0, ucc=0,
            success=False,
            message=f"No valid design found in range {power_min:.0f}-{power_max:.0f} kVA for budget ${target_price:.0f}",
            candidates_evaluated=candidates_evaluated
        )

    # Phase 2: Fine-tune power around best found
    best_power = best_power_result['power']
    fine_step = power_step / 4

    if progress_callback:
        progress_callback(3, 4, best_power, "Fine-tuning power level...")

    # Try slightly higher powers
    for offset in [1, 2, 3]:
        test_power = best_power + offset * fine_step
        if test_power > power_max:
            break

        guaranteed_nll = test_power * nll_limit_scale
        guaranteed_ll = test_power * ll_limit_scale

        result = _binary_search_margin(
            power=test_power,
            hv_voltage=hv_voltage,
            lv_voltage=lv_voltage,
            guaranteed_nll=guaranteed_nll,
            guaranteed_ll=guaranteed_ll,
            ucc_target=ucc_target,
            max_price=max_price,
            method=method,
            depth=depth,
            num_iterations=5,
            lv_material=lv_material,
            hv_material=hv_material,
            core_material=core_material,
            loss_tolerance=loss_tolerance,
        )
        candidates_evaluated += 5

        if result is not None:
            # Higher power fits! Update best
            best_power_result = result
            best_power = test_power
        else:
            # Can't go higher
            break

    # Phase 4: Optimize margin at final power level
    final_power = best_power_result['power']
    guaranteed_nll = final_power * nll_limit_scale
    guaranteed_ll = final_power * ll_limit_scale

    if progress_callback:
        progress_callback(4, 4, final_power, "Optimizing margin at best power...")

    # Run with better depth for final result
    best_mult = best_power_result.get('multiplier', 1.0)

    # Try multipliers around the best found, with wider range for loose specs
    for mult in [best_mult - 0.1, best_mult, best_mult + 0.1]:
        if mult < 0.3 or mult > 8.0:
            continue

        nll_limit = guaranteed_nll * mult
        ll_limit = guaranteed_ll * mult

        result = _run_single_optimization(
            final_power, hv_voltage, lv_voltage,
            nll_limit, ll_limit, ucc_target,
            method, 'normal',
            lv_material, hv_material, core_material
        )
        candidates_evaluated += 1

        if result is not None:
            price = result.get('total_price', float('inf'))
            if price <= max_price:
                nll = result.get('no_load_loss', 0)
                ll = result.get('load_loss', 0)
                ucc = result.get('impedance', ucc_target)

                nll_margin = (guaranteed_nll - nll) / guaranteed_nll if guaranteed_nll > 0 else 0
                ll_margin = (guaranteed_ll - ll) / guaranteed_ll if guaranteed_ll > 0 else 0
                total_margin = 0.5 * nll_margin + 0.5 * ll_margin

                if total_margin > best_power_result['total_margin']:
                    best_power_result = {
                        'power': final_power,
                        'price': price,
                        'design': result,
                        'nll': nll,
                        'll': ll,
                        'nll_margin': nll_margin,
                        'll_margin': ll_margin,
                        'total_margin': total_margin,
                        'ucc': ucc,
                    }

    return InverseResult(
        power=best_power_result['power'],
        price=best_power_result['price'],
        design=best_power_result['design'],
        nll=best_power_result['nll'],
        ll=best_power_result['ll'],
        nll_margin=best_power_result['nll_margin'],
        ll_margin=best_power_result['ll_margin'],
        total_margin=best_power_result['total_margin'],
        ucc=best_power_result['ucc'],
        success=True,
        message=f"Max power: {best_power_result['power']:.0f} kVA @ ${best_power_result['price']:.0f}, "
                f"margins: NLL {best_power_result['nll_margin']*100:.0f}%, LL {best_power_result['ll_margin']*100:.0f}%",
        candidates_evaluated=candidates_evaluated
    )


if __name__ == "__main__":
    # Test with parametric sweep
    print("Testing batch optimizer...")

    # Generate test specs
    specs = generate_parametric_sweep(
        power_values=[250, 400],
        hv_voltage=33000,
        lv_voltage=400,
    )

    print(f"\nGenerated {len(specs)} specifications:")
    for s in specs:
        print(f"  - {s.name}: {s.power} kVA, HV={s.hv_voltage}V, LV={s.lv_voltage}V")

    # Test CSV parsing
    print("\nTesting CSV parsing:")
    parsed = parse_specs_from_csv(TEMPLATE_CSV)
    print(f"Parsed {len(parsed)} specs from CSV template")

    print("\nBatch optimizer ready for use.")
