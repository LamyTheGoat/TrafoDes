"""
Synthetic Data Generation for Transformer Design ML Models

Generates training data by sampling the design parameter space and computing
outputs using the physics-based calculations from mainRect.py.

Usage:
    python generate_dataset.py --n_samples 100000 --output data/raw/transformer_data.h5
"""

import sys
import os
import argparse
import time
import numpy as np
import h5py
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

# Add parent directory to path to import mainRect
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from mainRect import (
    # Constants
    CORE_MINIMUM, CORE_MAXIMUM, CORELENGTH_MINIMUM,
    FOILHEIGHT_MINIMUM, FOILHEIGHT_MAXIMUM,
    FOILTHICKNESS_MINIMUM, FOILTHICKNESS_MAXIMUM,
    FOILTURNS_MINIMUM, FOILTURNS_MAXIMUM,
    HVTHICK_MINIMUM, HVTHICK_MAXIMUM,
    HV_LEN_MINIMUM, HV_LEN_MAXIMUM,
    LVRATE, HVRATE, POWERRATING, FREQUENCY,
    GUARANTEED_NO_LOAD_LOSS, GUARANTEED_LOAD_LOSS, GUARANTEED_UCC, UCC_TOLERANCE,
    MAX_GRADIENT,
    materialToBeUsedWire_Resistivity,
    # Calculation functions
    CalculateVoltsPerTurns,
    CalculateInduction,
    CalculateLoadLosses,
    CalculateNoLoadLosses,
    CalculateUx, CalculateUr, CalculateImpedance,
    CalculatePrice,
    CalculateCoreWeight,
    CalculateNumberOfCoolingDucts_WithLosses,
    CalculateGradientHeatLV,
    CalculateGradientHeatHV,
    CalculateEfficencyOfCoolingDuct,
    CalculateEfficencyOfMainGap,
    CalculateHeatFluxLV,
    CalculateHeatFluxHV,
)

try:
    from scipy.stats import qmc
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Using random sampling instead of LHS.")


@dataclass
class ParameterBounds:
    """Parameter bounds for design space sampling."""
    core_diameter: Tuple[float, float] = (CORE_MINIMUM, CORE_MAXIMUM)
    # core_length is sampled as ratio of core_diameter (0 to 1)
    core_length_ratio: Tuple[float, float] = (0.0, 1.0)
    lv_turns: Tuple[float, float] = (FOILTURNS_MINIMUM, FOILTURNS_MAXIMUM)
    foil_height: Tuple[float, float] = (FOILHEIGHT_MINIMUM, FOILHEIGHT_MAXIMUM)
    foil_thickness: Tuple[float, float] = (FOILTHICKNESS_MINIMUM, FOILTHICKNESS_MAXIMUM)
    hv_thickness: Tuple[float, float] = (HVTHICK_MINIMUM, HVTHICK_MAXIMUM)
    # hv_length is sampled as ratio of range from hv_thickness to HV_LEN_MAXIMUM
    hv_length_ratio: Tuple[float, float] = (0.0, 1.0)


def latin_hypercube_sample(n_samples: int, n_dims: int = 7) -> np.ndarray:
    """
    Generate samples using Latin Hypercube Sampling for better space coverage.

    Args:
        n_samples: Number of samples to generate
        n_dims: Number of dimensions (parameters)

    Returns:
        Array of shape (n_samples, n_dims) with values in [0, 1]
    """
    if SCIPY_AVAILABLE:
        sampler = qmc.LatinHypercube(d=n_dims, seed=42)
        samples = sampler.random(n=n_samples)
    else:
        # Fallback to stratified random sampling
        np.random.seed(42)
        samples = np.zeros((n_samples, n_dims))
        for dim in range(n_dims):
            perm = np.random.permutation(n_samples)
            samples[:, dim] = (perm + np.random.random(n_samples)) / n_samples
    return samples


def scale_samples(samples: np.ndarray, bounds: ParameterBounds) -> np.ndarray:
    """
    Scale [0,1] samples to actual parameter ranges.

    Args:
        samples: Array of shape (n_samples, 7) with values in [0, 1]
        bounds: Parameter bounds

    Returns:
        Array of shape (n_samples, 7) with scaled parameter values
    """
    scaled = np.zeros_like(samples)

    # Core diameter
    scaled[:, 0] = samples[:, 0] * (bounds.core_diameter[1] - bounds.core_diameter[0]) + bounds.core_diameter[0]

    # Core length (as ratio of core diameter)
    scaled[:, 1] = samples[:, 1] * scaled[:, 0]  # 0 to core_diameter

    # LV turns (integer)
    scaled[:, 2] = np.round(samples[:, 2] * (bounds.lv_turns[1] - bounds.lv_turns[0]) + bounds.lv_turns[0])

    # Foil height
    scaled[:, 3] = samples[:, 3] * (bounds.foil_height[1] - bounds.foil_height[0]) + bounds.foil_height[0]

    # Foil thickness
    scaled[:, 4] = samples[:, 4] * (bounds.foil_thickness[1] - bounds.foil_thickness[0]) + bounds.foil_thickness[0]

    # HV thickness
    scaled[:, 5] = samples[:, 5] * (bounds.hv_thickness[1] - bounds.hv_thickness[0]) + bounds.hv_thickness[0]

    # HV length (must be >= hv_thickness)
    hv_len_min = scaled[:, 5]  # At least hv_thickness
    hv_len_max = HV_LEN_MAXIMUM
    scaled[:, 6] = samples[:, 6] * (hv_len_max - hv_len_min) + hv_len_min

    return scaled


def evaluate_design(params: np.ndarray,
                   put_cooling_ducts: bool = True,
                   lvrate: float = LVRATE,
                   hvrate: float = HVRATE,
                   power: float = POWERRATING,
                   freq: float = FREQUENCY) -> Dict:
    """
    Evaluate a single transformer design using physics calculations.

    Args:
        params: Array of 7 parameters [core_dia, core_len, lv_turns, foil_h, foil_t, hv_t, hv_len]
        put_cooling_ducts: Whether to include cooling ducts
        lvrate, hvrate, power, freq: Transformer specifications

    Returns:
        Dictionary with calculated outputs or None if design is invalid
    """
    core_dia = params[0]
    core_len = params[1]
    lv_turns = int(params[2])
    foil_height = params[3]
    foil_thickness = params[4]
    hv_thickness = params[5]
    hv_length = params[6]

    # Check basic validity
    if core_len > core_dia:
        return None
    if hv_length < hv_thickness:
        return None
    if lv_turns < FOILTURNS_MINIMUM or lv_turns > FOILTURNS_MAXIMUM:
        return None

    try:
        # Calculate induction first (quick validity check)
        volts_per_turn = CalculateVoltsPerTurns(lvrate, lv_turns)
        induction = CalculateInduction(volts_per_turn, core_dia, core_len)

        # Filter out invalid induction range (typical valid range 0.8-1.95)
        if induction < 0.5 or induction > 2.5 or induction >= 1e17:
            return None

        # Initialize cooling duct counts
        lv_cd = 0
        hv_cd = 0

        if put_cooling_ducts:
            lv_cd, hv_cd, ll_zero, ll_lv_zero, ll_hv_zero = CalculateNumberOfCoolingDucts_WithLosses(
                lv_turns, foil_height, foil_thickness, hv_thickness, hv_length,
                core_dia, core_len, materialToBeUsedWire_Resistivity,
                power, hvrate, lvrate, False, False
            )
            if lv_cd == 0 and hv_cd == 0:
                ll, ll_lv, ll_hv = ll_zero, ll_lv_zero, ll_hv_zero
            else:
                ll, ll_hv, ll_lv = CalculateLoadLosses(
                    lv_turns, foil_thickness, core_dia, foil_height,
                    hv_thickness, hv_length, materialToBeUsedWire_Resistivity,
                    power, hvrate, lvrate, core_len, lv_cd, hv_cd, False
                )
        else:
            ll, ll_hv, ll_lv = CalculateLoadLosses(
                lv_turns, foil_thickness, core_dia, foil_height,
                hv_thickness, hv_length, materialToBeUsedWire_Resistivity,
                power, hvrate, lvrate, core_len, 0, 0, False
            )

        # Check for invalid values
        if ll >= 1e17 or ll < 0:
            return None

        # No-load losses
        nll = CalculateNoLoadLosses(
            lv_turns, foil_height, foil_thickness, hv_thickness, hv_length,
            core_dia, lvrate, core_len, lv_cd, hv_cd, False
        )
        if nll >= 1e17 or nll < 0:
            return None

        # Impedance calculations
        from mainRect import CalculateStrayDiameter
        stray_dia = CalculateStrayDiameter(
            lv_turns, foil_thickness, foil_height, hv_thickness, hv_length,
            core_dia, core_len, lv_cd, hv_cd, False
        )
        ux = CalculateUx(
            power, stray_dia, lv_turns, foil_thickness, foil_height,
            hv_thickness, hv_length, freq, lvrate, lv_cd, hv_cd, False
        )
        ur = CalculateUr(ll, power)
        ucc = CalculateImpedance(ux, ur)

        if ucc >= 1e17 or ucc < 0:
            return None

        # Price calculation
        price = CalculatePrice(
            lv_turns, foil_height, foil_thickness, hv_thickness, hv_length,
            core_dia, core_len, lv_cd, hv_cd, False, False
        )
        if price >= 1e17 or price < 0:
            return None

        # Core weight
        core_weight = CalculateCoreWeight(
            lv_turns, foil_height, foil_thickness, hv_thickness, hv_length,
            core_dia, core_len, lv_cd, hv_cd, False
        )

        # Calculate thermal gradient (optional, may fail for some designs)
        try:
            eff_duct = CalculateEfficencyOfCoolingDuct(foil_height)
            eff_main_gap = CalculateEfficencyOfMainGap(foil_height)
            heat_flux_lv = CalculateHeatFluxLV(ll_lv, foil_height, lv_turns, foil_thickness, core_dia, core_len)
            heat_flux_hv = CalculateHeatFluxHV(ll_hv, foil_height - 50, foil_height, lv_turns, foil_thickness, core_dia, hv_thickness, hv_length, core_len, False)

            gradient_lv = CalculateGradientHeatLV(lv_turns, heat_flux_lv, eff_main_gap, eff_duct, lv_cd)
            gradient_hv = CalculateGradientHeatHV(heat_flux_hv, eff_main_gap, eff_duct, hv_cd)

            if gradient_lv >= 1e17:
                gradient_lv = -1  # Mark as unknown
            if gradient_hv >= 1e17:
                gradient_hv = -1
        except:
            gradient_lv = -1
            gradient_hv = -1

        # Constraint violations
        nll_violation = max(0, nll - GUARANTEED_NO_LOAD_LOSS)
        ll_violation = max(0, ll - GUARANTEED_LOAD_LOSS)
        ucc_violation = max(0, abs(ucc - GUARANTEED_UCC) - UCC_TOLERANCE)

        # Validity check
        is_valid = (nll_violation == 0 and ll_violation == 0 and ucc_violation == 0)
        if gradient_lv > 0 and gradient_lv > MAX_GRADIENT:
            is_valid = False
        if gradient_hv > 0 and gradient_hv > MAX_GRADIENT:
            is_valid = False

        return {
            # Input parameters
            'core_diameter': core_dia,
            'core_length': core_len,
            'lv_turns': lv_turns,
            'foil_height': foil_height,
            'foil_thickness': foil_thickness,
            'hv_thickness': hv_thickness,
            'hv_length': hv_length,
            # Computed outputs
            'nll': nll,
            'll': ll,
            'ucc': ucc,
            'price': price,
            'induction': induction,
            'core_weight': core_weight,
            'gradient_lv': gradient_lv,
            'gradient_hv': gradient_hv,
            'cooling_ducts_lv': lv_cd,
            'cooling_ducts_hv': hv_cd,
            # Constraint info
            'nll_violation': nll_violation,
            'll_violation': ll_violation,
            'ucc_violation': ucc_violation,
            'is_valid': is_valid,
        }

    except Exception as e:
        return None


def generate_dataset(n_samples: int,
                    output_path: str,
                    put_cooling_ducts: bool = True,
                    show_progress: bool = True) -> Tuple[int, int]:
    """
    Generate full dataset and save to HDF5.

    Args:
        n_samples: Number of samples to generate
        output_path: Path to save HDF5 file
        put_cooling_ducts: Whether to include cooling ducts
        show_progress: Whether to print progress

    Returns:
        Tuple of (valid_samples, total_samples)
    """
    print(f"Generating {n_samples} samples...")
    start_time = time.time()

    # Generate samples
    bounds = ParameterBounds()
    raw_samples = latin_hypercube_sample(n_samples, n_dims=7)
    scaled_samples = scale_samples(raw_samples, bounds)

    # Evaluate all samples
    results = []
    valid_count = 0

    for i, params in enumerate(scaled_samples):
        result = evaluate_design(params, put_cooling_ducts=put_cooling_ducts)
        if result is not None:
            results.append(result)
            valid_count += 1

        if show_progress and (i + 1) % 1000 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (n_samples - i - 1) / rate
            print(f"  Progress: {i+1}/{n_samples} ({100*(i+1)/n_samples:.1f}%) | "
                  f"Valid: {valid_count} ({100*valid_count/(i+1):.1f}%) | "
                  f"Rate: {rate:.0f}/s | ETA: {eta:.0f}s")

    print(f"Generated {valid_count} valid samples out of {n_samples} ({100*valid_count/n_samples:.1f}%)")

    # Save to HDF5
    if results:
        save_to_hdf5(results, output_path)

    total_time = time.time() - start_time
    print(f"Total time: {total_time:.1f}s ({n_samples/total_time:.0f} samples/s)")

    return valid_count, n_samples


def save_to_hdf5(results: list, output_path: str):
    """Save results to HDF5 file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    n = len(results)

    with h5py.File(output_path, 'w') as f:
        # Input parameters
        f.create_dataset('core_diameter', data=np.array([r['core_diameter'] for r in results], dtype=np.float32))
        f.create_dataset('core_length', data=np.array([r['core_length'] for r in results], dtype=np.float32))
        f.create_dataset('lv_turns', data=np.array([r['lv_turns'] for r in results], dtype=np.int32))
        f.create_dataset('foil_height', data=np.array([r['foil_height'] for r in results], dtype=np.float32))
        f.create_dataset('foil_thickness', data=np.array([r['foil_thickness'] for r in results], dtype=np.float32))
        f.create_dataset('hv_thickness', data=np.array([r['hv_thickness'] for r in results], dtype=np.float32))
        f.create_dataset('hv_length', data=np.array([r['hv_length'] for r in results], dtype=np.float32))

        # Output targets
        f.create_dataset('nll', data=np.array([r['nll'] for r in results], dtype=np.float32))
        f.create_dataset('ll', data=np.array([r['ll'] for r in results], dtype=np.float32))
        f.create_dataset('ucc', data=np.array([r['ucc'] for r in results], dtype=np.float32))
        f.create_dataset('price', data=np.array([r['price'] for r in results], dtype=np.float32))
        f.create_dataset('induction', data=np.array([r['induction'] for r in results], dtype=np.float32))
        f.create_dataset('core_weight', data=np.array([r['core_weight'] for r in results], dtype=np.float32))
        f.create_dataset('gradient_lv', data=np.array([r['gradient_lv'] for r in results], dtype=np.float32))
        f.create_dataset('gradient_hv', data=np.array([r['gradient_hv'] for r in results], dtype=np.float32))

        # Auxiliary
        f.create_dataset('cooling_ducts_lv', data=np.array([r['cooling_ducts_lv'] for r in results], dtype=np.int32))
        f.create_dataset('cooling_ducts_hv', data=np.array([r['cooling_ducts_hv'] for r in results], dtype=np.int32))
        f.create_dataset('is_valid', data=np.array([r['is_valid'] for r in results], dtype=np.bool_))

        # Constraint violations
        f.create_dataset('nll_violation', data=np.array([r['nll_violation'] for r in results], dtype=np.float32))
        f.create_dataset('ll_violation', data=np.array([r['ll_violation'] for r in results], dtype=np.float32))
        f.create_dataset('ucc_violation', data=np.array([r['ucc_violation'] for r in results], dtype=np.float32))

        # Metadata
        f.attrs['n_samples'] = n
        f.attrs['generated_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
        f.attrs['power_rating'] = POWERRATING
        f.attrs['lv_rate'] = LVRATE
        f.attrs['hv_rate'] = HVRATE
        f.attrs['frequency'] = FREQUENCY
        f.attrs['guaranteed_nll'] = GUARANTEED_NO_LOAD_LOSS
        f.attrs['guaranteed_ll'] = GUARANTEED_LOAD_LOSS
        f.attrs['guaranteed_ucc'] = GUARANTEED_UCC

    print(f"Saved {n} samples to {output_path}")


def load_dataset(path: str) -> Dict[str, np.ndarray]:
    """Load dataset from HDF5 file."""
    data = {}
    with h5py.File(path, 'r') as f:
        for key in f.keys():
            data[key] = f[key][:]
        # Load metadata
        data['metadata'] = dict(f.attrs)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate transformer design training data')
    parser.add_argument('--n_samples', type=int, default=100000,
                       help='Number of samples to generate (default: 100000)')
    parser.add_argument('--output', type=str, default='ml/data/raw/transformer_data.h5',
                       help='Output HDF5 file path')
    parser.add_argument('--no_cooling_ducts', action='store_true',
                       help='Disable cooling duct calculations')

    args = parser.parse_args()

    # Ensure output directory exists
    output_path = os.path.join(os.path.dirname(__file__), '../../', args.output)
    output_path = os.path.normpath(output_path)

    generate_dataset(
        n_samples=args.n_samples,
        output_path=output_path,
        put_cooling_ducts=not args.no_cooling_ducts
    )
