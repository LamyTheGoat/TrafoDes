#!/usr/bin/env python3
"""
CLI wrapper for transformer optimization.

Accepts JSON input and outputs JSON result. Automatically selects the best
optimization method for the device and finds the optimal design by testing
all combinations of obround/non-obround and circular/rectangular wire.

Usage:
    echo '{"power_kva": 400, ...}' | python trafo_cli.py
    python trafo_cli.py input.json
    python trafo_cli.py input.json -o output.json
    python trafo_cli.py --help

Input JSON format:
{
    "power_kva": 400,
    "hv_voltage": 30000,
    "lv_voltage": 400,
    "no_load_loss": 610,
    "load_loss": 4600,
    "impedance": 4.5,
    "winding_material": "CU",
    "prices": {
        "copper_foil": 11.55,
        "copper_wire": 11.05,
        "core": 3.6
    }
}

Output JSON format:
{
    "design": {
        "core_diameter": 180,
        "core_length": 45,
        "lv_turns": 22,
        "lv_height": 450,
        "lv_thickness": 1.2,
        "hv_thickness": 2.1,
        "hv_length": 8.5
    },
    "performance": {
        "no_load_loss": 608,
        "load_loss": 4580,
        "impedance": 4.48
    },
    "cost": {
        "core_weight": 320,
        "foil_weight": 85,
        "wire_weight": 120,
        "total_price": 4500
    },
    "meta": {
        "method": "hybrid",
        "obround": true,
        "circular_wire": false,
        "time_seconds": 12.5
    }
}
"""

import argparse
import json
import sys
import os
import io

# Suppress all print output during import
_original_stdout = sys.stdout
_original_stderr = sys.stderr
sys.stdout = io.StringIO()
sys.stderr = io.StringIO()

try:
    import mainRect
    import tank_oil
finally:
    sys.stdout = _original_stdout
    sys.stderr = _original_stderr


def detect_best_method():
    """Detect the best optimization method for the current device."""
    # Check GPU availability (these are set at mainRect import time)
    if hasattr(mainRect, 'MPS_AVAILABLE') and mainRect.MPS_AVAILABLE:
        return 'hybrid'  # MPS hybrid is best for Apple Silicon
    if hasattr(mainRect, 'TORCH_CUDA_AVAILABLE') and mainRect.TORCH_CUDA_AVAILABLE:
        return 'cuda_hybrid'  # CUDA hybrid for NVIDIA GPUs
    if hasattr(mainRect, 'MLX_AVAILABLE') and mainRect.MLX_AVAILABLE:
        return 'mlx'  # MLX for Apple Silicon without PyTorch
    # CPU fallback - multi-seed DE is robust for CPU
    return 'multi_de'


def configure_mainrect(input_data):
    """Configure mainRect global parameters from input JSON."""
    # Power and voltage settings
    power_kva = input_data.get('power_kva', 1000)
    hv_voltage = input_data.get('hv_voltage', 20000)
    lv_voltage = input_data.get('lv_voltage', 400)

    # Connection types (default: HV=Delta, LV=Star)
    hv_connection = input_data.get('hv_connection', 'D')
    lv_connection = input_data.get('lv_connection', 'Y')

    # Map connection type to multiplier
    connection_map = {'D': 1.0, 'Y': 1.7320508075688772}  # sqrt(3)
    hv_conn_factor = connection_map.get(hv_connection, 1.0)
    lv_conn_factor = connection_map.get(lv_connection, 1.7320508075688772)

    # Set power and voltage
    mainRect.POWERRATING = power_kva
    mainRect.HVRATE = hv_voltage * hv_conn_factor
    mainRect.LVRATE = lv_voltage * lv_conn_factor
    mainRect.FREQUENCY = input_data.get('frequency', 50)

    # Guaranteed loss limits
    mainRect.GUARANTEED_NO_LOAD_LOSS = input_data.get('no_load_loss', 900) * 0.98
    mainRect.GUARANTEED_LOAD_LOSS = input_data.get('load_loss', 7500) * 0.98
    mainRect.GUARANTEED_UCC = input_data.get('impedance', 6)

    # Material selection
    winding_material = input_data.get('winding_material', 'CU').upper()

    if winding_material == 'CU':
        # Copper
        mainRect.materialToBeUsedFoil_Density = mainRect.CuDensity
        mainRect.materialToBeUsedWire_Density = mainRect.CuDensity
        mainRect.materialToBeUsedFoil_Resistivity = 0.021
        mainRect.materialToBeUsedWire_Resistivity = 0.021
    elif winding_material == 'AL':
        # Aluminum
        mainRect.materialToBeUsedFoil_Density = mainRect.AlDensity
        mainRect.materialToBeUsedWire_Density = mainRect.AlDensity
        mainRect.materialToBeUsedFoil_Resistivity = mainRect.AlRestivity
        mainRect.materialToBeUsedWire_Resistivity = mainRect.AlRestivity

    # Prices
    prices = input_data.get('prices', {})
    mainRect.materialToBeUsedFoil_Price = prices.get('copper_foil', 11.55)
    mainRect.materialToBeUsedWire_Price = prices.get('copper_wire', 11.05)
    mainRect.materialCore_Price = prices.get('core', 3.6)
    mainRect.CorePricePerKg = prices.get('core', 3.6)


class SuppressOutput:
    """Context manager to suppress stdout/stderr at file descriptor level."""

    def __init__(self):
        self.null_fd = None
        self.saved_stdout_fd = None
        self.saved_stderr_fd = None
        self.saved_stdout = None
        self.saved_stderr = None

    def __enter__(self):
        # Save Python-level streams
        self.saved_stdout = sys.stdout
        self.saved_stderr = sys.stderr

        # Redirect Python streams
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()

        # Save file descriptors
        self.saved_stdout_fd = os.dup(1)
        self.saved_stderr_fd = os.dup(2)

        # Open /dev/null and redirect
        self.null_fd = os.open(os.devnull, os.O_RDWR)
        os.dup2(self.null_fd, 1)
        os.dup2(self.null_fd, 2)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Flush before restoring
        sys.stdout.flush() if hasattr(sys.stdout, 'flush') else None
        sys.stderr.flush() if hasattr(sys.stderr, 'flush') else None

        # Restore file descriptors
        os.dup2(self.saved_stdout_fd, 1)
        os.dup2(self.saved_stderr_fd, 2)

        # Close saved descriptors
        os.close(self.saved_stdout_fd)
        os.close(self.saved_stderr_fd)
        os.close(self.null_fd)

        # Restore Python streams
        sys.stdout = self.saved_stdout
        sys.stderr = self.saved_stderr
        return False


def run_optimization(method, obround, circular_wire, tolerance=25, search_depth='normal', quiet=True):
    """Run a single optimization with specified parameters."""
    # Set wire type
    mainRect.HV_WIRE_CIRCULAR = circular_wire

    if quiet:
        with SuppressOutput():
            result = mainRect.StartFast(
                tolerance=tolerance,
                obround=obround,
                put_cooling_ducts=True,
                method=method,
                print_result=False,
                search_depth=search_depth
            )
    else:
        result = mainRect.StartFast(
            tolerance=tolerance,
            obround=obround,
            put_cooling_ducts=True,
            method=method,
            print_result=False,
            search_depth=search_depth
        )

    return result


def calculate_tank_oil_data(result, input_data):
    """
    Calculate tank and oil data from optimization result.

    Args:
        result: Optimization result dict
        input_data: Original input JSON with tank_oil config

    Returns:
        dict: Tank and oil calculation results, or None if disabled
    """
    tank_oil_config = input_data.get('tank_oil', {})

    # Check if tank/oil calculation is enabled
    if not tank_oil_config.get('include', False):
        return None

    # Get winding material density for oil volume calculation
    winding_material = input_data.get('winding_material', 'CU').upper()
    if winding_material == 'AL':
        lv_density = tank_oil.ALUMINUM_DENSITY
        hv_density = tank_oil.ALUMINUM_DENSITY
    else:
        lv_density = tank_oil.COPPER_DENSITY
        hv_density = tank_oil.COPPER_DENSITY

    # Tank clearances
    clearance_x = tank_oil_config.get('clearance_x', 40.0)
    clearance_y = tank_oil_config.get('clearance_y', 40.0)
    clearance_z = tank_oil_config.get('clearance_z', 40.0)
    tank_wall_thickness = tank_oil_config.get('tank_wall_thickness', 3.0)
    steel_price = tank_oil_config.get('steel_price', 2.50)

    # Finwall config
    finwall_config = tank_oil_config.get('finwall', {})
    enable_finwall = finwall_config.get('enable', True)
    finwall_surfaces = finwall_config.get('surfaces', ['front', 'back', 'left', 'right'])
    finwall_auto = finwall_config.get('auto_optimize', True)
    target_temp_rise = finwall_config.get('target_temp_rise', 65.0)
    fin_depth_manual = finwall_config.get('manual_depth', 80.0)
    finwall_thickness = finwall_config.get('thickness', 1.0)

    # Oil config
    oil_config = tank_oil_config.get('oil', {})
    oil_type = oil_config.get('type', 'mineral')
    oil_fill_level = oil_config.get('fill_level', 85.0)
    oil_price_per_liter = oil_config.get('price_per_liter', None)

    # Call tank_oil calculation
    tank_oil_result = tank_oil.calculate_tank_and_oil(
        # From optimization result
        core_diameter=result['core_diameter'],
        core_length=result['core_length'],
        lv_turns=result['lv_turns'],
        lv_height=result['lv_height'],
        lv_thickness=result['lv_thickness'],
        hv_thickness=result['hv_thickness'],
        hv_length=result['hv_length'],
        no_load_loss=result['no_load_loss'],
        load_loss=result['load_loss'],
        # Weights from optimization
        core_weight=result['core_weight'],
        lv_weight=result['lv_weight'],
        hv_weight=result['hv_weight'],
        n_ducts_lv=result.get('n_ducts_lv', 0),
        n_ducts_hv=result.get('n_ducts_hv', 0),
        circular_hv=result.get('_circular_wire', False),
        # Tank configuration
        clearance_x=clearance_x,
        clearance_y=clearance_y,
        clearance_z=clearance_z,
        tank_wall_thickness=tank_wall_thickness,
        steel_price_per_kg=steel_price,
        # Finwall configuration
        enable_finwall=enable_finwall,
        finwall_surfaces=finwall_surfaces,
        finwall_auto_optimize=finwall_auto,
        target_temp_rise=target_temp_rise,
        fin_depth_manual=fin_depth_manual,
        finwall_thickness=finwall_thickness,
        # Oil configuration
        oil_type=oil_type,
        oil_price_per_liter=oil_price_per_liter,
        oil_fill_level=oil_fill_level,
        # Material densities
        lv_density=lv_density,
        hv_density=hv_density
    )

    return tank_oil_result


def find_best_design(input_data, method=None, search_depth='normal', verbose=False):
    """
    Find the best transformer design by testing all combinations.

    Tests: obround vs non-obround, circular vs rectangular wire
    Returns the design with the lowest total price.
    """
    # Configure mainRect with input parameters
    configure_mainrect(input_data)

    # Auto-detect method if not specified
    if method is None:
        method = detect_best_method()

    if verbose:
        print(f"Using optimization method: {method}", file=sys.stderr)

    # Test all 4 combinations
    combinations = [
        {'obround': True, 'circular': False, 'name': 'obround+rect_wire'},
        {'obround': True, 'circular': True, 'name': 'obround+circ_wire'},
        {'obround': False, 'circular': False, 'name': 'flat+rect_wire'},
        {'obround': False, 'circular': True, 'name': 'flat+circ_wire'},
    ]

    best_result = None
    best_config = None
    best_price = float('inf')
    total_time = 0

    for combo in combinations:
        if verbose:
            print(f"Testing {combo['name']}...", file=sys.stderr, flush=True)

        try:
            # Always suppress internal output, show our own progress instead
            result = run_optimization(
                method=method,
                obround=combo['obround'],
                circular_wire=combo['circular'],
                tolerance=25,
                search_depth=search_depth,
                quiet=True
            )

            if result is not None:
                price = result.get('total_price', result.get('price', float('inf')))
                time_taken = result.get('time', 0)
                total_time += time_taken

                if verbose:
                    print(f"  -> Price: {price:.2f}, Time: {time_taken:.1f}s", file=sys.stderr, flush=True)

                if price < best_price:
                    best_price = price
                    best_result = result
                    best_config = combo
        except Exception as e:
            if verbose:
                print(f"  -> Failed: {e}", file=sys.stderr, flush=True)
            continue

    if best_result is None:
        return None

    if verbose:
        print(f"Best: {best_config['name']} with price {best_price:.2f}", file=sys.stderr, flush=True)

    # Add metadata
    best_result['_method'] = method
    best_result['_obround'] = best_config['obround']
    best_result['_circular_wire'] = best_config['circular']
    best_result['_total_time'] = total_time

    return best_result


def format_output(result, tank_oil_data=None):
    """Format optimization result to output JSON structure."""
    if result is None:
        return {
            'error': 'No valid design found',
            'design': None,
            'performance': None,
            'cost': None
        }

    active_part_price = round(result.get('total_price', result.get('price', 0)), 2)

    output = {
        'design': {
            'core_diameter': round(result.get('core_diameter', 0), 1),
            'core_length': round(result.get('core_length', 0), 1),
            'lv_turns': int(result.get('lv_turns', 0)),
            'lv_height': round(result.get('lv_height', 0), 1),
            'lv_thickness': round(result.get('lv_thickness', 0), 2),
            'hv_thickness': round(result.get('hv_thickness', 0), 2),
            'hv_length': round(result.get('hv_length', 0), 2)
        },
        'performance': {
            'no_load_loss': round(result.get('no_load_loss', 0), 1),
            'load_loss': round(result.get('load_loss', 0), 1),
            'impedance': round(result.get('impedance', 0), 2)
        },
        'cost': {
            'core_weight': round(result.get('core_weight', 0), 1),
            'foil_weight': round(result.get('lv_weight', 0), 1),
            'wire_weight': round(result.get('hv_weight', 0), 1),
            'active_part_price': active_part_price
        },
        'meta': {
            'method': result.get('_method', 'unknown'),
            'obround': result.get('_obround', True),
            'circular_wire': result.get('_circular_wire', False),
            'time_seconds': round(result.get('_total_time', result.get('time', 0)), 2),
            'n_ducts_lv': result.get('n_ducts_lv', 0),
            'n_ducts_hv': result.get('n_ducts_hv', 0)
        }
    }

    # Add tank_oil section if calculated
    tank_oil_price = 0.0
    if tank_oil_data:
        tank_oil_price = round(tank_oil_data.get('total_tank_oil_price', 0), 2)
        output['tank_oil'] = {
            'tank': {
                'width': round(tank_oil_data.get('tank_width', 0), 1),
                'depth': round(tank_oil_data.get('tank_depth', 0), 1),
                'height': round(tank_oil_data.get('tank_height', 0), 1),
                'shell_weight': round(tank_oil_data.get('tank_shell_weight', 0), 1),
                'finwall_weight': round(tank_oil_data.get('finwall_weight', 0), 1),
                'total_weight': round(tank_oil_data.get('total_tank_weight', 0), 1),
                'price': round(tank_oil_data.get('tank_price', 0), 2)
            },
            'finwall': {
                'enabled': tank_oil_data.get('finwall_enabled', False),
                'surfaces': tank_oil_data.get('finwall_surfaces', []),
                'fin_depth': round(tank_oil_data.get('fin_depth', 0), 1),
                'auto_optimized': tank_oil_data.get('finwall_auto_optimized', False)
            },
            'oil': {
                'type': tank_oil_data.get('oil_type', 'mineral'),
                'volume_liters': round(tank_oil_data.get('oil_volume_liters', 0), 1),
                'weight': round(tank_oil_data.get('oil_weight', 0), 1),
                'price': round(tank_oil_data.get('oil_price', 0), 2)
            },
            'total_price': tank_oil_price
        }

    # Grand total
    output['grand_total'] = round(active_part_price + tank_oil_price, 2)

    return output


def main():
    parser = argparse.ArgumentParser(
        description='Transformer optimization CLI. Accepts JSON input and outputs optimized design.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  echo '{"power_kva": 400, "hv_voltage": 30000, "lv_voltage": 400}' | python trafo_cli.py
  python trafo_cli.py input.json
  python trafo_cli.py input.json -o output.json
  python trafo_cli.py input.json --method hybrid --depth thorough
        '''
    )

    parser.add_argument(
        'input_file',
        nargs='?',
        help='Input JSON file (reads from stdin if not specified)'
    )

    parser.add_argument(
        '-o', '--output',
        help='Output JSON file (writes to stdout if not specified)'
    )

    parser.add_argument(
        '-m', '--method',
        choices=['auto', 'hybrid', 'cuda_hybrid', 'mps', 'mlx', 'de', 'multi_de', 'smart', 'parallel'],
        default='auto',
        help='Optimization method (default: auto-detect best for device)'
    )

    parser.add_argument(
        '-d', '--depth',
        choices=['fast', 'normal', 'thorough', 'exhaustive'],
        default='normal',
        help='Search depth for hybrid methods (default: normal)'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Print progress to stderr'
    )

    parser.add_argument(
        '--pretty',
        action='store_true',
        help='Pretty-print JSON output'
    )

    args = parser.parse_args()

    # Read input JSON
    try:
        if args.input_file:
            with open(args.input_file, 'r') as f:
                input_data = json.load(f)
        else:
            # Read from stdin
            input_data = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        print(json.dumps({'error': f'Invalid JSON input: {e}'}), file=sys.stdout)
        sys.exit(1)
    except FileNotFoundError:
        print(json.dumps({'error': f'Input file not found: {args.input_file}'}), file=sys.stdout)
        sys.exit(1)

    # Determine method
    method = None if args.method == 'auto' else args.method

    # Run optimization
    if args.verbose:
        print("Starting transformer optimization...", file=sys.stderr)

    result = find_best_design(
        input_data,
        method=method,
        search_depth=args.depth,
        verbose=args.verbose
    )

    # Calculate tank and oil if enabled
    tank_oil_data = None
    if result is not None:
        tank_oil_data = calculate_tank_oil_data(result, input_data)
        if tank_oil_data and args.verbose:
            print(f"Tank+Oil: ${tank_oil_data['total_tank_oil_price']:.2f}", file=sys.stderr, flush=True)

    # Format output
    output = format_output(result, tank_oil_data)

    # Write output
    indent = 2 if args.pretty else None
    output_json = json.dumps(output, indent=indent)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(output_json)
            f.write('\n')
        if args.verbose:
            print(f"Output written to {args.output}", file=sys.stderr)
    else:
        print(output_json)


if __name__ == '__main__':
    main()
