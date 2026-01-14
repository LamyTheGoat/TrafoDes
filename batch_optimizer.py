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
    progress_callback: Optional[Callable[[int, int, str, float], None]] = None,
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
        progress_callback: Optional function(completed, total, current_name, elapsed)

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

            try:
                # Set module values for this spec
                mainRect.POWERRATING = spec.power
                mainRect.HVRATE = spec.hv_voltage
                mainRect.LVRATE = spec.lv_voltage
                mainRect.GUARANTEED_NO_LOAD_LOSS = spec.nll_limit * 0.98
                mainRect.GUARANTEED_LOAD_LOSS = spec.ll_limit * 0.98
                mainRect.GUARANTEED_UCC = spec.ucc_target

                # Run optimization
                result = mainRect.StartFast(
                    tolerance=tolerance,
                    obround=spec.obround,
                    put_cooling_ducts=spec.put_cooling_ducts,
                    method=method,
                    search_depth=depth,
                    print_result=False
                )

                if result is None:
                    return BatchResult(
                        spec=spec,
                        result={},
                        elapsed_time=time.time() - start,
                        success=False,
                        error_message="No valid design found"
                    )

                # Add spec info to result
                result['power'] = spec.power
                result['hv_voltage'] = spec.hv_voltage
                result['lv_voltage'] = spec.lv_voltage
                result['name'] = spec.name

                return BatchResult(
                    spec=spec,
                    result=result,
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
