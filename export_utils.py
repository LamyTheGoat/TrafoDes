"""
Export utilities for transformer optimization results.

Supports CSV, Excel, JSON, and PDF export formats.
"""

import pandas as pd
import json
from datetime import datetime
from typing import List, Dict, Optional, Union
from pathlib import Path
from io import BytesIO, StringIO


def results_to_dataframe(results: List[Dict], include_specs: bool = True) -> pd.DataFrame:
    """
    Convert optimization results to pandas DataFrame.

    Args:
        results: List of result dictionaries from optimization
        include_specs: Whether to include input specifications if available

    Returns:
        DataFrame with all result fields
    """
    rows = []
    for i, r in enumerate(results):
        row = {
            'Name': r.get('name', f'Trafo-{i+1}'),
            # Input specs (if available)
            'Power (kVA)': r.get('power', r.get('spec', {}).get('power', '')),
            'HV Voltage (V)': r.get('hv_voltage', r.get('spec', {}).get('hv_voltage', '')),
            'LV Voltage (V)': r.get('lv_voltage', r.get('spec', {}).get('lv_voltage', '')),
            # Design parameters
            'Core Diameter (mm)': r.get('core_diameter', 0),
            'Core Length (mm)': r.get('core_length', 0),
            'LV Turns': r.get('lv_turns', 0),
            'LV Height (mm)': r.get('lv_height', 0),
            'LV Thickness (mm)': r.get('lv_thickness', 0),
            'HV Thickness (mm)': r.get('hv_thickness', 0),
            'HV Length (mm)': r.get('hv_length', 0),
            # Performance
            'No-Load Loss (W)': r.get('no_load_loss', 0),
            'Load Loss (W)': r.get('load_loss', 0),
            'Impedance (%)': r.get('impedance', 0),
            # Weights
            'Core Weight (kg)': r.get('core_weight', 0),
            'LV Weight (kg)': r.get('lv_weight', 0),
            'HV Weight (kg)': r.get('hv_weight', 0),
            'Total Weight (kg)': r.get('core_weight', 0) + r.get('lv_weight', 0) + r.get('hv_weight', 0),
            # Prices
            'Core Price ($)': r.get('core_price', 0),
            'LV Price ($)': r.get('lv_price', 0),
            'HV Price ($)': r.get('hv_price', 0),
            'Total Price ($)': r.get('total_price', 0),
            # Cooling
            'LV Cooling Ducts': r.get('n_ducts_lv', 0),
            'HV Cooling Ducts': r.get('n_ducts_hv', 0),
            # Timing
            'Optimization Time (s)': r.get('time', 0),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def export_csv(results: List[Dict], filepath: Optional[str] = None,
               return_string: bool = False) -> Optional[str]:
    """
    Export results to CSV format.

    Args:
        results: List of result dictionaries
        filepath: Path to save CSV file (optional)
        return_string: If True, return CSV as string instead of saving

    Returns:
        CSV string if return_string=True, else None
    """
    df = results_to_dataframe(results)

    if return_string:
        return df.to_csv(index=False)

    if filepath:
        df.to_csv(filepath, index=False)
        print(f"Exported {len(results)} results to {filepath}")

    return None


def export_excel(results: List[Dict], filepath: Optional[str] = None,
                 return_bytes: bool = False) -> Optional[bytes]:
    """
    Export results to Excel with multiple sheets.

    Sheets:
        - Summary: All transformers with key metrics
        - Design: Design parameters only
        - Performance: Loss and impedance data
        - Cost Breakdown: Detailed cost analysis

    Args:
        results: List of result dictionaries
        filepath: Path to save Excel file (optional)
        return_bytes: If True, return Excel as bytes instead of saving

    Returns:
        Excel bytes if return_bytes=True, else None
    """
    df = results_to_dataframe(results)

    # Prepare different views
    summary_cols = ['Name', 'Power (kVA)', 'Total Price ($)', 'No-Load Loss (W)',
                    'Load Loss (W)', 'Impedance (%)', 'Total Weight (kg)']
    summary_df = df[[c for c in summary_cols if c in df.columns]]

    design_cols = ['Name', 'Core Diameter (mm)', 'Core Length (mm)', 'LV Turns',
                   'LV Height (mm)', 'LV Thickness (mm)', 'HV Thickness (mm)',
                   'HV Length (mm)', 'LV Cooling Ducts', 'HV Cooling Ducts']
    design_df = df[[c for c in design_cols if c in df.columns]]

    perf_cols = ['Name', 'Power (kVA)', 'No-Load Loss (W)', 'Load Loss (W)',
                 'Impedance (%)', 'Optimization Time (s)']
    perf_df = df[[c for c in perf_cols if c in df.columns]]

    cost_cols = ['Name', 'Core Weight (kg)', 'Core Price ($)', 'LV Weight (kg)',
                 'LV Price ($)', 'HV Weight (kg)', 'HV Price ($)', 'Total Price ($)']
    cost_df = df[[c for c in cost_cols if c in df.columns]]

    output = BytesIO() if return_bytes else filepath

    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        design_df.to_excel(writer, sheet_name='Design', index=False)
        perf_df.to_excel(writer, sheet_name='Performance', index=False)
        cost_df.to_excel(writer, sheet_name='Cost Breakdown', index=False)
        df.to_excel(writer, sheet_name='All Data', index=False)

    if return_bytes:
        output.seek(0)
        return output.read()

    if filepath:
        print(f"Exported {len(results)} results to {filepath}")

    return None


def export_json(results: List[Dict], filepath: Optional[str] = None,
                return_string: bool = False, indent: int = 2) -> Optional[str]:
    """
    Export results to JSON format.

    Args:
        results: List of result dictionaries
        filepath: Path to save JSON file (optional)
        return_string: If True, return JSON as string instead of saving
        indent: JSON indentation level

    Returns:
        JSON string if return_string=True, else None
    """
    export_data = {
        'export_date': datetime.now().isoformat(),
        'transformer_count': len(results),
        'transformers': []
    }

    for i, r in enumerate(results):
        trafo = {
            'name': r.get('name', f'Trafo-{i+1}'),
            'specifications': {
                'power_kva': r.get('power', r.get('spec', {}).get('power')),
                'hv_voltage': r.get('hv_voltage', r.get('spec', {}).get('hv_voltage')),
                'lv_voltage': r.get('lv_voltage', r.get('spec', {}).get('lv_voltage')),
            },
            'design': {
                'core_diameter_mm': r.get('core_diameter'),
                'core_length_mm': r.get('core_length'),
                'lv_turns': r.get('lv_turns'),
                'lv_height_mm': r.get('lv_height'),
                'lv_thickness_mm': r.get('lv_thickness'),
                'hv_thickness_mm': r.get('hv_thickness'),
                'hv_length_mm': r.get('hv_length'),
                'lv_cooling_ducts': r.get('n_ducts_lv'),
                'hv_cooling_ducts': r.get('n_ducts_hv'),
            },
            'performance': {
                'no_load_loss_w': r.get('no_load_loss'),
                'load_loss_w': r.get('load_loss'),
                'impedance_pct': r.get('impedance'),
            },
            'weights': {
                'core_kg': r.get('core_weight'),
                'lv_kg': r.get('lv_weight'),
                'hv_kg': r.get('hv_weight'),
                'total_kg': (r.get('core_weight', 0) or 0) +
                           (r.get('lv_weight', 0) or 0) +
                           (r.get('hv_weight', 0) or 0),
            },
            'cost': {
                'core_usd': r.get('core_price'),
                'lv_usd': r.get('lv_price'),
                'hv_usd': r.get('hv_price'),
                'total_usd': r.get('total_price'),
            },
            'optimization_time_s': r.get('time'),
        }
        export_data['transformers'].append(trafo)

    json_str = json.dumps(export_data, indent=indent, default=str)

    if return_string:
        return json_str

    if filepath:
        with open(filepath, 'w') as f:
            f.write(json_str)
        print(f"Exported {len(results)} results to {filepath}")

    return None


def export_pdf(results: List[Dict], filepath: str,
               company_name: str = "TrafoDes",
               return_bytes: bool = False) -> Optional[bytes]:
    """
    Export results to PDF report.

    Args:
        results: List of result dictionaries
        filepath: Path to save PDF file
        company_name: Company name for header
        return_bytes: If True, return PDF as bytes instead of saving

    Returns:
        PDF bytes if return_bytes=True, else None
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm
    except ImportError:
        print("reportlab not installed. Install with: pip install reportlab")
        return None

    output = BytesIO() if return_bytes else filepath
    doc = SimpleDocTemplate(output, pagesize=landscape(A4),
                           leftMargin=15*mm, rightMargin=15*mm,
                           topMargin=15*mm, bottomMargin=15*mm)
    elements = []
    styles = getSampleStyleSheet()

    # Title
    title_style = ParagraphStyle('Title', parent=styles['Heading1'],
                                  fontSize=18, spaceAfter=12)
    elements.append(Paragraph(f"{company_name} - Transformer Optimization Report", title_style))
    elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                             styles['Normal']))
    elements.append(Spacer(1, 10*mm))

    # Summary table
    df = results_to_dataframe(results)

    # Select key columns for PDF (limited width)
    key_cols = ['Name', 'Power (kVA)', 'Total Price ($)', 'No-Load Loss (W)',
                'Load Loss (W)', 'Impedance (%)', 'Core Diameter (mm)']
    pdf_df = df[[c for c in key_cols if c in df.columns]]

    # Convert to table data
    table_data = [pdf_df.columns.tolist()]
    for _, row in pdf_df.iterrows():
        formatted_row = []
        for val in row:
            if isinstance(val, float):
                formatted_row.append(f"{val:.2f}")
            else:
                formatted_row.append(str(val))
        table_data.append(formatted_row)

    # Create table
    table = Table(table_data, repeatRows=1)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.Color(0.2, 0.3, 0.4)),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.Color(0.95, 0.95, 0.95)),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.Color(0.9, 0.9, 0.9)]),
    ]))
    elements.append(table)

    # Build PDF
    doc.build(elements)

    if return_bytes:
        output.seek(0)
        return output.read()

    print(f"Exported {len(results)} results to {filepath}")
    return None


# Convenience function for quick exports
def export_all(results: List[Dict], base_path: str, formats: List[str] = None):
    """
    Export results to multiple formats at once.

    Args:
        results: List of result dictionaries
        base_path: Base path without extension (e.g., "output/transformers")
        formats: List of formats to export. Default: ['csv', 'xlsx', 'json']
    """
    if formats is None:
        formats = ['csv', 'xlsx', 'json']

    base = Path(base_path)
    base.parent.mkdir(parents=True, exist_ok=True)

    exported = []

    if 'csv' in formats:
        export_csv(results, f"{base_path}.csv")
        exported.append('csv')

    if 'xlsx' in formats or 'excel' in formats:
        try:
            export_excel(results, f"{base_path}.xlsx")
            exported.append('xlsx')
        except ImportError:
            print("openpyxl not installed. Skipping Excel export.")

    if 'json' in formats:
        export_json(results, f"{base_path}.json")
        exported.append('json')

    if 'pdf' in formats:
        try:
            export_pdf(results, f"{base_path}.pdf")
            exported.append('pdf')
        except ImportError:
            print("reportlab not installed. Skipping PDF export.")

    print(f"Exported to: {', '.join(exported)}")


if __name__ == "__main__":
    # Test with sample data
    sample_results = [
        {
            'name': 'Test-Trafo-1',
            'power': 400,
            'hv_voltage': 33000,
            'lv_voltage': 400,
            'core_diameter': 220,
            'core_length': 140,
            'lv_turns': 35,
            'lv_height': 580,
            'lv_thickness': 1.5,
            'hv_thickness': 2.2,
            'hv_length': 7.5,
            'no_load_loss': 520,
            'load_loss': 4200,
            'impedance': 5.95,
            'core_weight': 185,
            'lv_weight': 42,
            'hv_weight': 28,
            'core_price': 666,
            'lv_price': 485,
            'hv_price': 309,
            'total_price': 1460,
            'n_ducts_lv': 0,
            'n_ducts_hv': 0,
            'time': 2.5
        }
    ]

    print("Testing export functions...")
    print("\nCSV output:")
    print(export_csv(sample_results, return_string=True)[:500])
    print("\nJSON output:")
    print(export_json(sample_results, return_string=True)[:500])
