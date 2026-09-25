"""
OD-Zoom — import/export of tabular data (CSV/XLSX) and text reports.
"""

from __future__ import annotations

import pandas as pd

from core import Project, Element, ODZoomError


def import_reference_table(path: str) -> Project:
    """Import reference data from CSV/XLSX.

    Expected format: first column — focal length f, remaining columns —
    thicknesses of movable elements (column headers are used as element names).
    """
    try:
        if path.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(path, sep=None, engine="python")
    except Exception as e:
        raise ODZoomError(f"Failed to read file: {e}")

    df = df.dropna(how="all")
    if df.shape[1] < 2:
        raise ODZoomError("The file must contain at least 2 columns: f and at least one element.")
    if df.shape[0] < 2:
        raise ODZoomError("The file must contain at least 2 data rows (configurations).")

    f_col = df.columns[0]
    project = Project()
    try:
        project.focal_lengths = df[f_col].astype(float).tolist()
        for col in df.columns[1:]:
            values = df[col].astype(float).tolist()
            project.elements.append(Element(name=str(col), values=values))
    except (ValueError, TypeError) as e:
        raise ODZoomError(f"The file contains non-numeric values: {e}")

    return project


def export_table(table: dict, path: str, element_order: list):
    """Export the calculated table (f + element values) to CSV/XLSX."""
    columns = ["f"] + list(element_order)
    data = {col: table[col] for col in columns}
    df = pd.DataFrame(data)
    try:
        if path.lower().endswith((".xlsx", ".xls")):
            df.to_excel(path, index=False)
        else:
            df.to_csv(path, index=False)
    except Exception as e:
        raise ODZoomError(f"Failed to save table: {e}")


def export_report(results, warnings, table, path: str, units: str = "mm"):
    """Text report: formulas, quality, and summary table."""
    lines = ["OD-Zoom — report on variable thickness calculation", "=" * 55, ""]

    if warnings:
        lines.append("WARNINGS:")
        for w in warnings:
            lines.append(f"  - {w}")
        lines.append("")

    for fit in results:
        lines.append(f"Element: {fit.element_name}")
        lines.append(f"  Method: {fit.method}")
        lines.append(f"  {fit.description}")
        if fit.max_residual < 1e-9:
            lines.append("  Exact interpolation — the curve passes through all reference points.")
        else:
            lines.append(f"  RMS deviation from reference points: {fit.rms_residual:.6g} {units}")
            lines.append(f"  Max deviation from reference points: {fit.max_residual:.6g} {units}")
        lines.append("")

    lines.append("Table of discrete values:")
    header = ["f"] + [fit.element_name for fit in results]
    lines.append("\t".join(header))
    n = len(table["f"])
    for i in range(n):
        row = [f"{table['f'][i]:.6g}"] + [f"{table[name][i]:.6g}" for name in header[1:]]
        lines.append("\t".join(row))

    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
    except Exception as e:
        raise ODZoomError(f"Failed to save report: {e}")