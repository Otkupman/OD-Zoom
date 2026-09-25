"""
OD-Zoom — computational core.

The module is responsible for:
 - storing project data (reference configurations, movable elements);
 - calculating (polynomial, splines) intermediate thicknesses;
 - generating a grid of focal lengths;
 - evaluating approximation quality.

The module does not depend on Tkinter and can be used / tested separately
from the graphical interface.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator, Akima1DInterpolator


METHODS = {
    "poly": "Polynomial (least squares)",
    "linear": "Piecewise linear",
    "cubic": "Cubic spline",
    "pchip": "PCHIP (monotonic)",
    "akima": "Akima spline",
}

GRID_MODES = {
    "uniform_f": "Uniform in focal length f",
    "uniform_logf": "Uniform in lg(f)",
    "custom": "Custom list",
}


class ODZoomError(Exception):
    """Base exception for the OD-Zoom application."""


# --------------------------------------------------------------------------- #
#  Project data model
# --------------------------------------------------------------------------- #

@dataclass
class ElementSettings:
    """Settings for a single movable element."""
    method: str = "poly"
    degree: int = 3           # used only when method == "poly"
    use_global: bool = True   # if True, global project settings are used


@dataclass
class Element:
    """A single movable element (variable thickness / air gap)."""
    name: str
    values: list
    settings: ElementSettings = field(default_factory=ElementSettings)


@dataclass
class GridSettings:
    mode: str = "uniform_f"
    n_points: int = 21
    custom_values: list = field(default_factory=list)
    f_min: Optional[float] = None
    f_max: Optional[float] = None


@dataclass
class FitResult:
    element_name: str
    method: str
    description: str
    coeffs: object
    rms_residual: float
    max_residual: float
    evaluate: Callable[[np.ndarray], np.ndarray]


class Project:
    """OD-Zoom project: a set of reference configurations + movable elements."""

    def __init__(self):
        self.focal_lengths: list = []
        self.elements: list = []
        self.global_method: str = "poly"
        self.global_degree: int = 3
        self.grid = GridSettings()
        self.units_length = "mm"
        self.notes = ""

    # ---------- auxiliary ----------
    def n_configs(self) -> int:
        return len(self.focal_lengths)

    def n_elements(self) -> int:
        return len(self.elements)

    def validate(self):
        if self.n_elements() == 0:
            raise ODZoomError("Add at least one movable element.")
        if self.n_configs() < 2:
            raise ODZoomError("At least 2 reference configurations are required.")
        f = self.focal_lengths
        if len(set(f)) != len(f):
            raise ODZoomError("Focal length values must not be repeated.")
        if sorted(f) != list(f) and sorted(f, reverse=True) != list(f):
            raise ODZoomError(
                "Focal lengths of reference configurations must be "
                "monotonic (increasing or decreasing)."
            )
        if any(v <= 0 for v in f):
            raise ODZoomError("Focal length must be a positive number.")
        for el in self.elements:
            if len(el.values) != self.n_configs():
                raise ODZoomError(
                    f"Element '{el.name}': the number of values does not match "
                    f"the number of configurations."
                )

    def add_element(self, name: Optional[str] = None):
        if name is None:
            name = f"d{len(self.elements) + 1}"
        self.elements.append(Element(name=name, values=[0.0] * self.n_configs()))

    def remove_element(self, index: int):
        del self.elements[index]

    def add_config(self, f_value: float = 0.0):
        self.focal_lengths.append(f_value)
        for el in self.elements:
            el.values.append(0.0)

    def remove_config(self, index: int):
        del self.focal_lengths[index]
        for el in self.elements:
            del el.values[index]

    def set_n_elements(self, n: int):
        cur = self.n_elements()
        if n > cur:
            for i in range(cur, n):
                self.add_element(f"d{i + 1}")
        elif n < cur:
            self.elements = self.elements[:n]

    # ---------- serialization ----------
    def to_dict(self) -> dict:
        return {
            "focal_lengths": self.focal_lengths,
            "units_length": self.units_length,
            "notes": self.notes,
            "global_method": self.global_method,
            "global_degree": self.global_degree,
            "grid": {
                "mode": self.grid.mode,
                "n_points": self.grid.n_points,
                "custom_values": self.grid.custom_values,
                "f_min": self.grid.f_min,
                "f_max": self.grid.f_max,
            },
            "elements": [
                {
                    "name": el.name,
                    "values": el.values,
                    "settings": {
                        "method": el.settings.method,
                        "degree": el.settings.degree,
                        "use_global": el.settings.use_global,
                    },
                }
                for el in self.elements
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Project":
        p = cls()
        p.focal_lengths = list(d.get("focal_lengths", []))
        p.units_length = d.get("units_length", "mm")
        p.notes = d.get("notes", "")
        p.global_method = d.get("global_method", "poly")
        p.global_degree = d.get("global_degree", 3)
        g = d.get("grid", {})
        p.grid = GridSettings(
            mode=g.get("mode", "uniform_f"),
            n_points=g.get("n_points", 21),
            custom_values=list(g.get("custom_values", [])),
            f_min=g.get("f_min"),
            f_max=g.get("f_max"),
        )
        p.elements = []
        for eld in d.get("elements", []):
            s = eld.get("settings", {})
            el = Element(
                name=eld["name"],
                values=list(eld["values"]),
                settings=ElementSettings(
                    method=s.get("method", "poly"),
                    degree=s.get("degree", 3),
                    use_global=s.get("use_global", True),
                ),
            )
            p.elements.append(el)
        return p

    def save_json(self, path: str):
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=2)

    @classmethod
    def load_json(cls, path: str) -> "Project":
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        return cls.from_dict(d)


# --------------------------------------------------------------------------- #
#  Calculating
# --------------------------------------------------------------------------- #

def _normalize(f, fmin, fmax):
    f = np.asarray(f, dtype=float)
    if fmax == fmin:
        raise ODZoomError("f_min and f_max are equal — normalization is impossible.")
    return (f - fmin) / (fmax - fmin)


def _format_poly(coeffs, fmin, fmax) -> str:
    """coeffs — from highest degree to lowest (numpy.polyfit convention)."""
    deg = len(coeffs) - 1
    terms = []
    for i, c in enumerate(coeffs):
        power = deg - i
        if power == 0:
            terms.append(f"{c:+.6g}")
        elif power == 1:
            terms.append(f"{c:+.6g}*t")
        else:
            terms.append(f"{c:+.6g}*t^{power}")
    formula = " ".join(terms)
    return (
        f"d(t) = {formula}\n"
        f"t = (f - {fmin:.6g}) / ({fmax:.6g} - {fmin:.6g}),  f — focal length"
    )


def _extract_spline_segments(spline):
    """Returns a list of (t_start, t_end, [c0, c1, c2, c3]) — coefficients
    of the cubic polynomial on each segment, scipy PPoly convention:
    S(t) = c0*(t-t_i)^3 + c1*(t-t_i)^2 + c2*(t-t_i) + c3."""
    x = spline.x
    c = spline.c
    segments = []
    for i in range(len(x) - 1):
        coeffs_i = [float(c[m, i]) for m in range(c.shape[0])]
        segments.append((float(x[i]), float(x[i + 1]), coeffs_i))
    return segments


def _format_spline(method: str, segments, fmin, fmax) -> str:
    names = {
        "cubic": "Cubic spline (natural boundary conditions)",
        "pchip": "PCHIP (monotonic cubic Hermite interpolation)",
        "akima": "Akima spline",
    }
    lines = [
        f"{names[method]}, segments: {len(segments)}.",
        f"t = (f - {fmin:.6g}) / ({fmax:.6g} - {fmin:.6g})",
        "On segment t∈[t_i; t_(i+1)):  d(t) = c0*(t-t_i)^3 + c1*(t-t_i)^2 + c2*(t-t_i) + c3",
    ]
    for i, (t0, t1, coeffs) in enumerate(segments):
        c0, c1, c2, c3 = coeffs
        lines.append(
            f"  segment {i + 1}: t∈[{t0:.6f}; {t1:.6f}]  "
            f"c0={c0:.6g}  c1={c1:.6g}  c2={c2:.6g}  c3={c3:.6g}"
        )
    return "\n".join(lines)


def fit_element(f_ref, y_ref, method: str = "poly", degree: int = 3) -> FitResult:
    """Builds d(f) for a single movable element.

    f_ref, y_ref — reference points (same length, >= 2 points).
    Returns FitResult with callback evaluate(f) -> d.
    """
    f_ref = np.asarray(f_ref, dtype=float)
    y_ref = np.asarray(y_ref, dtype=float)
    if len(f_ref) < 2:
        raise ODZoomError("At least 2 reference points are required for approximation.")

    order = np.argsort(f_ref)
    f_sorted = f_ref[order]
    y_sorted = y_ref[order]
    fmin, fmax = float(f_sorted[0]), float(f_sorted[-1])
    t_sorted = _normalize(f_sorted, fmin, fmax)

    if method == "poly":
        deg = max(1, min(degree, len(f_sorted) - 1))
        coeffs = np.polyfit(t_sorted, y_sorted, deg)
        poly = np.poly1d(coeffs)

        def evaluate(f_query, _poly=poly, _fmin=fmin, _fmax=fmax):
            t_query = _normalize(np.asarray(f_query, dtype=float), _fmin, _fmax)
            return _poly(t_query)

        y_fit = poly(t_sorted)
        desc = _format_poly(coeffs, fmin, fmax)

    elif method == "linear":
        def evaluate(f_query, _f=f_sorted, _y=y_sorted):
            return np.interp(np.asarray(f_query, dtype=float), _f, _y)

        y_fit = evaluate(f_sorted)
        coeffs = list(zip(f_sorted.tolist(), y_sorted.tolist()))
        desc = "Piecewise linear interpolation between adjacent reference points."

    elif method in ("cubic", "pchip", "akima"):
        if method == "cubic":
            spline = CubicSpline(t_sorted, y_sorted, bc_type="natural")
        elif method == "pchip":
            spline = PchipInterpolator(t_sorted, y_sorted)
        else:
            if len(t_sorted) < 3:
                raise ODZoomError("Akima spline requires at least 3 reference points.")
            spline = Akima1DInterpolator(t_sorted, y_sorted)

        def evaluate(f_query, _sp=spline, _fmin=fmin, _fmax=fmax):
            t_query = _normalize(np.asarray(f_query, dtype=float), _fmin, _fmax)
            return _sp(t_query)

        y_fit = evaluate(f_sorted)
        coeffs = _extract_spline_segments(spline)
        desc = _format_spline(method, coeffs, fmin, fmax)

    else:
        raise ODZoomError(f"Unknown approximation method: {method}")

    residuals = np.asarray(y_fit, dtype=float) - y_sorted
    rms = float(np.sqrt(np.mean(residuals ** 2)))
    max_res = float(np.max(np.abs(residuals)))

    return FitResult(
        element_name="",
        method=method,
        description=desc,
        coeffs=coeffs,
        rms_residual=rms,
        max_residual=max_res,
        evaluate=evaluate,
    )


def resolve_method(project: Project, element: Element):
    if element.settings.use_global:
        return project.global_method, project.global_degree
    return element.settings.method, element.settings.degree


def generate_grid(project: Project) -> np.ndarray:
    project.validate()
    f = np.asarray(project.focal_lengths, dtype=float)
    fmin = project.grid.f_min if project.grid.f_min is not None else float(np.min(f))
    fmax = project.grid.f_max if project.grid.f_max is not None else float(np.max(f))
    if fmin >= fmax:
        raise ODZoomError("f_min must be less than f_max.")

    mode = project.grid.mode
    if mode == "uniform_f":
        grid = np.linspace(fmin, fmax, max(2, int(project.grid.n_points)))
    elif mode == "uniform_logf":
        if fmin <= 0:
            raise ODZoomError(
                "For a grid uniform in lg(f), all f values must be positive."
            )
        grid = np.geomspace(fmin, fmax, max(2, int(project.grid.n_points)))
    elif mode == "custom":
        if not project.grid.custom_values:
            raise ODZoomError("The custom focal length list is empty.")
        grid = np.array(sorted(project.grid.custom_values), dtype=float)
    else:
        raise ODZoomError(f"Unknown grid mode: {mode}")
    return grid


def compute_project(project: Project):
    """Performs a complete calculation: all elements + grid + table.

    Returns (table, results, warnings):
      table    — dict {"f": [...], "<element name>": [...], ...}
      results  — list of FitResult (one per element)
      warnings — list of warning strings
    """
    project.validate()
    grid = generate_grid(project)
    f_ref = np.array(project.focal_lengths, dtype=float)
    fmin_ref, fmax_ref = float(np.min(f_ref)), float(np.max(f_ref))

    warnings = []
    if grid.min() < fmin_ref - 1e-9 or grid.max() > fmax_ref + 1e-9:
        warnings.append(
            "Some grid points fall outside the range of reference data — "
            "extrapolation is performed in this region, the result may be unreliable."
        )

    table = {"f": grid.tolist()}
    results = []
    for el in project.elements:
        method, degree = resolve_method(project, el)
        if method == "poly" and degree > project.n_configs() - 1:
            warnings.append(
                f"Element '{el.name}': polynomial degree reduced to "
                f"{project.n_configs() - 1} (insufficient reference points)."
            )
        fit = fit_element(f_ref, el.values, method=method, degree=degree)
        fit.element_name = el.name
        y_grid = np.asarray(fit.evaluate(grid), dtype=float)
        table[el.name] = y_grid.tolist()
        results.append(fit)

    return table, results, warnings