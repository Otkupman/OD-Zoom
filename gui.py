"""
OD-Zoom — graphical interface (Tkinter) on top of the computational core core.py.
"""

from __future__ import annotations

import re
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.cm as _cm

import core
import io_utils


# --------------------------------------------------------------------------- #
#  Helper functions
# --------------------------------------------------------------------------- #

def _fmt(x) -> str:
    return f"{x:g}"


def _parse_optional_float(s: str):
    s = s.strip()
    if not s:
        return None
    try:
        return float(s.replace(",", "."))
    except ValueError:
        return None


def _parse_float_list(s: str) -> list:
    parts = re.split(r"[,;\n\s]+", s.strip())
    values = []
    for part in parts:
        if not part:
            continue
        try:
            values.append(float(part.replace(",", ".")))
        except ValueError:
            pass
    return values


def _get_cmap(name="tab10"):
    try:
        return matplotlib.colormaps[name]
    except Exception:
        return _cm.get_cmap(name)


def _update_row_state(row: dict, method_keys: list):
    """Synchronizes the enabled/disabled state of the 'Method' and
    'Polynomial degree' fields in an element's individual settings row.
    """
    is_custom = bool(row["custom"].get())
    is_poly = method_keys[row["method_combo"].current()] == "poly"
    row["method_combo"].configure(state=("readonly" if is_custom else "disabled"))
    row["degree_spin"].configure(state=("normal" if (is_custom and is_poly) else "disabled"))


# --------------------------------------------------------------------------- #
#  Tab "Input data"
# --------------------------------------------------------------------------- #

class DataGridFrame(ttk.Frame):
    """Editable reference data table:
    rows — configurations, columns — movable elements."""

    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self.header_entries = []
        self.rows = []

        canvas = tk.Canvas(self, highlightthickness=0)
        vsb = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        hsb = ttk.Scrollbar(self, orient="horizontal", command=canvas.xview)
        self.grid_frame = ttk.Frame(canvas)
        self.grid_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.grid_frame, anchor="nw")
        canvas.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

    def rebuild(self):
        for child in self.grid_frame.winfo_children():
            child.destroy()
        self.header_entries = []
        self.rows = []

        project = self.app.project
        ttk.Label(self.grid_frame, text="#", width=4, anchor="center",
                  relief="ridge").grid(row=0, column=0, sticky="nsew")
        ttk.Label(self.grid_frame, text=f"f, {project.units_length}", width=12,
                  anchor="center", relief="ridge").grid(row=0, column=1, sticky="nsew")

        for j, el in enumerate(project.elements):
            e = tk.Entry(self.grid_frame, width=13, justify="center")
            e.insert(0, el.name)
            e.grid(row=0, column=2 + j, sticky="nsew")
            e.bind("<FocusOut>", lambda ev, idx=j: self._rename_element(idx))
            e.bind("<Return>", lambda ev, idx=j: self._rename_element(idx))
            self.header_entries.append(e)

        for i in range(project.n_configs()):
            self._add_row_widgets(i)

        add_btn = ttk.Button(self.grid_frame, text="➕ Add configuration",
                              command=self.app.add_config)
        add_btn.grid(row=project.n_configs() + 1, column=0,
                     columnspan=3 + len(project.elements), sticky="w", pady=6)

    def _add_row_widgets(self, i):
        project = self.app.project
        row = i + 1
        lbl = ttk.Label(self.grid_frame, text=str(i + 1), width=4, anchor="center",
                         relief="ridge")
        lbl.grid(row=row, column=0, sticky="nsew")

        f_entry = tk.Entry(self.grid_frame, width=13, justify="center")
        f_entry.insert(0, _fmt(project.focal_lengths[i]))
        f_entry.grid(row=row, column=1, sticky="nsew")
        f_entry.bind("<FocusOut>", lambda ev, idx=i: self._commit_f(idx))
        f_entry.bind("<Return>", lambda ev, idx=i: self._commit_f(idx))

        cells = []
        for j, el in enumerate(project.elements):
            c = tk.Entry(self.grid_frame, width=13, justify="center")
            c.insert(0, _fmt(el.values[i]))
            c.grid(row=row, column=2 + j, sticky="nsew")
            c.bind("<FocusOut>", lambda ev, ci=i, cj=j: self._commit_cell(ci, cj))
            c.bind("<Return>", lambda ev, ci=i, cj=j: self._commit_cell(ci, cj))
            cells.append(c)

        del_btn = ttk.Button(self.grid_frame, text="❌", width=3,
                              command=lambda idx=i: self.app.remove_config(idx))
        del_btn.grid(row=row, column=2 + len(project.elements), sticky="nsew")

        self.rows.append({"f": f_entry, "cells": cells, "del": del_btn, "label": lbl})

    def _rename_element(self, idx):
        name = self.header_entries[idx].get().strip()
        if name:
            self.app.project.elements[idx].name = name

    def _commit_f(self, idx):
        entry = self.rows[idx]["f"]
        try:
            self.app.project.focal_lengths[idx] = float(entry.get().replace(",", "."))
        except ValueError:
            entry.delete(0, "end")
            entry.insert(0, _fmt(self.app.project.focal_lengths[idx]))
            self.app.set_status("Invalid f value — please enter a number.", error=True)

    def _commit_cell(self, i, j):
        entry = self.rows[i]["cells"][j]
        try:
            self.app.project.elements[j].values[i] = float(entry.get().replace(",", "."))
        except ValueError:
            entry.delete(0, "end")
            entry.insert(0, _fmt(self.app.project.elements[j].values[i]))
            self.app.set_status("Invalid value — please enter a number.", error=True)


class DataTab(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app

        import_frame = ttk.Frame(self)
        import_frame.pack(fill="x", padx=6, pady=(6, 2))
        ttk.Button(import_frame, text="📂 Import from file...", command=app.import_data).pack(side="left")
        ttk.Button(import_frame, text="🧹 Clear all", command=app.new_project).pack(side="right")
        
        settings_frame = ttk.Frame(self)
        settings_frame.pack(fill="x", padx=6, pady=6)
        ttk.Label(settings_frame, text="Number of movable elements:").pack(side="left")
        self.n_elements_var = tk.IntVar(value=max(1, app.project.n_elements()))
        ttk.Spinbox(settings_frame, from_=1, to=50, width=5,
                    textvariable=self.n_elements_var).pack(side="left", padx=4)
        ttk.Button(settings_frame, text="Apply", command=self._apply_n_elements).pack(side="left", padx=6)

        ttk.Separator(settings_frame, orient="vertical").pack(side="left", fill="y", padx=8)
        ttk.Label(settings_frame, text="Units:").pack(side="left")
        self.units_var = tk.StringVar(value=app.project.units_length)
        units_entry = ttk.Entry(settings_frame, width=6, textvariable=self.units_var)
        units_entry.pack(side="left", padx=4)
        units_entry.bind("<FocusOut>", self._apply_units)
        units_entry.bind("<Return>", self._apply_units)

        hint = ("Enter focal lengths and thicknesses of the reference configurations")
        ttk.Label(self, text=hint, foreground="gray").pack(anchor="w", padx=6)

        self.grid_frame = DataGridFrame(self, app)
        self.grid_frame.pack(fill="both", expand=True, padx=6, pady=6)
        
        tk.Button(self, text="➡ Next", font=("Arial", 14, "bold"), bg="green", fg="white", activebackground="darkgreen", bd=4, command=self._go_next).pack(fill="x", padx=10, pady=10)
    def _go_next(self):
        self.app.notebook.select(self.app.settings_tab)

    def _apply_n_elements(self):
        try:
            n = int(self.n_elements_var.get())
        except (tk.TclError, ValueError):
            return
        self.app.set_n_elements(n)

    def _apply_units(self, event=None):
        self.app.project.units_length = self.units_var.get().strip() or "mm"

    def refresh(self):
        self.n_elements_var.set(max(1, self.app.project.n_elements()))
        self.units_var.set(self.app.project.units_length)
        self.grid_frame.rebuild()


# --------------------------------------------------------------------------- #
#  Tab "Calculation parameters"
# --------------------------------------------------------------------------- #

class SettingsTab(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app

        grid_box = ttk.LabelFrame(self, text="🧮 Grid of intermediate values")
        grid_box.pack(fill="x", padx=6, pady=6)

        self.grid_mode_var = tk.StringVar(value="uniform_f")
        for key, label in core.GRID_MODES.items():
            ttk.Radiobutton(grid_box, text=label, value=key, variable=self.grid_mode_var,
                             command=self._on_mode_change).pack(anchor="w", padx=8, pady=2)

        row1 = ttk.Frame(grid_box)
        row1.pack(fill="x", padx=8, pady=2)
        ttk.Label(row1, text="Number of grid points N:").pack(side="left")
        self.n_points_var = tk.IntVar(value=21)
        ttk.Spinbox(row1, from_=2, to=1000, width=6, textvariable=self.n_points_var).pack(side="left", padx=4)

        row2 = ttk.Frame(grid_box)
        row2.pack(fill="x", padx=8, pady=2)
        ttk.Label(row2, text="Focal length from:").pack(side="left")
        self.fmin_var = tk.StringVar(value="")
        ttk.Entry(row2, width=10, textvariable=self.fmin_var).pack(side="left", padx=4)
        ttk.Label(row2, text="to:").pack(side="left")
        self.fmax_var = tk.StringVar(value="")
        ttk.Entry(row2, width=10, textvariable=self.fmax_var).pack(side="left", padx=4)

        row3 = ttk.Frame(grid_box)
        row3.pack(fill="x", padx=8, pady=2)
        ttk.Label(row3, text="(empty — taken from reference data; outside range — extrapolation)",
                  foreground="gray").pack(side="left")

        self.custom_label = ttk.Label(
            grid_box, text="List of focal lengths (comma, semicolon or newline separated):"
        )
        self.custom_text = tk.Text(grid_box, height=4)
        self._on_mode_change()

        method_box = ttk.LabelFrame(self, text="⚙ Default calculation method")
        method_box.pack(fill="x", padx=6, pady=6)
        row = ttk.Frame(method_box)
        row.pack(fill="x", padx=8, pady=6)
        ttk.Label(row, text="Method:").pack(side="left")
        self._method_keys = list(core.METHODS.keys())
        self.method_combo = ttk.Combobox(row, state="readonly", width=26,
                                          values=list(core.METHODS.values()))
        self.method_combo.current(0)
        self.method_combo.pack(side="left", padx=4)
        self.method_combo.bind("<<ComboboxSelected>>", self._on_method_change)
        ttk.Label(row, text="Polynomial degree:").pack(side="left", padx=(12, 0))
        self.degree_var = tk.IntVar(value=3)
        self.degree_spin = ttk.Spinbox(row, from_=1, to=10, width=5, textvariable=self.degree_var)
        self.degree_spin.pack(side="left", padx=4)

        self.el_box = ttk.LabelFrame(self, text="🔧 Individual element settings")
        self.el_box.pack(fill="both", expand=True, padx=6, pady=6)
        self.el_rows = []

        tk.Button(self, text="⚡ Calculate", font=("Arial", 14, "bold"), bg="yellow", activebackground="orange", bd=4, command=self.app.compute).pack(fill="x", padx=10, pady=10)

    def _on_mode_change(self):
        if self.grid_mode_var.get() == "custom":
            self.custom_label.pack(anchor="w", padx=8)
            self.custom_text.pack(fill="x", padx=8, pady=(0, 6))
        else:
            self.custom_label.pack_forget()
            self.custom_text.pack_forget()

    def _on_method_change(self, event=None):
        is_poly = self._method_keys[self.method_combo.current()] == "poly"
        self.degree_spin.configure(state="normal" if is_poly else "disabled")

    def rebuild_elements(self):
        for child in self.el_box.winfo_children():
            child.destroy()
        self.el_rows = []
        keys = self._method_keys

        for el in self.app.project.elements:
            row_frame = ttk.Frame(self.el_box)
            row_frame.pack(fill="x", padx=6, pady=2)
            ttk.Label(row_frame, text=el.name, width=16).pack(side="left")

            # custom == True  ->  the element's own settings are used
            # custom == False ->  the global settings are used
            custom_var = tk.BooleanVar(value=not el.settings.use_global)
            method_combo = ttk.Combobox(row_frame, state="readonly", width=24,
                                         values=list(core.METHODS.values()))
            method_combo.current(keys.index(el.settings.method))
            degree_var = tk.IntVar(value=el.settings.degree)
            degree_spin = ttk.Spinbox(row_frame, from_=1, to=10, width=4, textvariable=degree_var)

            row = {
                "custom": custom_var,
                "method_combo": method_combo,
                "degree_var": degree_var,
                "degree_spin": degree_spin,
            }

            ttk.Checkbutton(
                row_frame, text="Custom", variable=custom_var,
                command=lambda r=row: _update_row_state(r, keys),
            ).pack(side="left", padx=4)
            method_combo.pack(side="left", padx=4)
            method_combo.bind(
                "<<ComboboxSelected>>", lambda e, r=row: _update_row_state(r, keys)
            )
            degree_spin.pack(side="left", padx=4)

            _update_row_state(row, keys)
            self.el_rows.append(row)

    def apply_to_project(self, project: core.Project):
        try:
            keys = self._method_keys
            project.global_method = keys[self.method_combo.current()]
            project.global_degree = int(self.degree_var.get())

            project.grid.mode = self.grid_mode_var.get()
            project.grid.n_points = int(self.n_points_var.get())
            project.grid.f_min = _parse_optional_float(self.fmin_var.get())
            project.grid.f_max = _parse_optional_float(self.fmax_var.get())
            project.grid.custom_values = _parse_float_list(self.custom_text.get("1.0", "end"))

            for idx, row in enumerate(self.el_rows):
                el = project.elements[idx]
                el.settings.use_global = not bool(row["custom"].get())
                el.settings.method = keys[row["method_combo"].current()]
                el.settings.degree = int(row["degree_var"].get())
        except (tk.TclError, ValueError) as e:
            raise core.ODZoomError(f"Invalid calculation parameters: {e}")

    def load_from_project(self, project: core.Project):
        """Inverse of apply_to_project: transfers values from the Project
        object into the widgets of the 'Calculation parameters' tab."""
        keys = self._method_keys
        self.method_combo.current(keys.index(project.global_method))
        self._on_method_change()
        self.degree_var.set(project.global_degree)

        self.grid_mode_var.set(project.grid.mode)
        self._on_mode_change()
        self.n_points_var.set(project.grid.n_points)
        self.fmin_var.set("" if project.grid.f_min is None else _fmt(project.grid.f_min))
        self.fmax_var.set("" if project.grid.f_max is None else _fmt(project.grid.f_max))

        self.custom_text.delete("1.0", "end")
        if project.grid.custom_values:
            self.custom_text.insert("1.0", ", ".join(_fmt(v) for v in project.grid.custom_values))


# --------------------------------------------------------------------------- #
#  Tab "Results"
# --------------------------------------------------------------------------- #

class ResultsTab(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app

        paned = ttk.PanedWindow(self, orient="vertical")
        paned.pack(fill="both", expand=True)

        text_frame = ttk.Frame(paned)
        self.text = tk.Text(text_frame, height=12, wrap="word", state="disabled")
        vsb = ttk.Scrollbar(text_frame, orient="vertical", command=self.text.yview)
        self.text.configure(yscrollcommand=vsb.set)
        self.text.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        paned.add(text_frame, weight=1)

        table_frame = ttk.Frame(paned)
        self.tree = ttk.Treeview(table_frame, show="headings")
        vsb2 = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        hsb2 = ttk.Scrollbar(table_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb2.set, xscrollcommand=hsb2.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb2.grid(row=0, column=1, sticky="ns")
        hsb2.grid(row=1, column=0, sticky="ew")
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)
        paned.add(table_frame, weight=2)

        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill="x", pady=4)
        ttk.Button(btn_frame, text="💾 Save table (*.xlsx, *.csv)...",
                   command=self.app.export_table).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="💾 Save report (*.txt)...",
                   command=self.app.export_report).pack(side="left", padx=4)

    def show_results(self, table, results, warnings, units):
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        if warnings:
            self.text.insert("end", "⚠️  WARNINGS:\n")
            for w in warnings:
                self.text.insert("end", f"  • {w}\n")
            self.text.insert("end", "\n")

        for fit in results:
            self.text.insert("end", f"• {fit.element_name} ({core.METHODS[fit.method]})\n")
            self.text.insert("end", fit.description + "\n")
            if fit.max_residual < 1e-9:
                self.text.insert(
                    "end", "Exact interpolation — the curve passes through all reference points.\n\n"
                )
            else:
                self.text.insert(
                    "end",
                    f"RMS: {fit.rms_residual:.5g} {units}   "
                    f"Max. deviation: {fit.max_residual:.5g} {units}\n\n",
                )
        self.text.configure(state="disabled")

        columns = ["f"] + [fit.element_name for fit in results]
        self.tree.delete(*self.tree.get_children())
        self.tree["columns"] = columns
        for col in columns:
            self.tree.heading(col, text=(f"f, {units}" if col == "f" else col))
            self.tree.column(col, width=100, anchor="center")
        n = len(table["f"])
        for i in range(n):
            row = [f"{table['f'][i]:.5g}"] + [f"{table[c][i]:.5g}" for c in columns[1:]]
            self.tree.insert("", "end", values=row)


# --------------------------------------------------------------------------- #
#  Plot panel
# --------------------------------------------------------------------------- #

class NoSaveToolbar(NavigationToolbar2Tk):
    """Standard matplotlib toolbar, but without the 'Save' button."""
    toolitems = [t for t in NavigationToolbar2Tk.toolitems if t[0] != "Save"]

class PlotPanel(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master)

        self.app = app
        self.figure = Figure(figsize=(6, 5), dpi=100)
        self.ax = self.figure.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        toolbar_frame = ttk.Frame(self)
        toolbar_frame.pack(fill="x")

        self.toolbar = NoSaveToolbar(self.canvas, toolbar_frame)
        self.toolbar.update()

        ttk.Button(
            toolbar_frame,
            text="💾 Save plot...",
            command=self.app.export_plot
        ).pack(side="right", padx=4, pady=2)

        self.accumulate_var = tk.BooleanVar(value=False)

        ttk.Checkbutton(
            toolbar_frame,
            text="📌 Accumulate curves",
            variable=self.accumulate_var
        ).pack(side="left", padx=8, pady=2)

        self._run_count = 0
        self._placeholder_visible = False

        self._draw_placeholder()

    def _draw_placeholder(self):
        self.ax.clear()
        self._run_count = 0
        self._placeholder_visible = True

        self.ax.set_title("Waiting for displacement calculation")
        self.ax.set_xlabel("f")
        self.ax.set_ylabel("d")
        self.ax.grid(True, alpha=0.3)

        self.ax.text(
            0.5,
            0.5,
            "OD-Zoom",
            ha="center",
            va="center",
            transform=self.ax.transAxes,
            color="gray"
        )

        self.canvas.draw_idle()

    def plot(self, project: core.Project, table: dict, results):
        if self._placeholder_visible:
            self.ax.clear()
            self._placeholder_visible = False


        if not self.accumulate_var.get():
            self.ax.clear()
            self._run_count = 0

        f_ref = project.focal_lengths
        fmin = min(min(table["f"]), min(f_ref))
        fmax = max(max(table["f"]), max(f_ref))

        f_fine = np.linspace(fmin, fmax, 400)
        cmap = _get_cmap("tab10")

        linestyles = ["-", "--", "-.", ":"]
        ls = (
            linestyles[self._run_count % len(linestyles)]
            if self.accumulate_var.get()
            else "-"
        )

        for i, fit in enumerate(results):
            color = cmap(i % 10)
            el = project.elements[i]

            y_fine = np.asarray(
                fit.evaluate(f_fine),
                dtype=float
            )

            if self.accumulate_var.get():
                label = (
                    f"{el.name} — run "
                    f"{self._run_count + 1} "
                    f"({core.METHODS[fit.method]})"
                )
            else:
                label = el.name

            self.ax.plot(
                f_fine,
                y_fine,
                ls,
                color=color,
                linewidth=1.6,
                label=label
            )

            self.ax.plot(
                f_ref,
                el.values,
                "o",
                color=color,
                markersize=5,
                markerfacecolor="white",
                markeredgewidth=1.4
            )

            self.ax.plot(
                table["f"],
                table[el.name],
                "x",
                color=color,
                markersize=5,
                alpha=0.8
            )

        self.ax.set_title("Displacement diagram of movable elements")
        self.ax.set_xlabel(f"f, {project.units_length}")
        self.ax.set_ylabel(f"d, {project.units_length}")
        self.ax.grid(True, alpha=0.3)
        self.ax.minorticks_on()
        self.ax.grid(
            which="minor",
            color="gray",
            linestyle=":",
            linewidth=0.1
        )
        self.ax.legend(loc="best", fontsize=8)

        self.figure.tight_layout()
        self.canvas.draw_idle()

        self._run_count += 1


# --------------------------------------------------------------------------- #
#  Main application window
# --------------------------------------------------------------------------- #

class ODZoomApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("OD-Zoom")
        self.geometry("1280x800")
        self.minsize(1024, 640)
        icon = "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAMAAAAoLQ9TAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAPUExURQCi6P///5nZ6gAAAAAAAAE8ze4AAAAFdFJOU/////8A+7YOUwAAAAlwSFlzAAAOxAAADsQBlSsOGwAAABl0RVh0U29mdHdhcmUAUGFpbnQuTkVUIDUuMS4xMhMBR3QAAAC4ZVhJZklJKgAIAAAABQAaAQUAAQAAAEoAAAAbAQUAAQAAAFIAAAAoAQMAAQAAAAIAAAAxAQIAEQAAAFoAAABphwQAAQAAAGwAAAAAAAAADHcBAOgDAAAMdwEA6AMAAFBhaW50Lk5FVCA1LjEuMTIAAAMAAJAHAAQAAAAwMjMwAaADAAEAAAABAAAABaAEAAEAAACWAAAAAAAAAAIAAQACAAQAAABSOTgAAgAHAAQAAAAwMTAwAAAAAF0islSpwhrYAAAAWElEQVQoU3WNCQ7AIAgEAfn/m7uzaNImFYgsw2F0R+jBLIg8ABHiOSMj5Fk1wEJeB1hcwXqBKN0CWPARG8sjVHQAHEEoXDAx1Bm1k4HbO+n+x/4mzrJT9wNLUALAz6R9EwAAAABJRU5ErkJggg=="
        img = tk.PhotoImage(data=icon)
        self.tk.call("wm", "iconphoto", self._w, img)
        self.iconphoto(True, img)
        try:
            ttk.Style(self).theme_use("xpnative")
        except tk.TclError:
            pass

        self.project = core.Project()
        self._seed_default_project()
        self.current_path = None
        self.last_table = None
        self.last_results = None
        self.last_warnings = []

        self._build_menu()
        self._build_body()
        self._build_statusbar()
        self.refresh_all()

    def _seed_default_project(self):
        p = self.project
        p.focal_lengths = [1, 2, 3]
        p.add_element("d1")
        p.add_element("d2")
        p.elements[0].values = [0, 0, 0]
        p.elements[1].values = [0, 0, 0]

    def _build_menu(self):
        menubar = tk.Menu(self)
        file_menu = tk.Menu(menubar, tearoff=False)
        file_menu.add_command(label="🆕 New project", command=self.new_project)
        file_menu.add_command(label="📂 Open project...", command=self.open_project)
        file_menu.add_command(label="💾 Save project", command=self.save_project)
        file_menu.add_command(label="💾 Save project as...", command=self.save_project_as)
        file_menu.add_separator()
        file_menu.add_command(label="Import reference data...", command=self.import_data)
        file_menu.add_command(label="Export table...", command=self.export_table)
        file_menu.add_command(label="Export report...", command=self.export_report)
        file_menu.add_command(label="Export plot...", command=self.export_plot)
        file_menu.add_separator()
        file_menu.add_command(label="🚪 Exit", command=self.destroy)
        menubar.add_cascade(label="File", menu=file_menu)

        help_menu = tk.Menu(menubar, tearoff=False)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="Help", command=self.show_help)
        menubar.add_cascade(label="Help", menu=help_menu)
        self.configure(menu=menubar)

    def _build_body(self):
        paned = ttk.PanedWindow(self, orient="horizontal")
        paned.pack(fill="both", expand=True)

        left = ttk.Frame(paned)
        self.notebook = ttk.Notebook(left)
        self.notebook.pack(fill="both", expand=True)

        self.data_tab = DataTab(self.notebook, self)
        self.settings_tab = SettingsTab(self.notebook, self)
        self.results_tab = ResultsTab(self.notebook, self)
        self.notebook.add(self.data_tab, text="📋 Input data")
        self.notebook.add(self.settings_tab, text="📐 Calculation parameters")
        self.notebook.add(self.results_tab, text="📊 Results")
        paned.add(left, weight=1)

        right = ttk.Frame(paned)
        self.plot_panel = PlotPanel(right, self)
        self.plot_panel.pack(fill="both", expand=True)
        paned.add(right, weight=1)

    def _build_statusbar(self):
        self.status_var = tk.StringVar(value="Ready.")
        bar = ttk.Label(self, textvariable=self.status_var, anchor="w", relief="sunken")
        bar.pack(fill="x", side="bottom")

    def set_status(self, text, error=False):
        self.status_var.set(text)

    # ---------- project operations ----------
    def refresh_all(self):
        self.data_tab.refresh()
        self.settings_tab.load_from_project(self.project)
        self.settings_tab.rebuild_elements()

    def add_config(self):
        last_f = self.project.focal_lengths[-1] if self.project.focal_lengths else 0.0
        self.project.add_config(last_f+1)
        self.data_tab.refresh()

    def remove_config(self, idx):
        if self.project.n_configs() <= 2:
            messagebox.showwarning("OD-Zoom", "At least 2 configurations must remain.")
            return
        self.project.remove_config(idx)
        self.data_tab.refresh()

    def set_n_elements(self, n):
        if n < 1:
            return
        self.project.set_n_elements(n)
        self.refresh_all()

    def new_project(self):
        if not messagebox.askyesno("OD-Zoom", "Create a new project? Unsaved data will be lost."):
            return
        self.project = core.Project()
        self.project.focal_lengths = [1.0, 2.0]
        self.project.add_element("d1")
        self.current_path = None
        self.last_table = None
        self.last_results = None
        self.refresh_all()
        self.plot_panel._draw_placeholder()
        self.set_status("New project created.")

    def open_project(self):
        path = filedialog.askopenfilename(
            title="Open OD-Zoom project", filetypes=[("OD-Zoom project", "*.json")]
        )
        if not path:
            return
        try:
            self.project = core.Project.load_json(path)
        except Exception as e:
            messagebox.showerror("OD-Zoom", f"Failed to open project:\n{e}")
            return
        self.current_path = path
        self.last_table = None
        self.last_results = None
        self.refresh_all()
        self.plot_panel._draw_placeholder()
        self.set_status(f"Project loaded: {path}")

    def save_project(self):
        if not self.current_path:
            return self.save_project_as()
        try:
            self._sync_settings()
            self.project.save_json(self.current_path)
            self.set_status(f"Project saved: {self.current_path}")
        except Exception as e:
            messagebox.showerror("OD-Zoom", f"Failed to save project:\n{e}")

    def save_project_as(self):
        path = filedialog.asksaveasfilename(
            title="Save OD-Zoom project", defaultextension=".json",
            filetypes=[("OD-Zoom project", "*.json")],
        )
        if not path:
            return
        self.current_path = path
        self.save_project()

    def import_data(self):
        path = filedialog.askopenfilename(
            title="Import reference data",
            filetypes=[("Tables", "*.csv *.xlsx *.xls"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            new_project = io_utils.import_reference_table(path)
        except core.ODZoomError as e:
            messagebox.showerror("OD-Zoom", f"Import error:\n{e}")
            return
        except Exception as e:
            messagebox.showerror("OD-Zoom", f"Import error:\n{e}")
            return
        new_project.units_length = self.project.units_length
        self.project = new_project
        self.current_path = None
        self.last_table = None
        self.last_results = None
        self.refresh_all()
        self.plot_panel._draw_placeholder()
        self.set_status(
            f"Imported configurations: {self.project.n_configs()}, "
            f"elements: {self.project.n_elements()}"
        )

    def _sync_settings(self):
        self.settings_tab.apply_to_project(self.project)

    def compute(self):
        try:
            self._sync_settings()
            table, results, warnings = core.compute_project(self.project)
        except core.ODZoomError as e:
            messagebox.showerror("OD-Zoom — error", str(e))
            self.set_status(str(e), error=True)
            return
        except Exception as e:
            messagebox.showerror("OD-Zoom — unexpected error", str(e))
            self.set_status(f"Error: {e}", error=True)
            return

        self.last_table = table
        self.last_results = results
        self.last_warnings = warnings
        self.results_tab.show_results(table, results, warnings, self.project.units_length)
        self.plot_panel.plot(self.project, table, results)
        msg = "Calculation completed."
        if warnings:
            msg += " There are warnings — see the 'Results' tab."
        self.set_status(msg)
        self.notebook.select(self.results_tab)

    def export_table(self):
        if not self.last_table:
            messagebox.showinfo("OD-Zoom", "Please run the calculation first.")
            return
        path = filedialog.asksaveasfilename(
            title="Export table", defaultextension=".xlsx",
            filetypes=[("Excel", "*.xlsx"), ("CSV", "*.csv")],
        )
        if not path:
            return
        try:
            element_order = [fit.element_name for fit in self.last_results]
            io_utils.export_table(self.last_table, path, element_order)
            self.set_status(f"Table saved: {path}")
        except Exception as e:
            messagebox.showerror("OD-Zoom", f"Export error:\n{e}")

    def export_report(self):
        if not self.last_results:
            messagebox.showinfo("OD-Zoom", "Please run the calculation first.")
            return
        path = filedialog.asksaveasfilename(
            title="Export report", defaultextension=".txt", filetypes=[("Text", "*.txt")]
        )
        if not path:
            return
        try:
            io_utils.export_report(self.last_results, self.last_warnings, self.last_table,
                                    path, self.project.units_length)
            self.set_status(f"Report saved: {path}")
        except Exception as e:
            messagebox.showerror("OD-Zoom", f"Export error:\n{e}")

    def export_plot(self):
        path = filedialog.asksaveasfilename(
            title="Export plot", defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")],
        )
        if not path:
            return
        try:
            self.plot_panel.figure.savefig(path, dpi=200, bbox_inches="tight")
            self.set_status(f"Plot saved: {path}")
        except Exception as e:
            messagebox.showerror("OD-Zoom", f"Export error:\n{e}")

    def show_about(self):
        messagebox.showinfo(
            "About OD-Zoom",
            "OD-Zoom v1.0\n\n"
            "A program for calculating intermediate values of variable thicknesses (air gaps) in varifocal (zoom) optical systems.\n"
            "Besides variator-compensator pairs, the program is applicable to focusing, athermalization, and other units where the mechanism's displacement law must be reconstructed from a finite set of reference points.\n\n"
            "Features:\n"
            " 📐 polynomial least-squares (LSQ) approximation, piecewise linear interpolation, cubic spline, monotonic cubic Hermite interpolation (PCHIP — Piecewise Cubic Hermite Interpolating Polynomial), Akima spline;\n"
            " 🧮 arbitrary output grid (by focal length, by logarithm, or from a list);\n"
            " 📊 quality estimation (RMS, maximum deviation) and displacement diagram of movable components;\n"
            " 💾 import/export of tables (CSV/XLSX) and project saving (JSON).\n"
            "\n"
            "© Otkupman D. G., 2026",
        )

    def show_help(self):
        help_window = tk.Toplevel(self)
        help_window.title("❓ Help")
        help_window.geometry("555x350")
        #help_window.resizable(False, False)

        BG_COLOR = "#ffffff"       # White background for text
        TEXT_COLOR = "#2c3e50"     # Dark gray text color
        LINE_COLOR = "#eaeded"     # Light gray separator line

        canvas = tk.Canvas(help_window, highlightthickness=0, bg=BG_COLOR)
        scrollbar = tk.Scrollbar(help_window, orient="vertical", command=canvas.yview)
        
        canvas.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        scrollable_frame = tk.Frame(canvas, bg=BG_COLOR)
        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

        def configure_scroll_region(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        scrollable_frame.bind("<Configure>", configure_scroll_region)

        def configure_canvas_width(event):
            canvas.itemconfig(canvas_window, width=event.width)
        canvas.bind("<Configure>", configure_canvas_width)

        # Mouse wheel scrolling
        #def on_mouse_wheel(event):
        #    canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        #canvas.bind_all("<MouseWheel>", on_mouse_wheel)
        
        def add_section(title, text, is_first=False):
            if not is_first:
                line = tk.Frame(scrollable_frame, height=1, bg=LINE_COLOR)
                line.pack(fill="x", padx=20, pady=15)

            lbl_title = tk.Label(
                scrollable_frame, 
                text=title, 
                font=("Segoe UI", 11, "bold"), 
                bg=BG_COLOR, 
                fg="#1576bb"
            )
            lbl_title.pack(anchor="w", padx=20, pady=(5, 5))

            lbl_text = tk.Label(
                scrollable_frame, 
                text=text, 
                font=("Segoe UI", 12), 
                bg=BG_COLOR, 
                fg=TEXT_COLOR,
                justify="left", 
                wraplength=500 # Automatic text wrapping by width
            )
            lbl_text.pack(anchor="w", padx=20, pady=(0, 5))

        main_title = tk.Label(
            scrollable_frame, 
            text="Quick user guide", 
            font=("Segoe UI", 14, "bold"), 
            bg=BG_COLOR, 
            fg="#2c3e50"
        )
        main_title.pack(anchor="w", padx=20, pady=(20, 10))

        add_section(
            "1. Entering input data (tab '📋 Input data').", 
            "1.1. Specify the number of movable elements (variable thicknesses) and click 'Apply' — the table will be rebuilt with the required number of columns.\n"
            "1.2. The name of each element can be changed directly in the column header (for example, 'd1 (L1–L2)') — this name will then be used everywhere: in formulas, the results table, and the plot legend.\n"
            "1.3. Fill in the table: each row is one reference configuration (focal length + thickness for each element). Minimum — 2 configurations.\n"
            "1.4. The '➕ Add configuration' button adds a row, '❌' deletes one.\n"
            "1.5. Instead of manual entry, you can click '📂 Import from file...' and load a ready-made table from CSV/XLSX.\n",
            is_first=True
        )
        
        add_section(
            "2. Setting parameters (tab '📐 Calculation parameters').", 
            "🧮 Grid of intermediate values — how to place the points you want to obtain:\n"
            " ⁃ uniform in f — ordinary arithmetic progression of focal lengths;\n"
            " ⁃ uniform in lg(f) — geometric progression;\n"
            " ⁃ custom list — enter the desired f values yourself.\n"
            "For uniform grids, specify the number of points N. The fields 'Focal length from / to' allow extending the range beyond the entered data (in this case the program will explicitly warn that these points involve extrapolation rather than interpolation/approximation from actual data).\n"
            "⚙ Default calculation method — a common method for all elements:\n"
            "◦ polynomial (LSQ) — approximation — a smooth law, not required to pass exactly through all points (degree 1–10);\n"
            "◦ piecewise linear — interpolation — the simplest and most predictable option;\n"
            "◦ cubic spline — interpolation — a smooth curve passing exactly through all points;\n"
            "◦ PCHIP — interpolation — like a cubic spline, but without 'overshoots';\n"
            "◦ Akima spline — interpolation — less sensitive to sharp neighboring jumps than a cubic spline.\n"
            "Interpolation methods always pass exactly through the entered points (residual = 0); the approximation method (polynomial) does not, but it gives a smoother and more predictable law when the input data itself is not perfectly smooth.\n"
            "🔧 Individual element settings — if a particular element needs its own method, check the 'Custom' box next to it — then that element gets its own method and polynomial degree (the degree is active only if the polynomial is selected). Without the checkbox, the element is calculated using the 'default' method.\n"
            "When everything is configured — click the '⚡ Calculate' button.\n"
        )
        
        add_section(
            "3. Checking results (tab '📊 Results' and the plot).", 
            " ∙ Top left — the formula/description of the method for each element (for the polynomial — explicit coefficients, for splines — coefficients per segment) and accuracy estimation (RMS and maximum deviation from reference points; for interpolations it is always 'exact interpolation').\n"
            " ∙ Below — the table of discrete values on the selected grid.\n"
            " ∙ On the right — the plot: solid line — the resulting dependence, circles — the entered reference points, crosses — the calculated discrete values.\n"
        )
        
        add_section(
            "4. Saving results.", 
            "'💾 Save table...' — export the calculated table to CSV or XLSX.\n"
            "'💾 Save report...' — a text file with formulas, accuracy estimation, and the table.\n"
            "'💾 Save plot...' — export the diagram to PNG/PDF/SVG.\n"
            "'💾 Save project' ('File' menu) — saves all entered data and settings to JSON so you can continue later.\n"
        )

        bottom_spacer = tk.Frame(scrollable_frame, height=20, bg=BG_COLOR)
        bottom_spacer.pack()

def main():
    app = ODZoomApp()
    app.mainloop()


if __name__ == "__main__":
    main()