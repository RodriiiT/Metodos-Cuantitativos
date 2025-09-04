from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import flet as ft
import numpy as np

from ProyectoFinal.núcleo import ExpressionParser, Problem, Plotter, ConstraintCheck
from ProyectoFinal.resolvedores.gradient_descent import GradientDescentSolver
from ProyectoFinal.resolvedores.lagrange import LagrangeSolver
from ProyectoFinal.resolvedores.unconstrained import UnconstrainedDirectSolver
from ProyectoFinal.resolvedores.base import SolverResult


@dataclass
class MethodSelection:
    maximize: bool
    gradient: bool
    lagrange: bool
    unconstrained: bool


class OptimizationAppUI:
    """
    Interfaz moderna de Flet para optimización no lineal con múltiples métodos y gráficos.
    Tema oscuro limpio, diseño OOP modular y reporte claro de resultados.
    """

    def __init__(self) -> None:
        # Theme
        self.bg_color = "#0b0c0d"
        self.surface = "#121314"
        self.surface_alt = "#181a1b"
        self.border = "#2a2d2f"
        self.text = "#e7e9ea"
        self.muted = "#9aa0a6"
        self.accent = "#20C997"   # teal
        self.warn = "#f59f00"     # amber
        self.error = "#e03131"    # red

        # Core helpers
        self.parser = ExpressionParser()
        self.plotter = Plotter(
            bg_color=self.bg_color,
            accent_color=self.accent,
            text_color=self.text,
            secondary_color=self.surface,
        )

        # Page and UI state
        self.page: Optional[ft.Page] = None

        # Inputs
        self.field_objective: Optional[ft.TextField] = None
        self.field_variables: Optional[ft.TextField] = None
        self.field_initial_guess: Optional[ft.TextField] = None

        # Constraints
        self.constraints_column: Optional[ft.Column] = None
        self.constraint_rows: List[Tuple[ft.Row, ft.TextField]] = []

        # Method selection
        self.cb_grad: Optional[ft.Checkbox] = None
        self.cb_lagrange: Optional[ft.Checkbox] = None
        self.cb_unconstrained: Optional[ft.Checkbox] = None
        self.cb_maximize: Optional[ft.Checkbox] = None

        # Result areas
        self.results_column: Optional[ft.Column] = None
        self.plot_container: Optional[ft.Container] = None
        self.tabs_container: Optional[ft.Tabs] = None

    # ---------- UI Build ----------

    def build(self, page: ft.Page):
        self.page = page
        page.title = "Calculadora de Optimización No Lineal"
        page.bgcolor = self.bg_color
        page.theme_mode = ft.ThemeMode.DARK
        page.padding = 0
        page.window_width = 1280
        page.window_height = 820

        appbar = ft.AppBar(
            bgcolor=self.surface,
            color=self.text,
            title=ft.Text("Calculadora de Optimización No Lineal", size=18, weight=ft.FontWeight.W_600),
            center_title=False,
        )

        # Left panel: inputs
        inputs_card = self._build_inputs_panel()

        # Right panel: results and plot
        results_card = self._build_results_panel()

        content = ft.Row(
            [
                ft.Container(
                    content=inputs_card,
                    width=520,
                    bgcolor=self.surface,
                    padding=20,
                    border_radius=0,
                    border=ft.border.only(right=ft.BorderSide(1, self.border)),
                ),
                ft.Container(
                    content=results_card,
                    expand=True,
                    bgcolor=self.surface_alt,
                    padding=20,
                ),
            ],
            expand=True,
        )

        page.add(appbar, content)

    def _build_inputs_panel(self) -> ft.Column:
        title = ft.Text("Modelo", size=16, weight=ft.FontWeight.BOLD, color=self.text)

        self.field_objective = ft.TextField(
            label="Función objetivo f(x)",
            hint_text="Ejemplo: x**2 + y**2",
            bgcolor=self.surface_alt,
            border_color=self.border,
            focused_border_color=self.accent,
            color=self.text,
            multiline=True,
            min_lines=2,
            max_lines=4,
            dense=False,
        )

        self.field_variables = ft.TextField(
            label="Variables (separadas por comas)",
            hint_text="Ejemplo: x, y",
            bgcolor=self.surface_alt,
            border_color=self.border,
            focused_border_color=self.accent,
            color=self.text,
        )

        self.field_initial_guess = ft.TextField(
            label="Punto inicial x0 (opcional, comas)",
            hint_text="Ejemplo: 0, 0",
            bgcolor=self.surface_alt,
            border_color=self.border,
            focused_border_color=self.accent,
            color=self.text,
        )

        # Constraints section
        constraints_header = ft.Row(
            [
                ft.Text("Restricciones", size=14, weight=ft.FontWeight.W_600, color=self.text),
                ft.Container(expand=True),
                ft.FilledButton(
                    "Agregar",
                    icon=ft.Icons.ADD_ROUNDED,
                    style=ft.ButtonStyle(bgcolor=self.accent, color=ft.Colors.BLACK),
                    on_click=self._on_add_constraint,
                ),
            ],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
        )

        self.constraints_column = ft.Column(spacing=8, scroll=ft.ScrollMode.AUTO)

        # Methods
        methods_title = ft.Text("Métodos de optimización", size=14, weight=ft.FontWeight.W_600, color=self.text)
        self.cb_maximize = ft.Checkbox(label="Maximizar (en lugar de minimizar)", value=False, fill_color=self.accent)
        self.cb_grad = ft.Checkbox(label="Gradiente descendente", value=True, fill_color=self.accent)
        self.cb_lagrange = ft.Checkbox(label="Multiplicadores de Lagrange (igualdad)", value=False, fill_color=self.accent)
        self.cb_unconstrained = ft.Checkbox(label="Método sin restricciones (directo)", value=False, fill_color=self.accent)

        run_btn = ft.ElevatedButton(
            text="Resolver",
            icon=ft.Icons.PLAY_ARROW_ROUNDED,
            style=ft.ButtonStyle(
                bgcolor=self.accent,
                color=ft.Colors.BLACK,
                shape=ft.RoundedRectangleBorder(radius=8),
                padding=12,
            ),
            on_click=self._on_solve,
        )

        left = ft.Column(
            [
                title,
                ft.Divider(color=self.border),
                self.field_objective,
                self.field_variables,
                self.field_initial_guess,
                ft.Container(height=6),
                constraints_header,
                self.constraints_column,
                ft.Container(height=6),
                ft.Container(
                    content=ft.Column(
                        [
                            methods_title,
                            self.cb_maximize,
                            self.cb_grad,
                            self.cb_lagrange,
                            self.cb_unconstrained,
                        ],
                        spacing=6,
                    ),
                    padding=10,
                    bgcolor=self.surface_alt,
                    border_radius=8,
                    border=ft.border.all(1, self.border),
                ),
                ft.Container(height=10),
                run_btn,
            ],
            spacing=12,
            expand=True,
        )
        return left

    def _build_results_panel(self) -> ft.Column:
        title = ft.Text("Resultados", size=16, weight=ft.FontWeight.BOLD, color=self.text)

        self.results_column = ft.Column(spacing=10, scroll=ft.ScrollMode.ALWAYS)
        # Wrap in container for better layout
        self.results_container = ft.Container(
            content=self.results_column,
            height=300,  # Fixed height for scroll
        )

        self.plot_container = ft.Container(
            content=ft.Text("El gráfico aparecerá aquí.", color=self.muted),
            bgcolor=self.surface,
            height=420,
            border_radius=8,
            border=ft.border.all(1, self.border),
            padding=10,
            alignment=ft.alignment.center,
        )

        # Tabs for per-method details (created dynamically)
        self.tabs_container = ft.Tabs(tabs=[], expand=1, selected_index=0)

        right = ft.Column(
            [
                title,
                ft.Divider(color=self.border),
                ft.Text("Comparación de métodos", color=self.muted, size=12),
                self.results_container,
                ft.Divider(color=self.border),
                ft.Text("Visualización", color=self.muted, size=12),
                self.plot_container,
                ft.Divider(color=self.border),
                ft.Text("Detalle por método", color=self.muted, size=12),
                self.tabs_container,
            ],
            spacing=12,
            expand=True,
        )
        return right

    # ---------- Event Handlers ----------

    def _on_add_constraint(self, e):
        self._add_constraint_row()

    def _add_constraint_row(self, preset: str = ""):
        tf = ft.TextField(
            value=preset,
            label="Restricción (<=, >=, =)",
            hint_text="Ejemplos: x**2 + y**2 <= 1   o   x + y = 1",
            bgcolor=self.surface_alt,
            border_color=self.border,
            focused_border_color=self.accent,
            color=self.text,
            dense=False,
            width=380,
        )

        def remove(_):
            # remove this row
            for i, (row, field) in enumerate(self.constraint_rows):
                if field == tf:
                    self.constraints_column.controls.remove(row)
                    self.constraint_rows.pop(i)
                    break
            self.page.update()

        btn_remove = ft.IconButton(
            icon=ft.Icons.DELETE_ROUNDED,
            icon_color=self.error,
            tooltip="Eliminar restricción",
            on_click=remove,
        )

        row = ft.Row([tf, btn_remove], alignment=ft.MainAxisAlignment.START)
        self.constraint_rows.append((row, tf))
        self.constraints_column.controls.append(row)
        self.page.update()

    def _on_solve(self, e):
        try:
            self._clear_results("Resolviendo...")

            variables, problem, x0, selection = self._collect_problem_from_inputs()

            # Execute selected methods
            results: Dict[str, SolverResult] = {}
            method_tabs: List[ft.Tab] = []

            if selection.gradient:
                gd = GradientDescentSolver()
                results["Gradiente Descendente"] = gd.solve(problem, x0)

            if selection.lagrange:
                lg = LagrangeSolver()
                results["Lagrange"] = lg.solve(problem, x0)

            if selection.unconstrained:
                uc = UnconstrainedDirectSolver()
                results["Sin Restricciones"] = uc.solve(problem, x0)

            # Display summary table
            self._render_summary_table(variables, results)

            # Pick best successful solution for plotting
            best = self._best_result(results)
            if best is not None:
                method_name, res = best
                b64 = self.plotter.render_auto(problem.objective_func, variables, res.x.tolist() if res.x is not None else None)
                if b64:
                    self.plot_container.content = ft.Image(src=b64, fit=ft.ImageFit.CONTAIN)
                else:
                    self.plot_container.content = ft.Text("No se pudo renderizar el gráfico.", color=self.warn)
            else:
                self.plot_container.content = ft.Text("Sin solución para graficar.", color=self.warn)

            # Per-method tabs with details and constraints verification
            tabs = []
            for name, res in results.items():
                tab_content = self._method_detail_tab(name, res, problem)
                tabs.append(ft.Tab(text=name, content=tab_content))
            self.tabs_container.tabs = tabs
            self.tabs_container.selected_index = 0 if tabs else None

            self.page.update()

        except Exception as ex:
            self.results_column.controls.append(
                ft.Text(f"Error: {ex}", color=self.error)
            )
            self.plot_container.content = ft.Text("Error durante el cálculo.", color=self.error)
            self.page.update()

    # ---------- Data Handling ----------

    def _collect_problem_from_inputs(self) -> Tuple[List[str], Problem, Optional[np.ndarray], MethodSelection]:
        objective_str = (self.field_objective.value or "").strip()
        variables_str = (self.field_variables.value or "").strip()
        x0_str = (self.field_initial_guess.value or "").strip()

        if not objective_str or not variables_str:
            raise ValueError("Ingrese la función objetivo y las variables.")

        variables = self.parser.parse_variables(variables_str)
        obj_expr, obj_func, grad_func = self.parser.parse_objective(objective_str, variables)

        # Handle maximization by negating the objective
        maximize = bool(self.cb_maximize.value) if self.cb_maximize else False
        if maximize:
            obj_expr = -obj_expr
            original_obj_func = obj_func
            original_grad_func = grad_func
            obj_func = lambda *args: -original_obj_func(*args)
            grad_func = lambda *args: -original_grad_func(*args)

        constraints_raw: List[str] = []
        for _, tf in self.constraint_rows:
            val = (tf.value or "").strip()
            if val:
                constraints_raw.append(val)

        parsed_constraints = self.parser.parse_constraints(constraints_raw, variables)
        problem = Problem(
            variables=variables,
            objective_expr=obj_expr,
            objective_func=obj_func,
            gradient_func=grad_func,
            constraints=parsed_constraints,
        )

        x0 = None
        if x0_str:
            try:
                x0_vals = [float(v.strip()) for v in x0_str.split(",") if v.strip()]
                if len(x0_vals) != len(variables):
                    raise ValueError(f"El tamaño de x0 debe ser {len(variables)}.")
                x0 = np.asarray(x0_vals, dtype=float)
            except Exception:
                raise ValueError("x0 inválido. Use números separados por comas.")

        selection = MethodSelection(
            maximize=maximize,
            gradient=bool(self.cb_grad.value),
            lagrange=bool(self.cb_lagrange.value),
            unconstrained=bool(self.cb_unconstrained.value),
        )

        if not (selection.gradient or selection.lagrange or selection.unconstrained):
            raise ValueError("Seleccione al menos un método.")

        return variables, problem, x0, selection

    # ---------- Rendering helpers ----------

    def _clear_results(self, message: str):
        self.results_column.controls.clear()
        self.results_column.controls.append(ft.Text(message, color=self.muted))
        self.plot_container.content = ft.Text("Procesando...", color=self.muted)
        self.tabs_container.tabs = []
        self.page.update()

    def _best_result(self, results: Dict[str, SolverResult]) -> Optional[Tuple[str, SolverResult]]:
        best_name = None
        best_res: Optional[SolverResult] = None
        for name, res in results.items():
            if res.x is None or res.fun is None:
                continue
            if best_res is None or (res.success and not best_res.success) or (res.fun < (best_res.fun or np.inf)):
                best_name = name
                best_res = res
        if best_name is None or best_res is None:
            return None
        return best_name, best_res

    def _render_summary_table(self, variables: List[str], results: Dict[str, SolverResult]):
        self.results_column.controls.clear()

        headers = [
            ft.DataColumn(ft.Text("Método", color=self.text)),
            ft.DataColumn(ft.Text("Éxito", color=self.text)),
            ft.DataColumn(ft.Text("f* (valor)", color=self.text)),
            ft.DataColumn(ft.Text("x* (vector)", color=self.text)),
            ft.DataColumn(ft.Text("Mensaje", color=self.text)),
        ]
        rows: List[ft.DataRow] = []

        for name, res in results.items():
            xval = "-"
            if res.x is not None:
                try:
                    xlist = [f"{float(v):.6f}" for v in list(res.x)]
                    xval = "[" + ", ".join(xlist) + "]"
                except Exception:
                    xval = "-"

            fval = "-" if res.fun is None else f"{res.fun:.6f}"
            success_str = "Sí" if res.success else "No"

            rows.append(
                ft.DataRow(
                    cells=[
                        ft.DataCell(ft.Text(name, color=self.text)),
                        ft.DataCell(ft.Text(success_str, color=self.text)),
                        ft.DataCell(ft.Text(fval, color=self.text)),
                        ft.DataCell(ft.Text(xval, color=self.text)),
                        ft.DataCell(ft.Text(res.message or "", color=self.muted)),
                    ]
                )
            )

        table = ft.DataTable(
            columns=headers,
            rows=rows,
            heading_row_color=self.surface,
            data_row_color={"hovered": self.surface_alt},
            divider_thickness=1,
            border=ft.border.all(1, self.border),
        )
        self.results_column.controls.append(table)

    def _constraints_verification_view(self, checks: List[ConstraintCheck]) -> ft.Column:
        if not checks:
            return ft.Column([ft.Text("Sin restricciones para verificar.", color=self.muted)], spacing=6)

        items: List[ft.Control] = []
        for c in checks:
            status_color = self.accent if c.satisfied else self.error
            kind = "eq" if c.type == "eq" else "ineq"
            items.append(
                ft.Container(
                    content=ft.Row(
                        [
                            ft.Icon(ft.Icons.CHECK_CIRCLE_OUTLINE_ROUNDED if c.satisfied else ft.Icons.ERROR_OUTLINE_ROUNDED, color=status_color),
                            ft.Text(f"{c.original}  |  tipo: {kind}  |  valor: {c.value:.6e}", color=self.text),
                        ],
                        alignment=ft.MainAxisAlignment.START,
                    ),
                    padding=6,
                    border=ft.border.all(1, self.border),
                    border_radius=6,
                    bgcolor=self.surface,
                )
            )
        return ft.Column(items, spacing=6)

    def _method_detail_tab(self, name: str, res: SolverResult, problem: Problem) -> ft.Container:
        content_controls: List[ft.Control] = []

        # Header
        content_controls.append(
            ft.Text(f"Resumen - {name}", size=14, weight=ft.FontWeight.W_600, color=self.text)
        )
        content_controls.append(ft.Divider(color=self.border))

        # Core info
        xval = "-"
        if res.x is not None:
            try:
                xlist = [f"{float(v):.6f}" for v in list(res.x)]
                xval = "[" + ", ".join(xlist) + "]"
            except Exception:
                pass
        fval = "-" if res.fun is None else f"{res.fun:.6f}"

        info = ft.Column(
            [
                ft.Text(f"Éxito: {'Sí' if res.success else 'No'}", color=self.text),
                ft.Text(f"f*: {fval}", color=self.text),
                ft.Text(f"x*: {xval}", color=self.text),
                ft.Text(f"Mensaje: {res.message or ''}", color=self.muted, size=12),
            ],
            spacing=4,
        )
        content_controls.append(info)
        content_controls.append(ft.Container(height=6))

        # Constraints verification
        checks_view: ft.Control
        if res.x is not None and len(problem.constraints) > 0:
            checks = problem.verify_constraints(res.x)
            checks_view = self._constraints_verification_view(checks)
        else:
            checks_view = ft.Text("Sin verificación de restricciones.", color=self.muted, size=12)

        content_controls.append(ft.Text("Verificación de restricciones", color=self.text, weight=ft.FontWeight.W_600))
        content_controls.append(checks_view)

        return ft.Container(
            content=ft.Column(content_controls, spacing=8),
            padding=10,
            bgcolor=self.surface_alt,
            border_radius=6,
        )


def app_main(page: ft.Page):
    app = OptimizationAppUI()
    app.build(page)