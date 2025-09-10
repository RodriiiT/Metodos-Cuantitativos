from __future__ import annotations

import io
import base64
from typing import Callable, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # needed for 3D projection


class Plotter:
    """
    Generates base64-encoded images for:
      - 1D function plot with optimal point
      - 2D contour plot with optimal point
      - 3D surface plot (if requested explicitly)
    Uses a dark theme consistent with a modern UI.
    """

    def __init__(
        self,
        bg_color: str = "#000000",
        accent_color: str = "#35bcb3",
        text_color: str = "#FFFFFF",
        secondary_color: str = "#1a1a1a",
    ) -> None:
        self.bg_color = bg_color
        self.accent_color = accent_color
        self.text_color = text_color
        self.secondary_color = secondary_color

    def _figure_axes(self, figsize=(10, 8)) -> Tuple[plt.Figure, plt.Axes]:
        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor(self.bg_color)
        ax.set_facecolor(self.bg_color)
        ax.tick_params(colors=self.text_color)
        return fig, ax

    def _to_base64(self, fig: plt.Figure) -> str:
        buffer = io.BytesIO()
        fig.savefig(
            buffer,
            format="png",
            facecolor=self.bg_color,
            edgecolor="none",
            bbox_inches="tight",
            dpi=110,
        )
        buffer.seek(0)
        img_b64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        return f"data:image/png;base64,{img_b64}"

    def plot_1d(
        self,
        func: Callable[..., float],
        variables: List[str],
        optimal_point: Optional[List[float]] = None,
        xlim: Tuple[float, float] = (-10.0, 10.0),
        num: int = 1200,
        title: str = "Función objetivo",
    ) -> Optional[str]:
        try:
            fig, ax = self._figure_axes()
            x = np.linspace(xlim[0], xlim[1], num=num)
            y = func(x)
            ax.plot(x, y, color=self.accent_color, linewidth=2)
            ax.set_xlabel(variables[0], color=self.text_color)
            ax.set_ylabel("f(x)", color=self.text_color)
            ax.grid(True, alpha=0.25)
            ax.set_title(title, color=self.text_color, pad=12)

            if optimal_point is not None and len(optimal_point) >= 1:
                xstar = float(optimal_point[0])
                try:
                    ystar = float(func(xstar))
                    ax.scatter([xstar], [ystar], c="#ff6b6b", s=60, zorder=3)
                    ax.annotate(
                        f"Óptimo: {variables[0]}={xstar:.3f}",
                        xy=(xstar, ystar),
                        xytext=(10, 10),
                        textcoords="offset points",
                        color=self.text_color,
                    )
                except Exception:
                    pass

            return self._to_base64(fig)
        except Exception:
            # In case of evaluation issues, do not crash UI
            return None

    def plot_2d_contour(
        self,
        func: Callable[..., float],
        variables: List[str],
        optimal_point: Optional[List[float]] = None,
        xlim: Tuple[float, float] = (-5.0, 5.0),
        ylim: Tuple[float, float] = (-5.0, 5.0),
        grid: int = 220,
        title: str = "Contorno de la función objetivo",
    ) -> Optional[str]:
        try:
            fig, ax = self._figure_axes()
            x = np.linspace(xlim[0], xlim[1], grid)
            y = np.linspace(ylim[0], ylim[1], grid)
            X, Y = np.meshgrid(x, y)
            Z = func(X, Y)

            # Filled contour + line contours
            cnf = ax.contourf(X, Y, Z, levels=24, cmap="viridis", alpha=0.85)
            cn = ax.contour(X, Y, Z, levels=24, colors=self.accent_color, linewidths=0.6, alpha=0.55)
            ax.set_xlabel(variables[0], color=self.text_color)
            ax.set_ylabel(variables[1], color=self.text_color)
            ax.grid(True, alpha=0.18)
            ax.set_title(title, color=self.text_color, pad=12)
            fig.colorbar(cnf, ax=ax, shrink=0.88)

            if optimal_point is not None and len(optimal_point) >= 2:
                xs, ys = float(optimal_point[0]), float(optimal_point[1])
                ax.scatter([xs], [ys], c="#ff6b6b", s=70, edgecolors="white", linewidths=0.6, zorder=5)
                ax.annotate(
                    f"Óptimo: ({xs:.3f}, {ys:.3f})",
                    xy=(xs, ys),
                    xytext=(10, 10),
                    textcoords="offset points",
                    color=self.text_color,
                )

            return self._to_base64(fig)
        except Exception:
            return None

    def plot_3d_surface(
        self,
        func: Callable[..., float],
        variables: List[str],
        optimal_point: Optional[List[float]] = None,
        xlim: Tuple[float, float] = (-5.0, 5.0),
        ylim: Tuple[float, float] = (-5.0, 5.0),
        grid: int = 120,
        title: str = "Superficie de la función (3D)",
    ) -> Optional[str]:
        try:
            plt.style.use("dark_background")
            fig = plt.figure(figsize=(10, 8))
            fig.patch.set_facecolor(self.bg_color)
            ax = fig.add_subplot(111, projection="3d")
            ax.set_facecolor(self.bg_color)

            x = np.linspace(xlim[0], xlim[1], grid)
            y = np.linspace(ylim[0], ylim[1], grid)
            X, Y = np.meshgrid(x, y)
            Z = func(X, Y)

            surf = ax.plot_surface(X, Y, Z, cmap="viridis", linewidth=0, antialiased=True, alpha=0.95)
            ax.set_xlabel(variables[0], color=self.text_color)
            ax.set_ylabel(variables[1], color=self.text_color)
            ax.set_zlabel("f(x,y)", color=self.text_color)
            ax.set_title(title, color=self.text_color, pad=16)
            fig.colorbar(surf, shrink=0.66, aspect=12)

            if optimal_point is not None and len(optimal_point) >= 2:
                xs, ys = float(optimal_point[0]), float(optimal_point[1])
                try:
                    zs = float(func(xs, ys))
                    ax.scatter(xs, ys, zs, c="#ff6b6b", s=40)
                except Exception:
                    pass

            return self._to_base64(fig)
        except Exception:
            return None

    def render_auto(
        self,
        func: Callable[..., float],
        variables: List[str],
        optimal_point: Optional[List[float]] = None,
    ) -> Optional[str]:
        """
        Choose the best plot automatically:
          - 1 variable: 1D line plot
          - 2 variables: 2D contour
          - >2 variables: not available -> None
        """
        n = len(variables)
        if n == 1:
            return self.plot_1d(func, variables, optimal_point)
        if n == 2:
            return self.plot_2d_contour(func, variables, optimal_point)
        return None