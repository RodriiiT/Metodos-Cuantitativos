from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np

from ProyectoFinal.core.problem import Problem


@dataclass
class SolverResult:
    success: bool
    x: Optional[np.ndarray]
    fun: Optional[float]
    message: str
    method: str
    meta: Dict[str, Any]


class OptimizationSolver:
    """
    Abstract solver interface. Concrete solvers must implement solve().
    """
    name: str = "BaseSolver"

    def solve(self, problem: Problem, x0: Optional[np.ndarray] = None) -> SolverResult:
        raise NotImplementedError("Subclasses must implement solve()")