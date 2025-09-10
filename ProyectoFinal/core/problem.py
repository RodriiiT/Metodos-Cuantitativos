from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Any
import numpy as np
import sympy as sp

from .parser import ConstraintSpec


@dataclass
class ConstraintCheck:
    original: str
    type: str           # 'eq' | 'ineq'
    value: float        # evaluated value (>=0 for ineq; =0 for eq)
    satisfied: bool     # within tolerance


class Problem:
    """
    Optimization problem definition with objective, gradient and constraints.
    Designed to provide:
      - numeric evaluation of f(x)
      - gradient evaluation ∇f(x)
      - SciPy-compatible constraint list
      - constraint verification for reporting
    """
    def __init__(
        self,
        variables: List[str],
        objective_expr: sp.Expr,
        objective_func: Callable[..., float],
        gradient_func: Callable[..., Any],   # returns column vector (Matrix) or array
        constraints: Optional[List[ConstraintSpec]] = None,
    ) -> None:
        self.variables = variables
        self.objective_expr = objective_expr
        self.objective_func = objective_func
        self.gradient_func = gradient_func
        self.constraints: List[ConstraintSpec] = constraints or []

    @property
    def dim(self) -> int:
        return len(self.variables)

    def evaluate(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        args = [x[i] for i in range(self.dim)]
        return float(self.objective_func(*args))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Returns gradient as a 1D numpy array of shape (n,)
        """
        x = np.asarray(x, dtype=float)
        args = [x[i] for i in range(self.dim)]
        g = self.gradient_func(*args)
        # Sympy Matrix lambdified may return shape (n, 1) or list-like
        g_arr = np.asarray(g, dtype=float).reshape(-1)
        return g_arr

    def scipy_constraints(self) -> List[Dict[str, Any]]:
        """
        Map internal constraints to SciPy format for minimize:
          - ineq: fun(x) >= 0
          - eq  : fun(x) = 0
        """
        cons: List[Dict[str, Any]] = []
        for c in self.constraints:
            def make_fun(func: Callable[..., float]) -> Callable[[np.ndarray], float]:
                def f(x: np.ndarray) -> float:
                    x = np.asarray(x, dtype=float)
                    args = [x[i] for i in range(self.dim)]
                    return float(func(*args))
                return f

            cons.append({
                "type": c.type,
                "fun": make_fun(c.func),
            })
        return cons

    def verify_constraints(self, x: np.ndarray, tol_eq: float = 1e-6, tol_ineq: float = 1e-8) -> List[ConstraintCheck]:
        """
        Evaluate constraints at x and report satisfaction with tolerances:
          - eq: |h(x)| <= tol_eq
          - ineq: g(x) >= -tol_ineq (allow tiny negative due to numeric noise)
        """
        x = np.asarray(x, dtype=float)
        results: List[ConstraintCheck] = []
        args = [x[i] for i in range(self.dim)]

        for c in self.constraints:
            val = float(c.func(*args))
            if c.type == "eq":
                satisfied = abs(val) <= tol_eq
            elif c.type == "ineq":
                satisfied = val >= -tol_ineq
            else:
                satisfied = False
            results.append(ConstraintCheck(original=c.original, type=c.type, value=val, satisfied=satisfied))

        return results

    def initial_guess(self, provided: Optional[List[float]] = None) -> np.ndarray:
        """
        Get an initial guess vector. If provided is None, use zeros.
        """
        if provided is None:
            return np.zeros(self.dim, dtype=float)
        if len(provided) != self.dim:
            raise ValueError(f"Tamaño de x0 inválido. Se esperaban {self.dim} valores.")
        return np.asarray(provided, dtype=float)