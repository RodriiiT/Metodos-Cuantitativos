from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import minimize

from ProyectoFinal.core.problem import Problem
from .base import OptimizationSolver, SolverResult


class UnconstrainedDirectSolver(OptimizationSolver):
    """
    Unconstrained direct minimization using several SciPy algorithms and
    selecting the best successful result (or the best objective value).
    Methods tried: BFGS, Nelder-Mead, Powell.
    """
    name = "Método Sin Restricciones"

    def __init__(self, methods: Optional[List[str]] = None, max_iters: int = 1000) -> None:
        self.methods = methods or ["BFGS", "Nelder-Mead", "Powell"]
        self.max_iters = max_iters

    def _run_method(self, problem: Problem, x0: np.ndarray, method: str):
        # Provide jac when method supports it
        kwargs: Dict[str, Any] = {"method": method, "options": {"maxiter": self.max_iters, "disp": False}}
        if method.upper() in {"BFGS"}:
            kwargs["jac"] = lambda z: problem.gradient(z)
        res = minimize(lambda z: problem.evaluate(z), x0=x0, **kwargs)
        return res

    def solve(self, problem: Problem, x0: Optional[np.ndarray] = None) -> SolverResult:
        if len(problem.constraints) > 0:
            return SolverResult(
                success=False,
                x=None,
                fun=None,
                message="El método sin restricciones no admite restricciones.",
                method=self.name,
                meta={},
            )

        x0v = problem.initial_guess(None if x0 is None else list(x0))
        best = None
        history: List[Tuple[str, Any]] = []

        for m in self.methods:
            try:
                res = self._run_method(problem, x0v, m)
                history.append((m, {"success": bool(res.success), "fun": float(getattr(res, "fun", np.inf))}))
                if best is None:
                    best = (m, res)
                else:
                    # Prefer successful lower fun; if both fail, pick lower fun anyway
                    prev_m, prev = best
                    prev_success = bool(getattr(prev, "success", False))
                    curr_success = bool(getattr(res, "success", False))
                    prev_fun = float(getattr(prev, "fun", np.inf))
                    curr_fun = float(getattr(res, "fun", np.inf))

                    if (curr_success and not prev_success) or (curr_fun < prev_fun):
                        best = (m, res)
            except Exception as e:
                history.append((m, {"success": False, "error": str(e)}))
                continue

        if best is None:
            return SolverResult(
                success=False,
                x=None,
                fun=None,
                message="No se pudo encontrar una solución con los métodos disponibles.",
                method=self.name,
                meta={"history": history},
            )

        m, res = best
        return SolverResult(
            success=bool(res.success),
            x=np.asarray(res.x, dtype=float),
            fun=float(res.fun) if hasattr(res, "fun") else None,
            message=str(res.message),
            method=f"{self.name} ({m})",
            meta={"nit": getattr(res, "nit", None), "status": getattr(res, "status", None), "history": history},
        )