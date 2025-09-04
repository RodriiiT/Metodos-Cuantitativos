from __future__ import annotations

from typing import Any, Dict, Optional
import numpy as np
from scipy.optimize import minimize

from ProyectoFinal.core.problem import Problem
from .base import OptimizationSolver, SolverResult


class GradientDescentSolver(OptimizationSolver):
    """
    Gradient Descent with backtracking line search (Armijo).
    If constraints are present, falls back to SLSQP using SciPy.
    """
    name = "Gradiente Descendente"

    def __init__(
        self,
        max_iters: int = 500,
        tol: float = 1e-6,
        init_step: float = 1.0,
        backtrack_beta: float = 0.5,
        armijo_c: float = 1e-4,
    ) -> None:
        self.max_iters = max_iters
        self.tol = tol
        self.init_step = init_step
        self.backtrack_beta = backtrack_beta
        self.armijo_c = armijo_c

    def _unconstrained_gd(self, problem: Problem, x0: np.ndarray) -> SolverResult:
        x = x0.copy().astype(float)
        meta: Dict[str, Any] = {"iterations": 0, "grad_norms": []}

        fx = problem.evaluate(x)
        for k in range(self.max_iters):
            g = problem.gradient(x)
            gnorm = float(np.linalg.norm(g))
            meta["grad_norms"].append(gnorm)

            if gnorm < self.tol:
                meta["iterations"] = k
                return SolverResult(
                    success=True,
                    x=x,
                    fun=fx,
                    message="Convergencia por norma del gradiente",
                    method=self.name + " (sin restricciones)",
                    meta=meta,
                )

            # Backtracking line search (Armijo)
            t = self.init_step
            dir = -g
            # Armijo condition: f(x + t*dir) <= f(x) + c*t*grad^T*dir
            while True:
                xn = x + t * dir
                fn = problem.evaluate(xn)
                if fn <= fx + self.armijo_c * t * float(np.dot(g, dir)):
                    break
                t *= self.backtrack_beta
                if t < 1e-12:
                    # Step size vanished
                    meta["iterations"] = k
                    return SolverResult(
                        success=False,
                        x=x,
                        fun=fx,
                        message="Fallo de búsqueda de línea (paso muy pequeño)",
                        method=self.name + " (sin restricciones)",
                        meta=meta,
                    )

            x = xn
            fx = fn

        meta["iterations"] = self.max_iters
        return SolverResult(
            success=False,
            x=x,
            fun=fx,
            message="Máximo de iteraciones alcanzado",
            method=self.name + " (sin restricciones)",
            meta=meta,
        )

    def _constrained_slsqp(self, problem: Problem, x0: np.ndarray) -> SolverResult:
        cons = problem.scipy_constraints()
        # Provide jacobian if available for efficiency
        res = minimize(
            fun=lambda z: problem.evaluate(z),
            x0=x0,
            jac=lambda z: problem.gradient(z),
            method="SLSQP",
            constraints=cons,
            options={"disp": False, "maxiter": self.max_iters},
        )
        return SolverResult(
            success=bool(res.success),
            x=np.asarray(res.x, dtype=float) if res.success else np.asarray(res.x, dtype=float),
            fun=float(res.fun) if hasattr(res, "fun") else None,
            message=str(res.message),
            method=self.name + " (con restricciones via SLSQP)",
            meta={"nit": getattr(res, "nit", None), "status": getattr(res, "status", None)},
        )

    def solve(self, problem: Problem, x0: Optional[np.ndarray] = None) -> SolverResult:
        x0v = problem.initial_guess(None if x0 is None else list(x0))

        if len(problem.constraints) == 0:
            return self._unconstrained_gd(problem, x0v)
        else:
            return self._constrained_slsqp(problem, x0v)