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
        current_point = x0.copy().astype(float)
        meta: Dict[str, Any] = {"iterations": 0}

        for i in range(self.max_iters):
            grad_vals = problem.gradient(current_point)
            next_point = current_point - self.init_step * grad_vals  # init_step as learning_rate

            if np.linalg.norm(next_point - current_point) < self.tol:
                current_point = next_point
                meta["iterations"] = i + 1
                fx = problem.evaluate(current_point)
                return SolverResult(
                    success=True,
                    x=current_point,
                    fun=fx,
                    message="Convergencia por cambio pequeño en el punto",
                    method=self.name + " (sin restricciones)",
                    meta=meta,
                )

            current_point = next_point

        meta["iterations"] = self.max_iters
        fx = problem.evaluate(current_point)
        return SolverResult(
            success=False,
            x=current_point,
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