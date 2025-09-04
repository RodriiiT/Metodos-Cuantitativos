from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import sympy as sp
from scipy.optimize import minimize

from ProyectoFinal.core.problem import Problem
from .base import OptimizationSolver, SolverResult


class LagrangeSolver(OptimizationSolver):
    """
    Lagrange multipliers method for equality-constrained problems.
    - Primary approach: symbolic stationarity + feasibility (SymPy) using nsolve/solve
    - Fallback: numeric SLSQP with equality constraints only
    Notes:
      - Only equality constraints are handled as classical Lagrange method (h_i(x) = 0)
      - Inequality constraints are ignored in the symbolic phase and handled only via SLSQP fallback.
    """
    name = "Método de Lagrange"

    def __init__(self, tol: float = 1e-6, max_iters: int = 200) -> None:
        self.tol = tol
        self.max_iters = max_iters

    def _eq_constraints(self, problem: Problem):
        eqs = [c for c in problem.constraints if c.type == "eq"]
        return eqs

    def _symbolic_attempt(self, problem: Problem, x0: Optional[np.ndarray]) -> Optional[SolverResult]:
        eqs = self._eq_constraints(problem)
        if not eqs:
            return None

        # Build Lagrangian: L(x, lambda) = f(x) + sum_i lam_i * h_i(x)
        x_syms = sp.symbols(problem.variables)
        lam_syms = sp.symbols(f"lam0:{len(eqs)}")
        f_expr = problem.objective_expr
        h_exprs = [e.expr for e in eqs]

        L = f_expr
        for lam, h in zip(lam_syms, h_exprs):
            L = L + lam * h

        # Stationarity conditions: dL/dx = 0
        stationarity = [sp.diff(L, xi) for xi in x_syms]
        feasibility = h_exprs[:]  # h_i(x) = 0

        equations = stationarity + feasibility
        unknowns = list(x_syms) + list(lam_syms)

        # Initial guess for nsolve: [x0, zeros for lambdas]
        if x0 is None:
            x0 = np.zeros(len(problem.variables), dtype=float)
        guess = list(np.asarray(x0, dtype=float).reshape(-1)) + [0.0] * len(lam_syms)

        # Try nsolve first
        try:
            sol_vec = sp.nsolve(equations, unknowns, guess, tol=self.tol, maxsteps=100)
            sol_vec = list(map(float, sol_vec))
            x_sol = np.array(sol_vec[: problem.dim], dtype=float)
            f_val = float(problem.evaluate(x_sol))
            return SolverResult(
                success=True,
                x=x_sol,
                fun=f_val,
                message="Solución encontrada con multiplicadores de Lagrange (nsolve)",
                method=self.name,
                meta={"approach": "symbolic_nsolve"},
            )
        except Exception:
            # Try algebraic solve (may return multiple solutions)
            try:
                sol_set = sp.solve(equations, unknowns, dict=True)
                if not sol_set:
                    return None
                best_x = None
                best_f = None
                for sol in sol_set:
                    try:
                        x_sol = np.array([float(sol[xi]) for xi in x_syms], dtype=float)
                        f_val = float(problem.evaluate(x_sol))
                        if best_f is None or f_val < best_f:
                            best_f = f_val
                            best_x = x_sol
                    except Exception:
                        continue
                if best_x is None:
                    return None
                return SolverResult(
                    success=True,
                    x=best_x,
                    fun=best_f,
                    message="Solución encontrada con multiplicadores de Lagrange (solve)",
                    method=self.name,
                    meta={"approach": "symbolic_solve", "num_candidates": len(sol_set)},
                )
            except Exception:
                return None

    def _slsqp_fallback(self, problem: Problem, x0: np.ndarray) -> SolverResult:
        # Equality constraints only
        cons = []
        for c in problem.constraints:
            if c.type == "eq":
                cons.append({"type": "eq", "fun": lambda z, cf=c: float(cf.func(*list(np.asarray(z, dtype=float))))})

        if not cons:
            # No equality constraints: nothing to do here
            return SolverResult(
                success=False,
                x=x0,
                fun=None,
                message="No hay restricciones de igualdad para aplicar Lagrange",
                method=self.name,
                meta={"approach": "slsqp", "note": "no_eq_constraints"},
            )

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
            x=np.asarray(res.x, dtype=float),
            fun=float(res.fun) if hasattr(res, "fun") else None,
            message=str(res.message),
            method=self.name + " (SLSQP fallback)",
            meta={"nit": getattr(res, "nit", None), "status": getattr(res, "status", None)},
        )

    def solve(self, problem: Problem, x0: Optional[np.ndarray] = None) -> SolverResult:
        if len(problem.constraints) == 0:
            return SolverResult(
                success=False,
                x=None,
                fun=None,
                message="El método de Lagrange requiere restricciones de igualdad.",
                method=self.name,
                meta={},
            )

        eqs = self._eq_constraints(problem)
        if not eqs:
            return SolverResult(
                success=False,
                x=None,
                fun=None,
                message="El método de Lagrange solo admite restricciones de igualdad (h(x)=0).",
                method=self.name,
                meta={},
            )

        x0v = problem.initial_guess(None if x0 is None else list(x0))

        # 1) Try symbolic approach
        sym_res = self._symbolic_attempt(problem, x0v)
        if sym_res is not None and sym_res.success:
            return sym_res

        # 2) Fallback to SLSQP numeric
        return self._slsqp_fallback(problem, x0v)