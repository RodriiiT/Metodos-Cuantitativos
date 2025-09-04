from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple
import re
import sympy as sp
from sympy import symbols, sympify, lambdify, Matrix


@dataclass
class ConstraintSpec:
    """
    Restricción parseada lista para evaluación numérica (compatible con SciPy).
    - Para 'ineq': expr(x) >= 0
    - Para 'eq'  : expr(x) = 0
    """
    original: str
    type: str                # 'eq' | 'ineq'
    expr_str: str            # cadena sympy normalizada (ej. derecha - izquierda para <=)
    expr: sp.Expr            # expresión sympy
    func: Callable[..., float]  # función compatible con numpy f(*vars) -> float


class ExpressionParser:
    """
    Parser centralizado para variables, función objetivo, restricciones y gradiente.
    Maneja casos comunes de multiplicación implícita para ser amigable con el usuario:
      - 2x  -> 2*x
      - x(y + 1) -> x*(y + 1)
      - 2(x + y) -> 2*(x + y)
      - (x + 1)y -> (x + 1)*y
    """
    def _normalize_implicit_multiplication(self, s: str, variables: List[str]) -> str:
        # Insert * between number and opening parenthesis: 2(x+1) -> 2*(x+1)
        s = re.sub(r'(\d)\s*\(', r'\1*(', s)
        # Insert * between ) and number: (x+1)2 -> (x+1)*2
        s = re.sub(r'\)\s*(\d)', r')*\1', s)

        # Insert * between any variable and opening parenthesis: x(y+1) -> x*(y+1)
        for var in sorted(variables, key=len, reverse=True):
            var_escaped = re.escape(var)
            s = re.sub(rf'({var_escaped})\s*\(', r'\1*(', s)
            # Insert * between ) and variable: (x+1)y -> (x+1)*y
            s = re.sub(rf'\)\s*({var_escaped})', r')*\1', s)
            # Insert * between number and variable: 2x -> 2*x
            s = re.sub(rf'(\d)\s*({var_escaped})', r'\1*\2', s)
            # Insert * between variable and number: x2 -> x*2
            # Caution: if users intend var names like x2, discourage via variables input. We assume simple var names.
            s = re.sub(rf'({var_escaped})\s*(\d)', r'\1*\2', s)

        # Insert * between adjacent variables: xy -> x*y (do pairwise)
        # Use variables list to avoid touching function names like sin, cos.
        if len(variables) > 1:
            for i, vi in enumerate(variables):
                vi_e = re.escape(vi)
                for vj in variables[i+1:]:
                    vj_e = re.escape(vj)
                    s = re.sub(rf'({vi_e})\s*({vj_e})', r'\1*\2', s)
                    s = re.sub(rf'({vj_e})\s*({vi_e})', r'\1*\2', s)

        return s

    def parse_variables(self, variables_str: str) -> List[str]:
        if not variables_str:
            raise ValueError("Debe ingresar las variables separadas por comas.")
        variables = [v.strip() for v in variables_str.split(',') if v.strip()]
        if not variables:
            raise ValueError("Lista de variables vacía.")
        # Validación básica: deben ser símbolos válidos de sympy (letras, guion bajo, dígitos sin empezar con dígito)
        for v in variables:
            if not re.match(r'^[A-Za-z_]\w*$', v):
                raise ValueError(f"Nombre de variable inválido: '{v}'. Use identificadores como x, y, z, x1, etc.")
        return variables

    def parse_objective(self, expr_str: str, variables: List[str]) -> Tuple[sp.Expr, Callable, Callable]:
        if not expr_str or not expr_str.strip():
            raise ValueError("Debe ingresar la función objetivo.")
        norm = self._normalize_implicit_multiplication(expr_str.strip(), variables)
        syms = symbols(variables)
        try:
            expr = sympify(norm)
        except Exception as e:
            raise ValueError(f"Error al interpretar la función objetivo: {e}")

        func = lambdify(syms, expr, 'numpy')
        # Gradiente como matriz columna para forma consistente
        grad_vec = Matrix([sp.diff(expr, s) for s in syms])
        grad_func = lambdify(syms, grad_vec, 'numpy')
        return expr, func, grad_func

    def parse_constraints(self, constraints_strs: List[str], variables: List[str]) -> List[ConstraintSpec]:
        parsed: List[ConstraintSpec] = []
        syms = symbols(variables)

        for raw in constraints_strs:
            if not raw or not raw.strip():
                continue
            s = self._normalize_implicit_multiplication(raw.strip(), variables)

            # Determinar tipo y normalizar a g(x) >= 0 o h(x) = 0 compatible con SciPy
            ctype: str
            expr_str: str
            if '<=' in s:
                parts = s.split('<=')
                if len(parts) != 2:
                    raise ValueError(f"Restricción inválida: '{raw}'")
                left = parts[0].strip()
                right = parts[1].strip()
                # Convert f(x) <= c  ->  c - f(x) >= 0
                expr_str = f"({right}) - ({left})"
                ctype = 'ineq'
            elif '>=' in s:
                parts = s.split('>=')
                if len(parts) != 2:
                    raise ValueError(f"Restricción inválida: '{raw}'")
                left = parts[0].strip()
                right = parts[1].strip()
                # Convert f(x) >= c  ->  f(x) - c >= 0
                expr_str = f"({left}) - ({right})"
                ctype = 'ineq'
            elif '=' in s and '!=' not in s:
                parts = s.split('=')
                if len(parts) != 2:
                    raise ValueError(f"Restricción inválida: '{raw}'")
                left = parts[0].strip()
                right = parts[1].strip()
                # Convert f(x) = c   ->  f(x) - c = 0
                expr_str = f"({left}) - ({right})"
                ctype = 'eq'
            else:
                raise ValueError(f"Tipo de restricción no reconocido en '{raw}'. Use <=, >= o =.")

            try:
                expr = sympify(expr_str)
                func = lambdify(syms, expr, 'numpy')
            except Exception as e:
                raise ValueError(f"Error al interpretar la restricción '{raw}': {e}")

            parsed.append(ConstraintSpec(original=raw, type=ctype, expr_str=expr_str, expr=expr, func=func))

        return parsed