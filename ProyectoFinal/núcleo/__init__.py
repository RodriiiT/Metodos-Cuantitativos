"""
Módulos núcleo:
- parser: ExpressionParser, ConstraintSpec
- problem: Problem, ConstraintCheck
- plotter: Plotter (renderizado 1D/2D/3D a base64)
"""
from .parser import ExpressionParser, ConstraintSpec  # noqa: F401
from .problem import Problem, ConstraintCheck  # noqa: F401
from .plotter import Plotter  # noqa: F401