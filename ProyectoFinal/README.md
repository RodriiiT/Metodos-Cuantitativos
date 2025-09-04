# Calculadora de Optimización No Lineal (UI moderna y modular)

Aplicación en Python con interfaz moderna (Flet), arquitectura modular y orientación a objetos para resolver problemas de optimización no lineal con y sin restricciones. Incluye comparación entre métodos, verificación de restricciones y visualización 1D/2D/3D.

Arquitectura principal:
- Núcleo (core): parseo simbólico/numérico, modelo del problema y graficador.
- Solvers (solvers): algoritmos de optimización.
- Interfaz (ui): UI profesional en Flet.

Entradas:
- Función objetivo no lineal (texto).
- Variables (lista separada por comas).
- Restricciones opcionales (<=, >=, =).
- Punto inicial opcional x0.

Salidas:
- Vector óptimo x* y valor f*.
- Mensajes de convergencia/estado.
- Verificación de restricciones.
- Comparación de métodos y gráficos.

## Características

- Métodos de optimización:
  - Gradiente descendente (con backtracking Line Search Armijo).
    - Sin restricciones: método propio de GD.
    - Con restricciones: vía SLSQP (SciPy) respetando restricciones (eq/ineq).
  - Multiplicadores de Lagrange (solo restricciones de igualdad):
    - Intento simbólico con SymPy (nsolve/solve).
    - Fallback numérico con SLSQP solo de igualdad.
  - Método sin restricciones (minimización directa):
    - Intenta BFGS, Nelder–Mead y Powell; escoge el mejor resultado.

- Visualización:
  - 1 variable: curva 1D con punto óptimo.
  - 2 variables: contorno 2D y superficie 3D opcional (interno).
  - >2 variables: omite gráficos (muestra resultados numéricos).

- Comparación de métodos:
  - Tabla con éxito, f*, x*, mensaje.
  - Pestañas por método con detalle y verificación de restricciones.

- Interfaz moderna:
  - Tema oscuro, controles limpios, sin emojis.
  - Separación clara de entradas, resultados y gráficos.
  - Soporte para maximización (checkbox para cambiar entre min/max).
  - Scroll vertical en resultados para manejar grandes cantidades de datos.

## Estructura del proyecto

- [ProyectoFinal/aplicacion.py](ProyectoFinal/aplicacion.py:1) — punto de entrada de la app Flet.
- Núcleo:
  - [ProyectoFinal/núcleo/parser.py](ProyectoFinal/núcleo/parser.py:1) — parseo de variables, función objetivo, restricciones y gradiente.
  - [ProyectoFinal/núcleo/problem.py](ProyectoFinal/núcleo/problem.py:1) — clase Problem y verificación de restricciones.
  - [ProyectoFinal/núcleo/plotter.py](ProyectoFinal/núcleo/plotter.py:1) — generador de gráficos a base64 (1D/2D/3D).
- Resolvedores:
  - [ProyectoFinal/resolvedores/base.py](ProyectoFinal/resolvedores/base.py:1) — interfaz base y resultado del solver.
  - [ProyectoFinal/resolvedores/gradient_descent.py](ProyectoFinal/resolvedores/gradient_descent.py:1) — gradiente descendente + SLSQP si hay restricciones.
  - [ProyectoFinal/resolvedores/lagrange.py](ProyectoFinal/resolvedores/lagrange.py:1) — método de Lagrange (igualdades) + fallback SLSQP.
  - [ProyectoFinal/resolvedores/unconstrained.py](ProyectoFinal/resolvedores/unconstrained.py:1) — métodos directos (BFGS/Nelder–Mead/Powell).
- Interfaz:
  - [ProyectoFinal/interfaz/main_view.py](ProyectoFinal/interfaz/main_view.py:1) — UI moderna (Flet).
  - [ProyectoFinal/interfaz/__init__.py](ProyectoFinal/interfaz/__init__.py:1) — exporta OptimizationAppUI y app_main.
- Compatibilidad con el ejecutable original:
  - [ProyectoFinal/calculadora_optimizacion.py](ProyectoFinal/calculadora_optimizacion.py:1) — ahora lanza la UI modular.

## Requisitos

- Python 3.10+
- Paquetes:
  - flet
  - numpy
  - scipy
  - sympy
  - matplotlib

Puede instalarse con:

```bash
pip install flet numpy scipy sympy matplotlib
# o usando el archivo de requerimientos si lo creas:
# pip install -r ProyectoFinal/requirements.txt
```

## Ejecución

Desde el directorio raíz del proyecto (`/workspace/Metodos-Cuantitativos`):

```bash
PYTHONPATH=/workspace/Metodos-Cuantitativos python ProyectoFinal/aplicacion.py
```

Esto iniciará la aplicación web de Flet en el navegador.

**Nota importante**: Es necesario establecer `PYTHONPATH` para que Python encuentre los módulos del proyecto correctamente.

Opción alternativa (vía lanzador legado):
```bash
PYTHONPATH=/workspace/Metodos-Cuantitativos python ProyectoFinal/calculadora_optimizacion.py
```

## Uso

1) En la sección Modelo:
- Función objetivo: por ejemplo, `x**2 + y**2`.
- Variables: `x, y`.
- x0 (opcional): `0, 0`.

2) Restricciones (opcionales):
- Ejemplos:
  - `x + y = 1`
  - `x**2 + y**2 <= 1`
  - `x - 2 >= 0`

Notas:
- Soporta multiplicación implícita amigable (e.g., `2x` -> `2*x`, `x(y+1)` -> `x*(y+1)`).
- Lagrange solo usa restricciones de igualdad.

3) Seleccione métodos:
- Gradiente descendente (si hay restricciones, usa SLSQP).
- Multiplicadores de Lagrange (igualdad).
- Sin restricciones (directo).

4) Pulse “Resolver”.

5) Revise:
- Tabla de comparación.
- Gráfico (si 1D o 2D).
- Pestañas por método con verificación de restricciones.

## Ejemplos rápidos

- Minimización simple sin restricciones:
  - f(x, y) = `x**2 + y**2`
  - Vars: `x, y`
  - Métodos: “Gradiente descendente” y/o “Sin restricciones”.
  - Resultado esperado: x* ≈ [0, 0], f* ≈ 0.

- Igualdad con Lagrange:
  - f(x, y) = `x**2 + y**2`
  - Restricción: `x + y = 1`
  - Vars: `x, y`
  - Método: “Multiplicadores de Lagrange”.
  - Resultado esperado: x* = y* = 0.5, f* = 0.5.

- Con inecuaciones (SLSQP):
  - f(x, y) = `x**2 + y**2`
  - Restricción: `x**2 + y**2 <= 1`
  - Gradiente descendente activado (con restricciones vía SLSQP).

- Maximizar función:
  - f(x, y) = `-(x**2 + y**2) + 4x + 6y`
  - Activar checkbox "Maximizar"
  - Método: Gradiente descendente
  - Resultado esperado: x* ≈ [2, 3], f* ≈ 13.

- Problema con múltiples restricciones:
  - f(x, y) = `5x + 8y - 0.1x**2 - 0.2y**2`
  - Restricciones: `x + y <= 40`, `x >= 0`, `y >= 0`
  - Método: Gradiente descendente (maneja restricciones automáticamente).

## Diseño y POO

- El modelo del problema está encapsulado en la clase `Problem` con evaluación y gradiente, y conversión a restricciones SciPy.
- Los solvers implementan la interfaz `OptimizationSolver` y retornan `SolverResult`.
- El parseo de expresiones y restricciones se centraliza en `ExpressionParser`.
- El graficador `Plotter` produce imágenes en base64 listas para UI.

## Notas y limitaciones

- La visualización 3D internamente está disponible; la UI prioriza 2D. Puede extenderse fácilmente.
- Lagrange (simbólico) puede fallar para ciertos problemas; se usa fallback SLSQP.
- Para variables > 2, no se renderiza gráfico (se muestran resultados numéricos).

## Licencia

Uso académico.