import sympy as sp

def lagrange_optimization(f, constraints, vars):
    """
    Resuelve un problema de optimización con restricciones usando multiplicadores de Lagrange.
    
    Parámetros:
    - f : función objetivo (sympy expression)
    - constraints : lista de restricciones de igualdad [g1, g2, ...] (sympy expressions)
    - vars : lista de variables [x, y, ...]
    
    Retorna:
    - soluciones: lista de diccionarios con los valores de las variables y los multiplicadores
    """
    
    # Definir multiplicadores de Lagrange
    lambdas = sp.symbols(f'lam0:{len(constraints)}')
    
    # Construir Lagrangiano
    L = f + sum(lambdas[i] * constraints[i] for i in range(len(constraints)))
    
    # Condiciones de primer orden
    eqs = [sp.diff(L, v) for v in vars] + [g for g in constraints]
    
    # Resolver
    soluciones = sp.solve(eqs, vars + list(lambdas), dict=True)
    
    return soluciones


def classify_solutions(f, solutions, vars):
    """
    Clasifica soluciones como máximo o mínimo evaluando f.
    """
    valores = []
    for sol in solutions:
        val = f.subs(sol)
        valores.append((sol, sp.N(val)))
    
    # Ordenar por valor de f
    valores.sort(key=lambda x: x[1])
    
    print("\n--- Clasificación ---")
    print("Mínimo encontrado:")
    print(valores[0][0], " -> f =", valores[0][1])
    
    print("\nMáximo encontrado:")
    print(valores[-1][0], " -> f =", valores[-1][1])
    
    return valores


# ===========================
# EJEMPLO DE USO
# ===========================

# Definir variables
x, y = sp.symbols('x y', real=True)

# Función objetivo
f = 4*x*y   # Ejemplo: minimizar la distancia al origen
a = 2
b= 3
# Restricciones
g1 = (x**2)/(a**2) + (y**2)/(b**2) - 1   # recta

# Resolver
soluciones = lagrange_optimization(f, [g1], [x, y])

# Mostrar resultados
for sol in soluciones:
    x_opt = sol[x]
    y_opt = sol[y]
    f_val = f.subs({x: x_opt, y: y_opt})
    print(f"Solución encontrada: x={x_opt}, y={y_opt}, f={f_val}")

# Clasificar soluciones
classify_solutions(f, soluciones, [x, y])
