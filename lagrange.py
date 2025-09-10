import sympy as sp
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Módulo de Optimización de Lagrange ---
def lagrange_optimization(f, constraints, vars):
    """
    Resuelve un problema de optimización con restricciones usando multiplicadores de Lagrange.
    
    Parámetros:
    - f (sympy expression): función objetivo.
    - constraints (list): lista de restricciones de igualdad [g1, g2, ...].
    - vars (list): lista de variables [x, y, ...].
    
    Retorna:
    - soluciones: lista de diccionarios con los valores de las variables y los multiplicadores.
    """
    if constraints:
        # Definir multiplicadores de Lagrange
        lambdas = sp.symbols(f'lam0:{len(constraints)}')
        
        # Construir Lagrangiano
        L = f + sum(lambdas[i] * constraints[i] for i in range(len(constraints)))
        
        # Condiciones de primer orden
        eqs = [sp.diff(L, v) for v in vars] + [g for g in constraints]
        
        # Resolver
        soluciones = sp.solve(eqs, vars + list(lambdas), dict=True)
    else:
        # Si no hay restricciones, solo resolvemos para la derivada de la función objetivo
        eqs = [sp.diff(f, v) for v in vars]
        soluciones = sp.solve(eqs, vars, dict=True)

    return soluciones

# --- Módulo de Optimización de Gradiente Descendente ---
def gradient_descent(f, vars, start_point, learning_rate=0.01, iterations=1000, precision=1e-6):
    """
    Encuentra un mínimo local de una función sin restricciones usando gradiente descendente.

    Parámetros:
    - f (sympy expression): La función a minimizar.
    - vars (list): Lista de variables [x, y, ...].
    - start_point (dict): Diccionario con los valores iniciales {'x': 1, 'y': 2}.
    - learning_rate (float): Tasa de aprendizaje, determina el tamaño del paso.
    - iterations (int): Número máximo de iteraciones.
    - precision (float): Precisión para detener el algoritmo.
    
    Retorna:
    - solución (dict): Un diccionario con los valores de las variables en el punto mínimo.
    - puntos (list): Una lista de tuplas con los puntos visitados durante la optimización.
    """
    if len(vars) != len(start_point):
        raise ValueError("El número de variables y el punto de inicio no coinciden.")
    
    # Calcular el gradiente (derivadas parciales)
    gradient = [sp.diff(f, var) for var in vars]
    
    # Convertir expresiones a funciones numéricas para un cálculo rápido
    grad_func = sp.lambdify(vars, gradient, 'numpy')
    f_func = sp.lambdify(vars, f, 'numpy')

    current_point = np.array([start_point[v] for v in vars], dtype=float)
    points = [tuple(current_point)]

    for i in range(iterations):
        # Calcular el valor del gradiente en el punto actual
        grad_vals = np.array(grad_func(*current_point))
        
        # Actualizar el punto
        next_point = current_point - learning_rate * grad_vals
        
        # Detener si el cambio es muy pequeño
        if np.linalg.norm(next_point - current_point) < precision:
            current_point = next_point
            break
            
        current_point = next_point
        points.append(tuple(current_point))

    solution = {var: val for var, val in zip(vars, current_point)}
    return solution, points

def plot_results(f, constraints, solutions, vars, method, path_points=None):
    """
    Grafica la función objetivo, la restricción y las soluciones y la guarda como archivo.
    Solo funciona para problemas con 2 variables (x, y).
    """
    if len(vars) != 2:
        print("\nLa visualización solo está disponible para problemas con dos variables (x, y).")
        return

    x, y = vars[0], vars[1]
    
    f_np = sp.lambdify((x, y), f, 'numpy')
    x_vals = np.linspace(-5, 5, 400)
    y_vals = np.linspace(-5, 5, 400)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = f_np(X, Y)

    plt.figure(figsize=(10, 8))
    plt.contour(X, Y, Z, levels=np.linspace(Z.min(), Z.max(), 30), cmap='viridis', zorder=1)
    plt.colorbar(label='Valor de f')
    plt.xlabel(str(x))
    plt.ylabel(str(y))

    if method == "Lagrange":
        plt.title('Lagrange: Mapa de Contornos y Solución')
        for i, g in enumerate(constraints):
            try:
                plt.contour(X, Y, sp.lambdify((x, y), g, 'numpy')(X, Y), levels=[0], colors='red', linewidths=2, linestyles='--', label=f'Restricción g{i+1}', zorder=2)
            except:
                print(f"Advertencia: No se pudo graficar la restricción {i+1}.")
        sol_x = [sol[x] for sol in solutions if x in sol]
        sol_y = [sol[y] for sol in solutions if y in sol]
        if sol_x and sol_y:
            plt.scatter(sol_x, sol_y, color='red', marker='o', s=100, edgecolors='k', zorder=3, label='Puntos de Solución')

    elif method == "Gradiente":
        plt.title('Gradiente Descendente: Puntos de Solución y Camino')
        plt.scatter(solutions[x], solutions[y], color='red', marker='o', s=100, edgecolors='k', zorder=3, label='Punto de Solución')
        if path_points:
            path_points = np.array(path_points)
            plt.plot(path_points[:, 0], path_points[:, 1], 'r-', linewidth=1, label='Camino de Descenso')

    plt.grid(True)
    plt.legend()
    
    file_path = os.path.join(os.getcwd(), f'optimization_plot_{method.lower()}.png')
    plt.savefig(file_path)
    print(f"\n¡Gráfica guardada exitosamente en: {file_path}!")
    plt.close()

def run_optimization_interactive():
    """
    Solicita al usuario el método y los datos del problema y ejecuta el optimizador.
    """
    print("--- Optimizador de Lagrange vs. Gradiente Descendente ---")
    method_choice = input("Elige un método (1 para Lagrange, 2 para Gradiente Descendente): ")
    
    # 1. Solicitar variables y función objetivo
    var_names_input = input("Introduce las variables separadas por espacios (ej: x y): ")
    var_names = var_names_input.split()
    vars = sp.symbols(var_names)
    if not isinstance(vars, sp.Symbol):
        vars = list(vars)
    else:
        vars = [vars]
        
    f_input = input("Introduce la función objetivo (f): ")
    f = sp.sympify(f_input)
    
    soluciones = None
    plot_points = None

    try:
        if method_choice == '1':
            print("\n--- Método de Lagrange: Optimización con restricciones ---")
            num_constraints = int(input("¿Cuántas restricciones de igualdad tienes?: "))
            constraints = []
            for i in range(num_constraints):
                g_input = input(f"Introduce la restricción g{i+1} (igualada a cero): ")
                constraints.append(sp.sympify(g_input))
            
            print("\nResolviendo el problema con Lagrange...")
            soluciones = lagrange_optimization(f, constraints, vars)

            if not soluciones:
                print("No se encontraron soluciones finitas para este sistema.")
                return

            print("\n--- Soluciones encontradas ---")
            for sol in soluciones:
                try:
                    f_val = f.subs(sol)
                    print(f"Solución: {sol}, Valor de f: {f_val}")
                except (KeyError, ValueError) as e:
                    print(f"Una solución no pudo ser evaluada correctamente: {sol} (Error: {e})")
            
            classify_solutions(f, soluciones)
            plot_results(f, constraints, soluciones, vars, "Lagrange")

        elif method_choice == '2':
            print("\n--- Método de Gradiente Descendente: Optimización sin restricciones ---")
            start_point_input = input("Introduce el punto de inicio como pares clave-valor (ej: x=1, y=2): ")
            start_point_dict = {key.strip(): float(val) for key, val in (pair.split('=') for pair in start_point_input.split(','))}
            start_point_sympy = {sp.Symbol(k): v for k, v in start_point_dict.items()}
            
            print("\nResolviendo el problema con Gradiente Descendente...")
            # Solución de gradiente devuelve una tupla, el primer elemento es la solución y el segundo los puntos
            solucion, plot_points = gradient_descent(f, vars, start_point_sympy)
            soluciones = [solucion] # Se empaqueta en una lista para compatibilidad
            
            print("\n--- Solución encontrada ---")
            f_val = f.subs(solucion)
            print(f"Solución: {solucion}, Valor de f: {f_val}")

            plot_results(f, [], solucion, vars, "Gradiente", plot_points)
        
        else:
            print("Opción no válida. Por favor, elige 1 o 2.")
            return

    except sp.SympifyError:
        print("\nError: Asegúrate de que tus expresiones sean válidas para SymPy.")
    except ValueError as e:
        print(f"\nError: Hubo un problema con la entrada de datos o al resolver el sistema. {e}")
    except Exception as e:
        print(f"\nOcurrió un error inesperado: {e}")

if __name__ == "__main__":
    run_optimization_interactive()