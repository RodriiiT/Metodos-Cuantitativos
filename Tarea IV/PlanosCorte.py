import math
import matplotlib.pyplot as plt
import numpy as np

class LinearProgram:
    def __init__(self, c, A, b):
        self.c = c
        self.A = A
        self.b = b
        self.num_vars = len(c)
        self.num_constraints = len(b)
        self.gomory_cuts = []

class Tableau:
    def __init__(self, lp):
        self.lp = lp
        self.original_num_vars = lp.num_vars
        self.rows = lp.num_constraints + 1
        self.cols = lp.num_vars + lp.num_constraints + 1
        self.table = [[0.0] * self.cols for _ in range(self.rows)]
        self.basis = []
        self.tol = 1e-6
        
        # Inicializar fila 0 (función objetivo)
        for j in range(lp.num_vars):
            self.table[0][j] = -lp.c[j]
        
        # Inicializar restricciones y variables de holgura
        for i in range(lp.num_constraints):
            for j in range(lp.num_vars):
                self.table[i+1][j] = lp.A[i][j]
            self.table[i+1][lp.num_vars + i] = 1.0
            self.table[i+1][-1] = lp.b[i]
            self.basis.append(lp.num_vars + i)
    
    def primal_simplex(self):
        max_iter = 100
        iter_count = 0
        while iter_count < max_iter:
            iter_count += 1
            # Encontrar columna pivote
            s = -1
            min_val = 0
            for j in range(self.cols - 1):
                if self.table[0][j] < min_val - self.tol:
                    min_val = self.table[0][j]
                    s = j
            if s == -1:
                break
                
            # Encontrar fila pivote
            r = -1
            min_ratio = float('inf')
            for i in range(1, self.rows):
                if self.table[i][s] > self.tol:
                    ratio = self.table[i][-1] / self.table[i][s]
                    if ratio < min_ratio - self.tol:
                        min_ratio = ratio
                        r = i
            if r == -1:
                raise Exception("Problema no acotado")
                
            self.pivot(r, s)
        else:
            print("Alcanzado máximo de iteraciones en simplex primal")
        return iter_count

    def pivot(self, r, s):
        pivot_val = self.table[r][s]
        if abs(pivot_val) < self.tol:
            raise Exception("Pivote cercano a cero")
            
        # Normalizar fila pivote
        for j in range(self.cols):
            self.table[r][j] /= pivot_val
        
        # Actualizar otras filas
        for i in range(self.rows):
            if i == r:
                continue
            factor = self.table[i][s]
            for j in range(self.cols):
                self.table[i][j] -= factor * self.table[r][j]
        
        # Actualizar base
        self.basis[r-1] = s
    
    def add_gomory_cut(self, row_idx):
        row = self.table[row_idx][:]
        b_i = row[-1]
        
        # CORRECCIÓN 1: Cálculo correcto de parte fraccionaria
        f_i = b_i - math.floor(b_i)
        
        new_row = [0.0] * (self.cols + 1)
        for j in range(self.cols - 1):
            a_ij = row[j]
            # CORRECCIÓN 2: Usar floor para todos los casos
            f_ij = a_ij - math.floor(a_ij)
            new_row[j] = -f_ij
        
        # Guardar corte para visualización (solo variables originales)
        cut_coeffs = new_row[:self.original_num_vars]
        self.lp.gomory_cuts.append((cut_coeffs, -f_i))
        
        new_row[self.cols - 1] = 1.0
        new_row.append(-f_i)
        
        # Expandir tableau
        for i in range(self.rows):
            self.table[i].insert(-1, 0.0)
        self.table.append(new_row)
        self.rows += 1
        self.cols += 1
        self.basis.append(self.cols - 2)
        
        return new_row
    
    def dual_simplex(self):
        max_iter = 100
        iter_count = 0
        
        while iter_count < max_iter:
            iter_count += 1
            r = -1
            min_val = 0
            for i in range(1, self.rows):
                if self.table[i][-1] < min_val - self.tol:
                    min_val = self.table[i][-1]
                    r = i
            if r == -1:
                break
                
            s = -1
            min_ratio = float('inf')
            for j in range(self.cols - 1):
                if self.table[r][j] < -self.tol:
                    ratio = abs(self.table[0][j] / self.table[r][j])
                    if ratio < min_ratio - self.tol:
                        min_ratio = ratio
                        s = j
            if s == -1:
                raise Exception("Problema infactible")
                
            self.pivot(r, s)
        else:
            print("Alcanzado máximo de iteraciones en simplex dual")
        return iter_count
    
    def get_solution(self):
        solution = [0.0] * (self.cols - 1)
        for i in range(1, self.rows):
            var_idx = self.basis[i-1]
            solution[var_idx] = self.table[i][-1]
        return solution[:self.original_num_vars]
    
    def is_integer_solution(self, tol=1e-6):
        solution = self.get_solution()
        for x in solution:
            if abs(x - round(x)) > tol:
                return False
        return True

def plot_solution(lp, solution=None, title="Solución Óptima"):
    if lp.num_vars != 2:
        print("La visualización solo está disponible para problemas con 2 variables")
        return
    
    plt.figure(figsize=(10, 8))
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True)
    
    # Crear rango para las variables
    x = np.linspace(0, max(lp.b)*1.2, 400)
    
    # Graficar restricciones originales
    for i, (a, b_val) in enumerate(zip(lp.A, lp.b)):
        if abs(a[1]) > 1e-6:  # Restricción normal
            y = (b_val - a[0]*x) / a[1]
            plt.plot(x, y, label=f"Restricción {i+1}: {a[0]:.1f}x1 + {a[1]:.1f}x2 ≤ {b_val:.1f}")
            plt.fill_between(x, y, 0, alpha=0.1)
        else:  # Línea vertical
            plt.axvline(x=b_val/a[0], label=f"Restricción {i+1}: {a[0]:.1f}x1 ≤ {b_val:.1f}")
    
    # Graficar cortes de Gomory
    for i, (coeffs, rhs) in enumerate(lp.gomory_cuts):
        if all(abs(coeff) < 1e-6 for coeff in coeffs):
            continue
            
        if abs(coeffs[1]) > 1e-6:
            y_cut = (rhs - coeffs[0]*x) / coeffs[1]
            # CORRECCIÓN 3: Solo mostrar cortes válidos
            valid_y = np.where(y_cut >= 0, y_cut, np.nan)
            plt.plot(x, valid_y, '--', linewidth=2, label=f"Corte Gomory {i+1}")
        elif abs(coeffs[0]) > 1e-6:
            plt.axvline(x=rhs/coeffs[0], linestyle='--', linewidth=2, label=f"Corte Gomory {i+1}")
    
    # Graficar solución óptima si existe
    if solution:
        plt.scatter(solution[0], solution[1], color='red', s=100, 
                   label=f"Solución: ({solution[0]:.2f}, {solution[1]:.2f})")
        plt.annotate(f"Óptimo\n({solution[0]:.2f}, {solution[1]:.2f})", 
                    (solution[0], solution[1]), 
                    xytext=(solution[0]+0.5, solution[1]+0.5),
                    arrowprops=dict(arrowstyle='->'))
    
    # Configurar límites y leyenda
    plt.xlim(0, max(lp.b)*1.1)
    plt.ylim(0, max(lp.b)*1.1)
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    print("=== Resolvedor de Programación Lineal Entera ===")
    print("Formato requerido: Maximizar c^T x sujeto a Ax <= b, x >= 0, x entero")
    
    # Entrada de datos
    num_vars = int(input("\nNúmero de variables: "))
    num_constraints = int(input("Número de restricciones: "))
    
    c = []
    print("\nCoeficientes de la función objetivo (c):")
    for i in range(num_vars):
        c.append(float(input(f"c[{i}]: ")))
    
    A = []
    b = []
    print("\nCoeficientes de las restricciones (A):")
    for i in range(num_constraints):
        row = []
        print(f"Restricción {i+1}:")
        for j in range(num_vars):
            row.append(float(input(f"A[{i}][{j}]: ")))
        A.append(row)
        b.append(float(input(f"Lado derecho b[{i}]: ")))
    
    # Crear modelo
    lp = LinearProgram(c, A, b)
    tableau = Tableau(lp)
    
    try:
        # Fase 1: Resolver relajación lineal
        iter_primal = tableau.primal_simplex()
        solution_frac = tableau.get_solution()
        obj_frac = sum(c[i] * solution_frac[i] for i in range(len(solution_frac)))
        
        print("\n--- Solución de la relajación lineal ---")
        print(f"x = {[round(x, 2) for x in solution_frac]}")
        print(f"Valor objetivo: {round(obj_frac, 2)}")
        print(f"Iteraciones simplex: {iter_primal}")
        
        # Visualizar solución inicial
        if lp.num_vars == 2:
            plot_solution(lp, solution_frac, "Relajación Lineal (Solución Fraccionaria)")
        
        # Fase 2: Aplicar cortes de Gomory
        max_gomory_cuts = 10
        iteration = 0
        
        while not tableau.is_integer_solution() and iteration < max_gomory_cuts:
            iteration += 1
            print(f"\n--- Iteración de corte {iteration} ---")
            
            # Buscar fila para generar corte
            found = False
            for i in range(1, tableau.rows):
                var_idx = tableau.basis[i-1]
                # CORRECCIÓN 4: Solo generar cortes para variables no enteras
                val = tableau.table[i][-1]
                fractional = val - math.floor(val)
                if fractional > 1e-6 and var_idx < tableau.original_num_vars:
                    cut_row = tableau.add_gomory_cut(i)
                    print(f"Corte añadido: {cut_row[:tableau.original_num_vars]}")
                    found = True
                    break
            
            if not found:
                print("No se encontró fila adecuada para corte")
                break
            
            # Reoptimizar con dual simplex
            iter_dual = tableau.dual_simplex()
            current_sol = tableau.get_solution()
            current_obj = sum(c[i] * current_sol[i] for i in range(len(current_sol)))
            
            print(f"Solución actual: {[round(x, 2) for x in current_sol]}")
            print(f"Valor objetivo: {round(current_obj, 2)}")
            print(f"Iteraciones simplex dual: {iter_dual}")
            
            # Visualizar progreso
            if lp.num_vars == 2:
                plot_solution(lp, current_sol, f"Iteración {iteration} - Corte de Gomory")
        
        # Mostrar solución final
        solution = tableau.get_solution()
        obj_value = sum(c[i] * solution[i] for i in range(len(solution)))
        
        print("\n*** Resultado Final ***")
        print("Valores de las variables:")
        for i, x in enumerate(solution):
            print(f"x[{i}] = {round(x, 2)}")
        
        print(f"\nValor óptimo: {round(obj_value, 2)}")
        print(f"Total iteraciones Gomory: {iteration}")
        
        if not tableau.is_integer_solution():
            print("\nADVERTENCIA: Solución no entera (límite de cortes alcanzado)")
        
        # Visualizar solución final
        if lp.num_vars == 2:
            plot_solution(lp, solution, "Solución Óptima Entera")
    
    except Exception as e:
        print(f"\nError: {str(e)}")

if __name__ == "__main__":
    main()
