import math

class LinearProgram:
    def __init__(self, c, A, b):
        self.c = c
        self.A = A
        self.b = b
        self.num_vars = len(c)
        self.num_constraints = len(b)

class Tableau:
    def __init__(self, lp):
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
        f_i = b_i - math.floor(b_i)
        
        # Crear nueva fila para el corte
        new_row = [0.0] * (self.cols + 1)
        for j in range(self.cols - 1):
            a_ij = row[j]
            f_ij = a_ij - math.floor(a_ij)
            new_row[j] = -f_ij
        new_row[self.cols - 1] = 1.0  
        new_row.append(-f_i) 
        
        # Expandir tableau
        for i in range(self.rows):
            self.table[i].insert(-1, 0.0)
        self.table.append(new_row)
        self.rows += 1
        self.cols += 1
        self.basis.append(self.cols - 2)
    
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
    
    # Crear y resolver modelo
    lp = LinearProgram(c, A, b)
    tableau = Tableau(lp)
    
    try:
        # Fase 1: Resolver relajación lineal
        tableau.primal_simplex()
        
        # Fase 2: Aplicar cortes de Gomory
        max_gomory_cuts = 5
        iteration = 0
        while not tableau.is_integer_solution() and iteration < max_gomory_cuts:
            iteration += 1
            print(f"\n--- Iteración de corte {iteration} ---")
            
            # Buscar fila para generar corte
            found = False
            for i in range(1, tableau.rows):
                var_idx = tableau.basis[i-1]
                if var_idx < tableau.original_num_vars:
                    val = tableau.table[i][-1]
                    if abs(val - round(val)) > 1e-6:
                        tableau.add_gomory_cut(i)
                        found = True
                        break
            
            if not found:
                break
            
            # Reoptimizar con dual simplex
            tableau.dual_simplex()
        
        # Mostrar solución
        solution = tableau.get_solution()
        print("\n*** Resultado ***")
        print("Valores de las variables:")
        for i, x in enumerate(solution):
            print(f"x[{i}] = {round(x, 2)}")
        
        obj_value = sum(c[i] * solution[i] for i in range(len(solution)))
        print(f"\nValor objetivo: {round(obj_value, 2)}")
        
        if not tableau.is_integer_solution():
            print("\nADVERTENCIA: Solución no entera (límite de cortes alcanzado)")
    
    except Exception as e:
        print(f"\nError: {str(e)}")

if __name__ == "__main__":
    main()
