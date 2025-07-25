# Metodos-Cuantitativos

Repositorio creado para la materia Métodos Cuantitativos, centrado en la implementación y visualización de algoritmos clave en la optimización.

---

## 🎯 Proyecto Actual: Método de Planos de Corte (Gomory)

Este repositorio contiene una implementación didáctica del **Método de Planos de Corte de Gomory** para resolver problemas de Programación Lineal Entera (PLE). Utiliza el algoritmo Simplex (primal y dual) para encontrar la solución óptima de la relajación lineal y luego aplica cortes de Gomory para obtener una solución entera, si existe. Además, incluye capacidades de visualización para problemas de 2 variables.

### 📚 Conceptos Clave Implementados

* **Programación Lineal (PL):** Representación de problemas de optimización.
* **Algoritmo Simplex:** Implementación del Simplex Primal y Dual para resolver programas lineales.
* **Relajación Lineal:** Resolución de la versión continua de un problema de PL Entera.
* **Cortes de Gomory:** Generación e incorporación de restricciones adicionales (cortes) para eliminar soluciones fraccionarias y conducir a una solución entera.
* **Visualización:** Graficación de la región factible, soluciones y cortes para problemas con dos variables de decisión.

### 🚀 Cómo Empezar

Sigue estos pasos para configurar tu entorno, instalar las dependencias y ejecutar el programa.

#### 📦 Requisitos

Asegúrate de tener Python 3.x instalado en tu sistema. Las librerías necesarias son:
* `numpy`
* `matplotlib`

#### 🛠️ Instalación y Configuración

1.  **Clona este repositorio:**
    ```bash
    git clone [https://github.com/tu_usuario/Metodos-Cuantitativos.git](https://github.com/tu_usuario/Metodos-Cuantitativos.git)
    cd Metodos-Cuantitativos
    ```
    (Reemplaza `tu_usuario` con el nombre de usuario real del repositorio).

2.  **Crea y activa un entorno virtual (recomendado):**
    Esto aísla las dependencias de tu proyecto de tu instalación global de Python.

    * **Crear el entorno virtual:**
        ```bash
        python -m venv venv
        ```
    * **Activar el entorno virtual:**
        * **En Windows (Símbolo del Sistema):**
            ```bash
            venv\Scripts\activate
            ```
        * **En Windows (PowerShell):**
            ```powershell
            .\venv\Scripts\Activate.ps1
            ```
        * **En Linux o macOS:**
            ```bash
            source venv/bin/activate
            ```
    Verás `(venv)` al inicio de tu línea de comandos, indicando que el entorno está activo.

3.  **Instala las dependencias:**
    Aunque no hay un `requirements.txt` explícito en la raíz, el código utiliza `numpy` y `matplotlib`. Instálalos así:
    ```bash
    pip install numpy matplotlib
    ```
    (Se recomienda generar un `requirements.txt` para futuras referencias ejecutando `pip freeze > requirements.txt` una vez que tengas las librerías instaladas).

### 🏃‍♂️ Ejecución del Programa

Una vez que el entorno virtual esté activado y las dependencias instaladas, puedes ejecutar el programa principal:

```bash
python PlanosCorte.py