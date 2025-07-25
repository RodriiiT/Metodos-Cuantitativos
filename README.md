# Repositorio de Métodos Cuantitativos

¡Bienvenido al repositorio de la materia **Métodos Cuantitativos**!

Este repositorio está diseñado para alojar todos los proyectos, tareas y ejemplos de código desarrollados durante el curso de **Métodos Cuantitativos**. Aquí encontrarás implementaciones de diversos algoritmos y técnicas cuantitativas, pensadas para facilitar el aprendizaje y la comprensión de los conceptos vistos en clase.

## Información del Estudiante

* **Nombre:** Rodrigo Torres
* **Cédula:** 31.015.922
* **Sección:** 308C1

---

## Visión General del Contenido

Este repositorio contendrá una variedad de proyectos relacionados con los métodos cuantitativos, que pueden incluir, entre otros:

* **Programación Lineal (PL):** Implementaciones de métodos como Simplex, Dualidad, Análisis de Sensibilidad.
* **Programación Lineal Entera (PLE):** Algoritmos como Ramificación y Acotamiento, Planos de Corte.
* **Optimización No Lineal:** Métodos de búsqueda, gradientes.
* **Modelos de Inventario y Gestión de Proyectos.**
* **Simulación de Eventos Discretos.**

Cada carpeta de tarea o proyecto principal estará acompañada de su propia documentación detallada dentro de un archivo `README.md` específico de la tarea, que cubrirá los objetivos, requisitos, instrucciones de ejecución y ejemplos de uso.

---

## Estructura del Repositorio

El repositorio está organizado por tareas o proyectos principales. Cada uno se encuentra en su propia carpeta para mantener una estructura clara y modular.

├── Tarea IV/
│   ├── ImagenesCorte/
│   ├── PlanosCorte.py
│   ├── README.md
│   └── requeriments.txt
└── README.md

---

## Entorno de Desarrollo (Recomendado)

Se recomienda encarecidamente utilizar un entorno virtual para gestionar las dependencias de los proyectos. Esto ayuda a evitar conflictos entre las versiones de las librerías y mantiene tu entorno global de Python limpio.

### Creación y Activación de un Entorno Virtual

1.  **Crear el entorno virtual:**
    ```bash
    python -m venv venv
    ```
    Esto creará una carpeta `venv` en la raíz del repositorio.

2.  **Activar el entorno virtual:**
    * **En Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    * **En macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```
    Una vez activado, verás `(venv)` al inicio de tu línea de comandos, indicando que estás dentro del entorno virtual.

### Desactivar el Entorno Virtual

Para salir del entorno virtual en cualquier momento, simplemente ejecuta:
```bash
deactivate
```

## Tareas y Proyectos

A continuación, se detalla la información para cada tarea o proyecto principal que se encuentra en el repositorio.

Tarea IV: Resolvedor de Programación Lineal Entera con Planos de Corte
Esta carpeta contiene la implementación de un programa para resolver problemas de Programación Lineal Entera (PLE) utilizando el método de planos de corte de Gomory. Es una herramienta útil para comprender cómo este algoritmo aborda la optimización discreta.

Contenido de la carpeta `Tarea IV`:

- PlanosCorte.py: El script principal que implementa el algoritmo de planos de corte.
- ImagenesCorte/: Una carpeta que contiene imágenes generadas por el programa.
- requeriments.txt: Archivo que lista las dependencias específicas para esta tarea.
- README.md: Este archivo, que proporciona una descripción detallada de la tarea.

---

## Contribuciones
Si deseas contribuir a este repositorio (¡lo cual es muy bienvenido!), por favor, sigue las siguientes pautas:

1. Haz un "fork" del repositorio.
2. Crea una nueva rama para tus cambios (`git checkout -b feature/nueva-tarea`).
3. Realiza tus modificaciones y asegúrate de que el código sea claro y esté bien comentado.
4. Asegúrate de actualizar o crear un `README.md` detallado para cualquier nueva tarea o proyecto que añadas.
5. Realiza "commits" significativos (`git commit -m "feat: Añadir solución para la Tarea X"`).
6. Sube tus cambios a tu "fork" (`git push origin feature/nueva-tarea`).
7. Abre un "Pull Request" a la rama `main` de este repositorio.


