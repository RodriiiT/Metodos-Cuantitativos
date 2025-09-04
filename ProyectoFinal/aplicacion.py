"""
Aplicación: Calculadora de Optimización No Lineal (UI moderna y modular)
Ejecutar desde el directorio padre (/workspace/Metodos-Cuantitativos) con:
    PYTHONPATH=/workspace/Metodos-Cuantitativos python ProyectoFinal/aplicacion.py
"""

import flet as ft
from ProyectoFinal.interfaz import app_main


def main(page: ft.Page):
    app_main(page)


if __name__ == "__main__":
    ft.app(target=main)