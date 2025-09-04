"""
Lanzador alternativo para la Calculadora de Optimización No Lineal.
"""
import flet as ft
from ProyectoFinal.interfaz import app_main


def main(page: ft.Page):
    app_main(page)


if __name__ == "__main__":
    ft.app(target=main)
