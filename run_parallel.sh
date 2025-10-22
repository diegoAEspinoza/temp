#!/bin/bash

# Script para ejecutar el algoritmo genético paralelo.
# El número de procesos (-n) debe coincidir con NUM_ISLANDS en settings.py

NUM_PROCESSES=4

echo "Ejecutando el algoritmo con $NUM_PROCESSES islas..."
mpiexec -n $NUM_PROCESSES python main.py

echo "Ejecución completada."