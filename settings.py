# Este archivo centraliza todos los parámetros para facilitar los experimentos.
# settings.py

# --- PARÁMETROS DEL ALGORITMO GENÉTICO ---
TOTAL_POPULATION_SIZE = 512
NUM_GENERATIONS = 200
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.2
TOURNAMENT_SIZE = 3
ELITISM_COUNT = 2  # Número de individuos élite a conservar en cada generación por isla

# --- PARÁMETROS DEL MODELO DE ISLAS ---
NUM_ISLANDS = 4  # Debe ser igual al número de procesos MPI (-n)
MIGRATION_FREQUENCY = 25  # Migración cada 25 generaciones
MIGRATION_RATE = 0.10  # 10% de la subpoblación migra
# Topologías disponibles: "RING", "RANDOM"
MIGRATION_TOPOLOGY = "RING"

# --- PARÁMETROS DEL PROBLEMA JSSP ---
# Cambia esto para usar un benchmark diferente
BENCHMARK_FILE = "benchmarks/ft06.txt"
# BENCHMARK_FILE = "benchmarks/ta01.txt"

# --- VALIDACIÓN DE PARÁMETROS ---
if TOTAL_POPULATION_SIZE % NUM_ISLANDS!= 0:
    raise ValueError("TOTAL_POPULATION_SIZE debe ser divisible por NUM_ISLANDS")

SUB_POPULATION_SIZE = TOTAL_POPULATION_SIZE // NUM_ISLANDS
MIGRATION_COUNT = int(SUB_POPULATION_SIZE * MIGRATION_RATE)
if MIGRATION_COUNT == 0 and MIGRATION_RATE > 0:
    MIGRATION_COUNT = 1 # Asegurar que al menos un individuo migre