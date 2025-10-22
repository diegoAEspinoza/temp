# main.py
from mpi4py import MPI
import random
import time
import settings
from jssp_tool import parse_benchmark
from ga_core import create_initial_population, evaluate_population, evolve_generation

def main():
    # --- Inicialización de MPI ---
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()


    if size!= settings.NUM_ISLANDS:
        if rank == 0:
            print(f"Error: El número de procesos MPI ({size}) no coincide con NUM_ISLANDS ({settings.NUM_ISLANDS}) en settings.py.")
        return

    # --- Carga de datos y configuración (cada proceso hace lo mismo) ---
    if rank == 0:
        print("--- Algoritmo Genético Paralelo con Modelo de Islas para JSSP ---")
        print(f"Problema: {settings.BENCHMARK_FILE}")
        print(f"Configuración: {settings.NUM_ISLANDS} islas, {settings.SUB_POPULATION_SIZE} individuos/isla")
        print(f"Migración: Frecuencia={settings.MIGRATION_FREQUENCY}, Tasa={settings.MIGRATION_RATE*100}%, Topología={settings.MIGRATION_TOPOLOGY}")
        print("-" * 60)
    
    start_time = time.time()
    
    num_jobs, num_machines, jobs_data = parse_benchmark(settings.BENCHMARK_FILE)
    
    # --- Creación de la subpoblación inicial en cada isla ---
    random.seed(rank) # Semilla diferente para cada isla
    sub_population = create_initial_population(settings.SUB_POPULATION_SIZE, num_jobs, num_machines)
    
    best_makespan_so_far = float('inf')
    best_solution = None

    # --- Bucle Evolutivo Principal ---
    for generation in range(settings.NUM_GENERATIONS):
        # Evaluar la población actual
        evaluated_pop = evaluate_population(sub_population, jobs_data, num_jobs, num_machines)
        
        # Actualizar el mejor local
        current_best = min(evaluated_pop, key=lambda x: x[1])
        if current_best[1] < best_makespan_so_far:
            best_makespan_so_far = current_best[1]
            best_solution = current_best
            if rank == 0:
                print(f"[Gen {generation}] Nuevo mejor makespan global (encontrado en isla 0): {best_makespan_so_far}")

        # Evolucionar una generación
        sub_population = evolve_generation(evaluated_pop, num_jobs, num_machines)

        # --- Fase de Migración ---
        if generation > 0 and generation % settings.MIGRATION_FREQUENCY == 0:
            # 1. Seleccionar emigrantes (los mejores de la isla)
            evaluated_pop.sort(key=lambda x: x[1])
            emigrants = [item for item in evaluated_pop]

            # 2. Determinar destino y fuente
            if settings.MIGRATION_TOPOLOGY == "RING":
                dest = (rank + 1) % size
                source = (rank - 1 + size) % size
            elif settings.MIGRATION_TOPOLOGY == "RANDOM":
                dest = random.choice([i for i in range(size) if i!= rank])
                source = dest # En un modelo aleatorio simple, la comunicación puede no ser simétrica
            
            # 3. Intercambiar individuos (evitando deadlock)
            immigrants = None
            if rank % 2 == 0:
                comm.send(emigrants, dest=dest, tag=11)
                immigrants = comm.recv(source=source, tag=11)
            else:
                immigrants = comm.recv(source=source, tag=11)
                comm.send(emigrants, dest=dest, tag=11)

            # 4. Integrar inmigrantes (reemplazando a los peores)
            if immigrants:
                # Reevaluar para ordenar y encontrar los peores
                evaluated_pop = evaluate_population(sub_population, jobs_data, num_jobs, num_machines)
                evaluated_pop.sort(key=lambda x: x[1], reverse=True) # Peores primero
                
                current_pop_chromosomes = [item for item in evaluated_pop]
                for i in range(min(len(immigrants), len(current_pop_chromosomes))):
                    current_pop_chromosomes[i] = immigrants[i]
                
                sub_population = current_pop_chromosomes

    # --- Recopilación de Resultados ---
    comm.Barrier() # Esperar a que todos los procesos terminen
    
    final_best_solution = (best_solution, best_makespan_so_far)
    
    # Reunir los mejores de cada isla en el proceso raíz (rank 0)
    all_bests = comm.gather(final_best_solution, root=0)
    
    if rank == 0:
        overall_best = min(all_bests, key=lambda x: x[1])
        end_time = time.time()
        
        print("-" * 60)
        print("Evolución finalizada.")
        print(f"Mejor makespan encontrado: {overall_best[1]}")
        # print(f"Mejor solución (cromosoma): {overall_best}")
        print(f"Tiempo total de ejecución: {end_time - start_time:.2f} segundos")
        print("-" * 60)

if __name__ == "__main__":
    main()