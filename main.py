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


    if size != settings.NUM_ISLANDS:
        if rank == 0:
            print(f"Error: El número de procesos MPI ({size}) no coincide con NUM_ISLANDS ({settings.NUM_ISLANDS}) en settings.py.")
        return

    # --- Carga de datos y configuración (cada proceso hace lo mismo) ---
    if rank == 0:
        print("--- Algoritmo Genetico Paralelo con Modelo de Islas para JSSP ---")
        print(f"Problema: {settings.BENCHMARK_FILE}")
        print(f"Configuracion: {settings.NUM_ISLANDS} islas, {settings.SUB_POPULATION_SIZE} individuos/isla")
        print(f"Generaciones: {settings.NUM_GENERATIONS}")
        print(f"Migración: Frecuencia={settings.MIGRATION_FREQUENCY}, Tasa={settings.MIGRATION_RATE*100}% ({settings.MIGRATION_COUNT} indiv.), Topología={settings.MIGRATION_TOPOLOGY}")
        print("-" * 60)
    
    start_time = time.time()
    
    num_jobs, num_machines, jobs_data = parse_benchmark(settings.BENCHMARK_FILE)
    
    # --- Creación de la subpoblación inicial en cada isla ---
    random.seed(rank) # Semilla diferente para cada isla
    sub_population = create_initial_population(settings.SUB_POPULATION_SIZE, num_jobs, num_machines)
    
    # CORREGIDO: Inicializar best_solution como tupla para consistencia
    best_solution = (None, float('inf'))

    # --- Bucle Evolutivo Principal ---
    for generation in range(settings.NUM_GENERATIONS):
        # Evaluar la población actual (sub_population es una lista de cromosomas)
        evaluated_pop = evaluate_population(sub_population, jobs_data, num_jobs, num_machines)
        
        # Actualizar el mejor local
        current_best = min(evaluated_pop, key=lambda x: x[1])
        if current_best[1] < best_solution[1]:
            best_solution = current_best
            
            # Opcional: Imprimir si CUALQUIER isla encuentra un nuevo mejor.
            # (Esto puede ser mucho output, por eso el original lo limitaba a rank 0)
            # print(f"[Rank {rank}][Gen {generation}] Nuevo mejor local: {best_solution[1]}")


        # Evolucionar una generación
        # sub_population ahora es una nueva lista de cromosomas
        sub_population = evolve_generation(evaluated_pop, num_jobs, num_machines)

        # --- Fase de Migración ---
        if generation > 0 and generation % settings.MIGRATION_FREQUENCY == 0:
            # 1. Seleccionar emigrantes (los mejores de la isla)
            # Reevaluar la población evolucionada para la migración
            migrant_evaluated_pop = evaluate_population(sub_population, jobs_data, num_jobs, num_machines)
            migrant_evaluated_pop.sort(key=lambda x: x[1])
            
            # CORREGIDO: Enviar solo MIGRATION_COUNT
            emigrants = migrant_evaluated_pop[:settings.MIGRATION_COUNT]

            # 2. Determinar destino y fuente
            if settings.MIGRATION_TOPOLOGY == "RING":
                dest = (rank + 1) % size
                source = (rank - 1 + size) % size
            elif settings.MIGRATION_TOPOLOGY == "RANDOM":
                dest = random.choice([i for i in range(size) if i != rank])
                source = MPI.ANY_SOURCE # Recibir de cualquiera
            
            # 3. Intercambiar individuos (Usando Sendrecv para seguridad con topología RANDOM)
            # CORREGIDO: Usar Sendrecv es más robusto para todas las topologías
            # y evita deadlocks si la lógica par/impar falla o es incorrecta.
            immigrants = comm.sendrecv(
                emigrants, dest=dest, sendtag=11,
                source=source, recvtag=11
            )

            # 4. Integrar inmigrantes (reemplazando a los peores)
            if immigrants:
                # No es necesario re-evaluar 'sub_population', ya tenemos 'migrant_evaluated_pop'
                # Ordenar por peor fitness (mayor makespan)
                migrant_evaluated_pop.sort(key=lambda x: x[1], reverse=True)
                
                # CORREGIDO: Extraer solo los cromosomas de la población actual
                current_pop_chromosomes = [item[0] for item in migrant_evaluated_pop]
                
                num_to_replace = min(len(immigrants), len(current_pop_chromosomes))
                
                for i in range(num_to_replace):
                    # immigrants[i] es una tupla (chromo, fit)
                    # current_pop_chromosomes[i] es el peor cromosoma
                    current_pop_chromosomes[i] = immigrants[i][0] # Reemplazar el peor
                
                # CORREGIDO: sub_population es ahora la lista de cromosomas actualizada
                sub_population = current_pop_chromosomes

    # --- Recopilación de Resultados ---
    comm.Barrier() # Esperar a que todos los procesos terminen
    
    # CORREGIDO: Enviar best_solution (que es la tupla (chromo, fit))
    all_bests = comm.gather(best_solution, root=0)
    
    if rank == 0:
        # Filtrar islas que quizás no encontraron solución (improbable)
        valid_bests = [b for b in all_bests if b[0] is not None]
        
        if not valid_bests:
            print("No se encontró ninguna solución válida.")
            return

        overall_best = min(valid_bests, key=lambda x: x[1])
        end_time = time.time()
        
        print("-" * 60)
        print("Evolución finalizada.")
        print(f"Mejor makespan global encontrado: {overall_best[1]}")
        # print(f"Mejor solución (cromosoma): {overall_best[0]}")
        print(f"Tiempo total de ejecución: {end_time - start_time:.2f} segundos")
        print("-" * 60)

if __name__ == "__main__":
    main()