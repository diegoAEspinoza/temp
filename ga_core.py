# ga_core.py
import random
from jssp_tool import calculate_makespan
from operators import selection_tournament, crossover_jbx, mutation_swap
import settings

def create_individual(num_jobs, num_machines):
    """Crea un cromosoma válido para el JSSP."""
    chromosome = []
    for i in range(num_jobs):
        chromosome.extend([i] * num_machines)
    random.shuffle(chromosome)
    return chromosome

def create_initial_population(pop_size, num_jobs, num_machines):
    """Crea la población inicial."""
    return [create_individual(num_jobs, num_machines) for _ in range(pop_size)]

def evaluate_population(population, jobs_data, num_jobs, num_machines):
    """Evalúa cada individuo y devuelve una lista de tuplas (cromosoma, makespan)."""
    evaluated_pop = []
    for individual in population:
        makespan = calculate_makespan(individual, jobs_data, num_jobs, num_machines)
        evaluated_pop.append((individual, makespan))
    return evaluated_pop

def evolve_generation(population_with_fitness, num_jobs, num_machines):
    """
    Evoluciona la población una generación.
    population_with_fitness es una lista de tuplas (cromosoma, makespan).
    Devuelve una NUEVA lista de CROMOSOMAS (no tuplas).
    """
    # Ordenar por makespan (menor es mejor)
    population_with_fitness.sort(key=lambda x: x[1])
    
    # 1. Elitismo (Corregido: Usar ELITISM_COUNT)
    # Guardamos los mejores cromosomas para la siguiente generación
    new_population_chromosomes = [population_with_fitness[i][0] for i in range(settings.ELITISM_COUNT)]
    
    # 2. Selección de padres (Corregido: 'parents' es una lista de cromosomas)
    parents = selection_tournament(population_with_fitness, settings.TOURNAMENT_SIZE)
    
    # 3. Creación de la nueva generación (Corregido: Rellenar el resto de la población)
    while len(new_population_chromosomes) < settings.SUB_POPULATION_SIZE:
        p1, p2 = random.sample(parents, 2)
        
        # Cruce
        if random.random() < settings.CROSSOVER_RATE:
            child = crossover_jbx(p1, p2, num_jobs)
        else:
            child = p1[:] # Clonar el padre 1
            
        # Mutación
        if random.random() < settings.MUTATION_RATE:
            child = mutation_swap(child)
            
        new_population_chromosomes.append(child)
        
    return new_population_chromosomes