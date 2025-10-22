# operators.py
import random

def selection_tournament(population, k):
    """
    Selección por torneo.
    Recibe una población de tuplas (cromosoma, fitness).
    Devuelve una lista de CROMOSOMAS (los padres ganadores).
    """
    selected = []
    for _ in range(len(population)):
        participants = random.sample(population, k)
        # El individuo con menor makespan (mejor fitness) gana
        winner = min(participants, key=lambda x: x[1])
        selected.append(winner[0]) # CORREGIDO: Añadir solo el cromosoma
    return selected

def crossover_jbx(parent1, parent2, num_jobs):
    """Job-Based Order Crossover (JBX)."""
    # Asegurarse de que k esté en el rango [1, num_jobs-1] si num_jobs > 1
    k_max = max(1, num_jobs - 1)
    k_min = 1
    if k_max == 1 and num_jobs == 1: # Caso borde
        k_min = 0 
        
    k = random.randint(k_min, k_max)
    job_subset = set(random.sample(range(num_jobs), k=k))
    
    child1 = [None] * len(parent1)
    
    # Copiar genes del subconjunto de trabajos de parent1 a child1
    for i, gene in enumerate(parent1):
        if gene in job_subset:
            child1[i] = gene
            
    # Rellenar los huecos con genes de parent2
    p2_idx = 0
    for i in range(len(child1)):
        if child1[i] is None:
            # Avanzar p2_idx hasta encontrar un gen que NO esté en el subset
            while parent2[p2_idx] in job_subset:
                p2_idx += 1
            child1[i] = parent2[p2_idx]
            p2_idx += 1
            
    return child1

def mutation_swap(chromosome):
    """Mutación por intercambio."""
    idx1, idx2 = random.sample(range(len(chromosome)), 2)
    chromosome[idx1], chromosome[idx2] = chromosome[idx2], chromosome[idx1]
    return chromosome