# jssp_tool.py
import numpy as np

def parse_benchmark(file_path):
    """
    Parsea un archivo de benchmark JSSP (formato Taillard/OR-Library como ft06, ta01).
    Devuelve:
    - num_jobs: número de trabajos
    - num_machines: número de máquinas
    - jobs_data: lista de trabajos con pares (máquina, duración)
    """
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    # Leer cabecera
    first_line = lines[0].split()
    num_jobs = int(first_line[0])
    num_machines = int(first_line[1])

    # El formato Taillard tiene num_jobs líneas de duraciones, 
    # seguidas de num_jobs líneas de máquinas.
    if len(lines[1:]) < 2 * num_jobs:
        raise ValueError(f"Formato de archivo incorrecto. Se esperaban {2 * num_jobs} líneas de datos, pero se encontraron {len(lines[1:])}.")

    # Procesar las líneas de duraciones
    duration_lines = [list(map(int, line.split())) for line in lines[1 : 1 + num_jobs]]
    
    # Procesar las líneas de máquinas
    machine_lines = [list(map(int, line.split())) for line in lines[1 + num_jobs : 1 + 2 * num_jobs]]

    if len(duration_lines) != num_jobs or len(machine_lines) != num_jobs:
        raise ValueError(f"Formato de archivo incorrecto. Se esperaban {num_jobs} líneas de duración y {num_jobs} de máquinas.")

    jobs_data = []
    for i in range(num_jobs):
        job_ops = []
        for j in range(num_machines):
            # Los archivos Taillard usan índices 1-based para máquinas
            machine_idx = machine_lines[i][j] - 1  # Convertir a 0-based
            duration = duration_lines[i][j]
            
            if machine_idx < 0 or machine_idx >= num_machines:
                 raise ValueError(f"Machine ID {machine_lines[i][j]} fuera de rango en el archivo.")
                 
            job_ops.append((machine_idx, duration))
        jobs_data.append(job_ops)

    return num_jobs, num_machines, jobs_data


def calculate_makespan(chromosome, jobs_data, num_jobs, num_machines):
    """
    Calcula el makespan (tiempo total de finalización) para un cromosoma dado.
    El cromosoma es una secuencia de identificadores de trabajos.
    """
    # Inicializar tiempos
    machine_finish_times = np.zeros(num_machines, dtype=int)
    job_finish_times = np.zeros(num_jobs, dtype=int)
    job_op_counters = np.zeros(num_jobs, dtype=int)

    # Validar cromosoma (rápida verificación)
    if len(chromosome) != num_jobs * num_machines:
        raise ValueError("Longitud de cromosoma incorrecta.")

    # Decodificar cromosoma a secuencia de operaciones
    for job_id in chromosome:
        if job_id < 0 or job_id >= num_jobs:
            raise ValueError(f"Job ID fuera de rango: {job_id} (num_jobs = {num_jobs})")

        op_index = job_op_counters[job_id]
        
        try:
            machine_id, duration = jobs_data[job_id][op_index]
        except IndexError:
            raise IndexError(f"Índice inválido: job_id={job_id}, op_index={op_index}. "
                             f"jobs_data[{job_id}] tiene {len(jobs_data[job_id])} operaciones.")
        
        if machine_id < 0 or machine_id >= num_machines:
            raise ValueError(f"Machine ID fuera de rango: {machine_id} (num_machines = {num_machines})")

        # Calcular tiempos
        start_time = max(machine_finish_times[machine_id], job_finish_times[job_id])
        finish_time = start_time + duration

        machine_finish_times[machine_id] = finish_time
        job_finish_times[job_id] = finish_time
        
        job_op_counters[job_id] += 1

    return np.max(machine_finish_times)