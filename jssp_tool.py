# jssp_tool.py
import numpy as np

def parse_benchmark(file_path):
    """
    Parsea un archivo de benchmark JSSP (formato Taillard o FT06).
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

    # Procesar las líneas de datos
    data_lines = [line.split() for line in lines[1:]]  # omitir cabecera

    jobs_data = []

    for line in data_lines:
        values = list(map(int, line))
        # Cada trabajo tiene pares (máquina, tiempo)
        job_ops = []
        for i in range(0, len(values), 2):
            machine_idx = values[i] - 1  # índice base 0
            duration = values[i + 1]
            job_ops.append((machine_idx, duration))
        jobs_data.append(job_ops)

    return num_jobs, num_machines, jobs_data


def calculate_makespan(chromosome, jobs_data, num_jobs, num_machines):
    """
    Calcula el makespan (tiempo total de finalización) para un cromosoma dado.
    El cromosoma es una secuencia de identificadores de trabajos.
    """
    # Inicializar tiempos
    machine_finish_times = [0] * num_machines
    job_finish_times = [0] * num_jobs
    job_op_counters = [0] * num_jobs

    # Validar cromosoma
    for gene in chromosome:
        if gene < 0 or gene >= num_jobs:
            raise ValueError(f" Job ID fuera de rango: {gene} (num_jobs = {num_jobs})")

    # Decodificar cromosoma a secuencia de operaciones
    op_sequence = []
    for job_id in chromosome:
        op_index = job_op_counters[job_id]
        if op_index >= len(jobs_data[job_id]):
            raise ValueError(f" Job {job_id} tiene más operaciones en el cromosoma de las esperadas.")
        op_sequence.append((job_id, op_index))
        job_op_counters[job_id] += 1

    # Construir el cronograma
    for job_id, op_index in op_sequence:
        try:
            machine_id, duration = jobs_data[job_id][op_index]
        except IndexError:
            raise IndexError(f"❌ Índice inválido: job_id={job_id}, op_index={op_index}. "
                             f"jobs_data[{job_id}] tiene {len(jobs_data[job_id])} operaciones.")
        if machine_id < 0 or machine_id >= num_machines:
            raise ValueError(f"❌ Machine ID fuera de rango: {machine_id} (num_machines = {num_machines})")

        # Calcular tiempos
        start_time = max(machine_finish_times[machine_id], job_finish_times[job_id])
        finish_time = start_time + duration

        machine_finish_times[machine_id] = finish_time
        job_finish_times[job_id] = finish_time

    return max(machine_finish_times)