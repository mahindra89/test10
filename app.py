import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
import random

# --- Page Setup ---
st.set_page_config(layout="wide", page_title="CPU Scheduling Simulator")
st.markdown("""
    <style>
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
            max-width: 100%;
        }
    </style>
""", unsafe_allow_html=True)

# --- Title ---
st.title("CPU Scheduling Simulator")

# --- JobScheduler Class for Round Robin ---
class JobScheduler:
    def __init__(self):
        self.processes = []
        self.gantt_data = []
        self.queue_snapshots = []

    def add_job(self, job_id, arrival_time, burst_time):
        self.processes.append({
            'id': job_id,
            'arrival_time': arrival_time,
            'burst_time': burst_time,
            'remaining_time': burst_time
        })

    def reset(self):
        # Reset simulation data but keep job definitions
        for process in self.processes:
            process['remaining_time'] = process['burst_time']
            if 'start_time' in process:
                del process['start_time']
            if 'end_time' in process:
                del process['end_time']
            if 'turnaround_time' in process:
                del process['turnaround_time']

        self.gantt_data = []
        self.queue_snapshots = []

    def run_round_robin_with_quantum(self, num_cpus, time_quantum, chunk_size):
        self.reset()

        # Sort processes by arrival time
        self.processes.sort(key=lambda x: x['arrival_time'])

        # Initialize tracking variables
        current_time = 0
        completed_jobs = 0

        # CPU state
        cpu_names = [f"CPU{i+1}" for i in range(num_cpus)]
        busy_until = {cpu: 0 for cpu in cpu_names}
        current_jobs = {cpu: None for cpu in cpu_names}
        busy_jobs = set()

        # Job tracking
        start_times = {}
        end_times = {}

        # Ready queue for Round Robin
        ready_queue = []

        # Capture initial queue state
        self.capture_queue_state(current_time, self.get_available_jobs(current_time, busy_jobs))

        # Time quantization - round to nearest quantum
        def next_quantum_time(time):
            return (int(time / time_quantum) + 1) * time_quantum

        # Main simulation loop
        while completed_jobs < len(self.processes):
            # Check for newly arrived jobs
            for process in self.processes:
                if process['arrival_time'] <= current_time and process['remaining_time'] > 0 and process['id'] not in busy_jobs and process['id'] not in ready_queue:
                    ready_queue.append(process['id'])

            # Check for completed jobs
            for cpu, busy_time in busy_until.items():
                if busy_time <= current_time and current_jobs[cpu] is not None:
                    job_id = current_jobs[cpu]
                    if job_id in busy_jobs:
                        busy_jobs.remove(job_id)

                    job = next((p for p in self.processes if p['id'] == job_id), None)
                    if job['remaining_time'] > 0:
                        # If job isn't finished, put it back in queue
                        ready_queue.append(job_id)

                    current_jobs[cpu] = None

            # Only assign jobs at quantum boundaries
            is_quantum_boundary = abs(current_time % time_quantum) < 0.001 or len(self.gantt_data) == 0
            
            # Get available CPUs
            available_cpus = [cpu for cpu, time in busy_until.items() if time <= current_time and current_jobs[cpu] is None]

            if available_cpus and ready_queue and is_quantum_boundary:
                # Capture queue state when scheduling decisions are made
                self.capture_queue_state(current_time, ready_queue.copy())

                # Assign jobs to available CPUs in round robin fashion
                for cpu in available_cpus:
                    if not ready_queue:
                        break

                    selected_job_id = ready_queue.pop(0)
                    selected_job = next((p for p in self.processes if p['id'] == selected_job_id), None)

                    if selected_job_id not in start_times:
                        start_times[selected_job_id] = current_time
                        selected_job['start_time'] = current_time

                    # Get the next chunk for this job, respecting both time quantum and chunk size
                    # A job can only run for as long as the time quantum
                    # But it can only process as much as the chunk size allows
                    chunk_size_for_job = min(chunk_size, selected_job['remaining_time'])
                    time_for_chunk = min(time_quantum, chunk_size_for_job)

                    # Update job and CPU state
                    busy_jobs.add(selected_job_id)
                    current_jobs[cpu] = selected_job_id
                    selected_job['remaining_time'] -= chunk_size_for_job
                    busy_until[cpu] = current_time + time_for_chunk

                    # Record for Gantt chart
                    self.gantt_data.append((current_time, cpu, selected_job_id, time_for_chunk))

                    # Check if job is completed
                    if abs(selected_job['remaining_time']) < 0.001:
                        end_times[selected_job_id] = current_time + time_for_chunk
                        selected_job['end_time'] = current_time + time_for_chunk
                        selected_job['turnaround_time'] = selected_job['end_time'] - selected_job['arrival_time']
                        completed_jobs += 1

            # Advance time to next event
            next_times = []
            
            # Next quantum boundary
            next_quantum = next_quantum_time(current_time)
            next_times.append(next_quantum)
            
            # Next job completion
            for cpu, time in busy_until.items():
                if time > current_time:
                    next_times.append(time)
            
            # Next job arrival
            for process in self.processes:
                if process['arrival_time'] > current_time and process['remaining_time'] > 0:
                    next_times.append(process['arrival_time'])

            if next_times:
                current_time = min(next_times)
            else:
                current_time += time_quantum  # Increment by quantum if no events

        # Calculate metrics
        total_turnaround = 0
        for process in self.processes:
            total_turnaround += process['turnaround_time']

        avg_turnaround = total_turnaround / len(self.processes)
        return avg_turnaround

    def run_round_robin_without_quantum(self, num_cpus, chunk_size):
        self.reset()

        # Sort processes by arrival time
        self.processes.sort(key=lambda x: x['arrival_time'])

        # Initialize tracking variables
        current_time = 0
        completed_jobs = 0

        # CPU state
        cpu_names = [f"CPU{i+1}" for i in range(num_cpus)]
        busy_until = {cpu: 0 for cpu in cpu_names}
        current_jobs = {cpu: None for cpu in cpu_names}
        busy_jobs = set()

        # Job tracking
        start_times = {}
        end_times = {}

        # Ready queue for Round Robin
        ready_queue = []

        # Capture initial queue state
        self.capture_queue_state(current_time, self.get_available_jobs(current_time, busy_jobs))

        # Main simulation loop
        while completed_jobs < len(self.processes):
            # Check for newly arrived jobs
            for process in self.processes:
                if process['arrival_time'] <= current_time and process['remaining_time'] > 0 and process['id'] not in busy_jobs and process['id'] not in ready_queue:
                    ready_queue.append(process['id'])

            # Check for completed jobs
            for cpu, busy_time in busy_until.items():
                if busy_time <= current_time and current_jobs[cpu] is not None:
                    job_id = current_jobs[cpu]
                    if job_id in busy_jobs:
                        busy_jobs.remove(job_id)

                    job = next((p for p in self.processes if p['id'] == job_id), None)
                    if job['remaining_time'] > 0:
                        # If job isn't finished, put it back in queue
                        ready_queue.append(job_id)

                    current_jobs[cpu] = None

            # Get available CPUs - assign jobs immediately when CPU becomes free
            available_cpus = [cpu for cpu, time in busy_until.items() if time <= current_time and current_jobs[cpu] is None]

            if available_cpus and ready_queue:
                # Capture queue state when scheduling decisions are made
                self.capture_queue_state(current_time, ready_queue.copy())

                # Assign jobs to available CPUs in round robin fashion
                for cpu in available_cpus:
                    if not ready_queue:
                        break

                    selected_job_id = ready_queue.pop(0)
                    selected_job = next((p for p in self.processes if p['id'] == selected_job_id), None)

                    if selected_job_id not in start_times:
                        start_times[selected_job_id] = current_time
                        selected_job['start_time'] = current_time

                    # Get the next chunk for this job based on chunk size
                    chunk_size_for_job = min(chunk_size, selected_job['remaining_time'])

                    # Update job and CPU state
                    busy_jobs.add(selected_job_id)
                    current_jobs[cpu] = selected_job_id
                    selected_job['remaining_time'] -= chunk_size_for_job
                    busy_until[cpu] = current_time + chunk_size_for_job

                    # Record for Gantt chart
                    self.gantt_data.append((current_time, cpu, selected_job_id, chunk_size_for_job))

                    # Check if job is completed
                    if abs(selected_job['remaining_time']) < 0.001:
                        end_times[selected_job_id] = current_time + chunk_size_for_job
                        selected_job['end_time'] = current_time + chunk_size_for_job
                        selected_job['turnaround_time'] = selected_job['end_time'] - selected_job['arrival_time']
                        completed_jobs += 1

            # Advance time to next event
            next_times = []
            
            # Next job completion
            for cpu, time in busy_until.items():
                if time > current_time:
                    next_times.append(time)
            
            # Next job arrival
            for process in self.processes:
                if process['arrival_time'] > current_time and process['remaining_time'] > 0:
                    next_times.append(process['arrival_time'])

            if next_times:
                current_time = min(next_times)
            else:
                current_time += 1  # Just to avoid infinite loop in edge cases

        # Calculate metrics
        total_turnaround = 0
        for process in self.processes:
            total_turnaround += process['turnaround_time']

        avg_turnaround = total_turnaround / len(self.processes)
        return avg_turnaround

    def get_available_jobs(self, current_time, busy_jobs):
        """Get IDs of jobs that have arrived and are not busy."""
        return [p['id'] for p in self.processes
                if p['arrival_time'] <= current_time and
                p['remaining_time'] > 0 and
                p['id'] not in busy_jobs]

    def capture_queue_state(self, time, available_jobs):
        """Record the state of the queue at a given time."""
        if not available_jobs:
            return

        job_info = []
        for job_id in available_jobs:
            remaining = next((p['remaining_time'] for p in self.processes if p['id'] == job_id), 0)
            job_info.append((job_id, round(remaining, 1)))

        if job_info:
            self.queue_snapshots.append((time, job_info))

    def get_results_dataframe(self):
        """Get results as a pandas DataFrame for display."""
        data = []
        for process in sorted(self.processes, key=lambda x: x['id']):
            data.append({
                "Job": process['id'],
                "Arrival": process['arrival_time'],
                "Burst": process['burst_time'],
                "Start": round(process['start_time'], 1) if 'start_time' in process else None,
                "End": round(process['end_time'], 1) if 'end_time' in process else None,
                "Turnaround": round(process['turnaround_time'], 1) if 'turnaround_time' in process else None
            })
        return pd.DataFrame(data)

    def get_average_turnaround(self):
        """Calculate and return the average turnaround time."""
        total = 0
        count = 0
        for process in self.processes:
            if 'turnaround_time' in process:
                total += process['turnaround_time']
                count += 1
        return total / count if count > 0 else 0

# --- Utility Function for Drawing Gantt Chart ---
def draw_gantt_chart(gantt_data, queue_snapshots, processes, quantum_time=None, algorithm_name=""):
    # Find the end time for each process to get max_time
    end_times = {}
    for process in processes:
        if isinstance(process, dict) and 'id' in process and 'end_time' in process:
            end_times[process['id']] = process['end_time']
    
    max_time = max(end_times.values()) if end_times else 0
    
    # Get a list of CPUs from gantt data
    cpus = set(cpu for _, cpu, _, _ in gantt_data)
    cpu_names = sorted(list(cpus))
    
    fig, ax = plt.subplots(figsize=(18, 8))
    cmap = plt.cm.get_cmap('tab20')
    
    # Create color mapping for jobs
    process_ids = []
    if isinstance(processes[0], dict):
        process_ids = [p['id'] for p in processes]
    else:
        for p in processes:
            process_ids.append(p)
    
    colors = {p_id: mcolors.to_hex(cmap(i / max(len(process_ids), 1))) 
             for i, p_id in enumerate(process_ids)}
    
    y_pos = {cpu: len(cpu_names) - idx for idx, cpu in enumerate(cpu_names)}
    
    # Plot job blocks
    for start, cpu, job, duration in gantt_data:
        y = y_pos[cpu]
        ax.barh(y, duration, left=start, color=colors[job], edgecolor='black')
        ax.text(start + duration / 2, y, job, ha='center', va='center', color='white', fontsize=9)
    
    # Add time grid and quantum markers if needed
    for t in range(int(max_time) + 1):
        if quantum_time and t % int(quantum_time) == 0:
            ax.axvline(x=t, color='red', linestyle='-', linewidth=0.5, alpha=0.6)
        else:
            ax.axvline(x=t, color='black', linestyle='--', alpha=0.2)
    
    # Plot queue snapshots
    for time, jobs in queue_snapshots:
        for i, (jid, rem) in enumerate(jobs):
            y = -1 - i * 0.6
            rect = patches.Rectangle((time - 0.25, y - 0.25), 0.5, 0.5, edgecolor='black', facecolor='white')
            ax.add_patch(rect)
            ax.text(time, y, f"{jid}={rem}", ha='center', va='center', fontsize=7)
    
    max_q = max((len(q[1]) for q in queue_snapshots), default=0)
    ax.set_ylim(-1 - max_q * 0.6 - 0.5, len(cpu_names) + 1)
    ax.set_yticks(list(y_pos.values()))
    ax.set_yticklabels(cpu_names)
    ax.set_xlabel("Time")
    ax.set_title(f"{algorithm_name} Gantt Chart")
    
    if quantum_time:
        ax.legend([Line2D([0], [0], color='red', lw=2)], ['Quantum Marker'], loc='upper right')
    
    plt.grid(axis='x')
    plt.tight_layout()
    return fig

# --- Algorithm Selection ---
st.sidebar.title("Algorithm Selection")
algo = st.sidebar.radio("Choose CPU Scheduling Algorithm", (
    "STRF Scheduling with Quantum Time",
    "STRF Scheduling Without Quantum Time",
    "Round Robin with Quantum Time",
    "Round Robin without Quantum Time"
))
st.markdown("---")

# --- STRF With Quantum Time ---
def run_strf_with_quantum():
    st.subheader("STRF Scheduling with Quantum Time")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        num_jobs = st.number_input("Number of Jobs", 1, 10, 4, key="strf_wq_jobs")
    with col2:
        num_cpus = st.number_input("Number of CPUs", 1, 4, 2, key="strf_wq_cpus")
    with col3:
        chunk_unit = st.number_input("Chunk Unit (e.g., 0.5, 1.0)", value=1.0, key="strf_wq_chunk")
    with col4:
        quantum_time = st.number_input("Quantum Time", value=2.0, key="strf_wq_quantum")

    if st.button("Randomize Job Times", key="strf_wq_rand"):
        st.session_state.strf_random_jobs = [
            {'arrival': round(random.uniform(0, 5) * 2) / 2, 'burst': round(random.uniform(1, 10) * 2) / 2}
            for _ in range(num_jobs)
        ]

    processes = []
    for i in range(num_jobs):
        default = st.session_state.get("strf_random_jobs", [{}]*num_jobs)[i]
        c1, c2 = st.columns(2)
        with c1:
            arrival = st.number_input(f"Arrival Time for J{i+1}", value=default.get('arrival', 0.0), key=f"strf_wq_arr_{i}")
        with c2:
            burst = st.number_input(f"Burst Time for J{i+1}", value=default.get('burst', 3.0), key=f"strf_wq_burst_{i}")
        processes.append({'id': f'J{i+1}', 'arrival_time': arrival, 'burst_time': burst})

    if st.button("Run Simulation", key="strf_wq_run"):
        strf_simulate(processes, num_cpus, chunk_unit, quantum_time, "STRF with Quantum Time")

# --- STRF Without Quantum Time ---
def run_strf_without_quantum():
    st.subheader("STRF Scheduling Without Quantum Time")
    col1, col2, col3 = st.columns(3)
    with col1:
        num_jobs = st.number_input("Number of Jobs", 1, 10, 4, key="strf_woq_jobs")
    with col2:
        num_cpus = st.number_input("Number of CPUs", 1, 4, 2, key="strf_woq_cpus")
    with col3:
        chunk_unit = st.number_input("Chunk Unit (e.g., 0.5, 1.0)", value=1.0, key="strf_woq_chunk")

    if st.button("Randomize Job Times", key="strf_woq_rand"):
        st.session_state.strf_special_jobs = [
            {'arrival': round(random.uniform(0, 5) * 2) / 2, 'burst': round(random.uniform(1, 10) * 2) / 2}
            for _ in range(num_jobs)
        ]

    processes = []
    for i in range(num_jobs):
        default = st.session_state.get("strf_special_jobs", [{}]*num_jobs)[i]
        c1, c2 = st.columns(2)
        with c1:
            arrival = st.number_input(f"Arrival Time for J{i+1}", value=default.get('arrival', 0.0), key=f"strf_woq_arr_{i}")
        with c2:
            burst = st.number_input(f"Burst Time for J{i+1}", value=default.get('burst', 3.0), key=f"strf_woq_burst_{i}")
        processes.append({'id': f'J{i+1}', 'arrival_time': arrival, 'burst_time': burst})

    if st.button("Run Simulation", key="strf_woq_run"):
        strf_simulate(processes, num_cpus, chunk_unit, quantum_time=None, algorithm_name="STRF without Quantum Time")

# --- Round Robin With Quantum Time ---
def run_rr_with_quantum():
    st.subheader("Round Robin Scheduling with Quantum Time")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        num_jobs = st.number_input("Number of Jobs", 1, 10, 4, key="rr_wq_jobs")
    with col2:
        num_cpus = st.number_input("Number of CPUs", 1, 4, 2, key="rr_wq_cpus")
    with col3:
        chunk_unit = st.number_input("Chunk Unit (e.g., 0.5, 1.0)", value=1.0, key="rr_wq_chunk")
    with col4:
        quantum_time = st.number_input("Quantum Time", value=2.0, key="rr_wq_quantum")

    if st.button("Randomize Job Times", key="rr_wq_rand"):
        st.session_state.rr_random_jobs = [
            {'arrival': round(random.uniform(0, 5) * 2) / 2, 'burst': round(random.uniform(1, 10) * 2) / 2}
            for _ in range(num_jobs)
        ]

    # Initialize JobScheduler
    scheduler = JobScheduler()
    
    # Add jobs to scheduler
    for i in range(num_jobs):
        default = st.session_state.get("rr_random_jobs", [{}]*num_jobs)[i]
        c1, c2 = st.columns(2)
        with c1:
            arrival = st.number_input(f"Arrival Time for J{i+1}", value=default.get('arrival', 0.0), key=f"rr_wq_arr_{i}")
        with c2:
            burst = st.number_input(f"Burst Time for J{i+1}", value=default.get('burst', 3.0), key=f"rr_wq_burst_{i}")
        scheduler.add_job(f'J{i+1}', arrival, burst)

    if st.button("Run Simulation", key="rr_wq_run"):
        avg_turnaround = scheduler.run_round_robin_with_quantum(num_cpus, quantum_time, chunk_unit)
        
        st.subheader("Result Table")
        df = scheduler.get_results_dataframe()
        st.dataframe(df, use_container_width=True)
        
        st.markdown(f"**Average Turnaround Time:** `{avg_turnaround:.2f}`")
        
        st.subheader("Gantt Chart")
        st.pyplot(draw_gantt_chart(
            scheduler.gantt_data, 
            scheduler.queue_snapshots, 
            scheduler.processes, 
            quantum_time,
            "Round Robin with Quantum Time"
        ), use_container_width=True)

# --- Round Robin Without Quantum Time ---
def run_rr_without_quantum():
    st.subheader("Round Robin Scheduling without Quantum Time")
    col1, col2, col3 = st.columns(3)
    with col1:
        num_jobs = st.number_input("Number of Jobs", 1, 10, 4, key="rr_woq_jobs")
    with col2:
        num_cpus = st.number_input("Number of CPUs", 1, 4, 2, key="rr_woq_cpus")
    with col3:
        chunk_unit = st.number_input("Chunk Unit (e.g., 0.5, 1.0)", value=1.0, key="rr_woq_chunk")

    if st.button("Randomize Job Times", key="rr_woq_rand"):
        st.session_state.rr_special_jobs = [
            {'arrival': round(random.uniform(0, 5) * 2) / 2, 'burst': round(random.uniform(1, 10) * 2) / 2}
            for _ in range(num_jobs)
        ]

    # Initialize JobScheduler
    scheduler = JobScheduler()
    
    # Add jobs to scheduler
    for i in range(num_jobs):
        default = st.session_state.get("rr_special_jobs", [{}]*num_jobs)[i]
        c1, c2 = st.columns(2)
        with c1:
            arrival = st.number_input(f"Arrival Time for J{i+1}", value=default.get('arrival', 0.0), key=f"rr_woq_arr_{i}")
        with c2:
            burst = st.number_input(f"Burst Time for J{i+1}", value=default.get('burst', 3.0), key=f"rr_woq_burst_{i}")
        scheduler.add_job(f'J{i+1}', arrival, burst)

    if st.button("Run Simulation", key="rr_woq_run"):
        avg_turnaround = scheduler.run_round_robin_without_quantum(num_cpus, chunk_unit)
        
        st.subheader("Result Table")
        df = scheduler.get_results_dataframe()
        st.dataframe(df, use_container_width=True)
        
        st.markdown(f"**Average Turnaround Time:** `{avg_turnaround:.2f}`")
        
        st.subheader("Gantt Chart")
        st.pyplot(draw_gantt_chart(
            scheduler.gantt_data, 
            scheduler.queue_snapshots, 
            scheduler.processes,
            quantum_time=None,
            algorithm_name="Round Robin without Quantum Time"
        ), use_container_width=True)

# --- Core STRF Simulation Logic ---
def strf_simulate(processes, num_cpus, chunk_unit, quantum_time=None, algorithm_name="STRF"):
    arrival_time = {p['id']: p['arrival_time'] for p in processes}
    burst_time = {p['id']: p['burst_time'] for p in processes}
    remaining_time = burst_time.copy()
    job_chunks, start_time, end_time = {}, {}, {}
    for job_id, bt in burst_time.items():
        chunks, rem = [], bt
        while rem > 0:
            chunk = min(chunk_unit, rem)
            chunks.append(chunk)
            rem -= chunk
        job_chunks[job_id] = chunks

    cpu_names = [f"CPU{i+1}" for i in range(num_cpus)]
    busy_until = {cpu: 0 for cpu in cpu_names}
    current_jobs = {cpu: None for cpu in cpu_names}
    busy_jobs = set()
    gantt_data, queue_snapshots = [], []
    current_time, jobs_completed = 0, 0
    next_sched = 0

    def capture_queue(time):
        queue = sorted([j for j in remaining_time if remaining_time[j] > 0 and arrival_time[j] <= time and j not in busy_jobs],
                       key=lambda j: (remaining_time[j], arrival_time[j]))
        if queue:
            queue_snapshots.append((time, [(j, round(remaining_time[j], 1)) for j in queue]))

    capture_queue(current_time)

    while jobs_completed < len(processes):
        for cpu in cpu_names:
            if busy_until[cpu] <= current_time and current_jobs[cpu]:
                busy_jobs.discard(current_jobs[cpu])
                current_jobs[cpu] = None

        available_cpus = [c for c in cpu_names if busy_until[c] <= current_time and current_jobs[c] is None]
        can_schedule = quantum_time is None or current_time >= next_sched

        if available_cpus and can_schedule:
            capture_queue(current_time)
            queue = sorted([j for j in remaining_time if remaining_time[j] > 0 and arrival_time[j] <= current_time and j not in busy_jobs],
                           key=lambda j: (remaining_time[j], arrival_time[j]))
            for cpu in available_cpus:
                if not queue: break
                job = queue.pop(0)
                chunk = job_chunks[job].pop(0)
                if job not in start_time:
                    start_time[job] = current_time
                current_jobs[cpu] = job
                busy_jobs.add(job)
                busy_until[cpu] = current_time + chunk
                remaining_time[job] -= chunk
                gantt_data.append((current_time, cpu, job, chunk))
                if remaining_time[job] < 1e-3:
                    end_time[job] = current_time + chunk
                    jobs_completed += 1
            if quantum_time:
                next_sched = current_time + quantum_time

        future_events = (
            [busy_until[c] for c in cpu_names if busy_until[c] > current_time] +
            [arrival_time[j] for j in arrival_time if arrival_time[j] > current_time and remaining_time[j] > 0]
        )
        if quantum_time and next_sched > current_time:
            future_events.append(next_sched)
        current_time = min(future_events) if future_events else current_time + 0.1

    for p in processes:
        p['start_time'] = start_time[p['id']]
        p['end_time'] = end_time[p['id']]
        p['turnaround_time'] = p['end_time'] - p['arrival_time']

    df = pd.DataFrame([{
        "Job": p['id'],
        "Arrival": p['arrival_time'],
        "Burst": p['burst_time'],
        "Start": round(p['start_time'], 1),
        "End": round(p['end_time'], 1),
        "Turnaround": round(p['turnaround_time'], 1)
    } for p in processes])
    avg_tat = sum(p['turnaround_time'] for p in processes) / len(processes)

    st.subheader("Result Table")
    st.dataframe(df, use_container_width=True)
    st.markdown(f"**Average Turnaround Time:** `{avg_tat:.2f}`")
    st.subheader("Gantt Chart")
    st.pyplot(draw_gantt_chart(gantt_data, queue_snapshots, processes, quantum_time, algorithm_name), use_container_width=True)

# --- Run Based on Choice ---
if algo == "STRF Scheduling with Quantum Time":
    run_strf_with_quantum()
elif algo == "STRF Scheduling Without Quantum Time":
    run_strf_without_quantum()
elif algo == "Round Robin with Quantum Time":
    run_rr_with_quantum()
elif algo == "Round Robin without Quantum Time":
    run_rr_without_quantum()
