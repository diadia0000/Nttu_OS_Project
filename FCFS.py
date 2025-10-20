import matplotlib.pyplot as plt
import numpy as np

class Process:
    def __init__(self, pid, arrival_time, burst_time):
        self.pid = pid
        self.arrival_time = arrival_time
        self.burst_time = burst_time
        self.completion_time = 0
        self.turnaround_time = 0
        self.waiting_time = 0
        self.response_time = -1

def fcfs_scheduling(processes):
    current_time = 0
    total_waiting_time = 0
    total_turnaround_time = 0
    total_response_time = 0
    total_burst_time = sum(p.burst_time for p in processes)

    # Sort processes by arrival time
    processes.sort(key=lambda x: x.arrival_time)

    for process in processes:
        if current_time < process.arrival_time:
            current_time = process.arrival_time

        if process.response_time == -1:
            process.response_time = current_time - process.arrival_time

        process.completion_time = current_time + process.burst_time
        process.turnaround_time = process.completion_time - process.arrival_time
        process.waiting_time = process.turnaround_time - process.burst_time

        total_waiting_time += process.waiting_time
        total_turnaround_time += process.turnaround_time
        total_response_time += process.response_time

        current_time = process.completion_time

    avg_waiting_time = total_waiting_time / len(processes)
    avg_turnaround_time = total_turnaround_time / len(processes)
    avg_response_time = total_response_time / len(processes)
    cpu_utilization = (total_burst_time / current_time) * 100
    throughput = len(processes) / current_time

    return processes, avg_waiting_time, avg_turnaround_time, avg_response_time, cpu_utilization, throughput

def plot_gantt_chart(processes):
    fig, gnt = plt.subplots()

    gnt.set_xlabel('Time')
    gnt.set_ylabel('Processes')

    y_ticks = [15 + i*10 for i in range(len(processes))]
    y_labels = [f'P{p.pid}' for p in processes]

    gnt.set_yticks(y_ticks)
    gnt.set_yticklabels(y_labels)

    gnt.grid(True)

    for i, process in enumerate(processes):
        gnt.broken_barh([(process.completion_time - process.burst_time, process.burst_time)], (y_ticks[i]-5, 10), facecolors=('tab:blue'))

    plt.title('FCFS Gantt Chart')
    plt.savefig(r'E:\Class\OS\Nttu_OS_Project\picture_output\FCFS\FCFS_Gantt.png')
    plt.show()

if __name__ == "__main__":
    # Simulate 5 processes
    processes = [
        Process(1, 0, 3),
        Process(2, 5, 4),
        Process(3, 6, 2),
        Process(4, 12, 5),
        Process(5, 15, 3)
    ]

    scheduled_processes, avg_wt, avg_tat, avg_rt, cpu_util, throughput = fcfs_scheduling(processes.copy())
    print("FCFS Scheduling Results:")
    print(f"Average Waiting Time: {avg_wt:.2f}")
    print(f"Average Turnaround Time: {avg_tat:.2f}")
    print(f"Average Response Time: {avg_rt:.2f}")
    print(f"Throughput: {throughput:.2f} processes/unit time")

    print("\nProcess Details:")
    print("PID\tArrival\tBurst\tCompletion\tTurnaround\tWaiting\tResponse")
    for p in scheduled_processes:
        print(f"{p.pid}\t{p.arrival_time}\t{p.burst_time}\t{p.completion_time}\t\t{p.turnaround_time}\t\t{p.waiting_time}\t{p.response_time}")

    plot_gantt_chart(scheduled_processes)

