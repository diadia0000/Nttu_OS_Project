import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import deque
import os

# 創建輸出目錄
OUTPUT_DIR = "picture_output/RR"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def round_robin_scheduling(processes, time_quantum):
    """
    Round Robin 排程演算法實作

    參數:
        processes: list of tuples (process_id, arrival_time, burst_time)
        time_quantum: 時間量子 (time quantum)

    返回:
        字典包含所有效能指標
    """
    print(f"\n{'='*60}")
    print(f"Round Robin Scheduling (Time Quantum = {time_quantum})")
    print(f"{'='*60}\n")

    # 初始化資料結構
    ready_queue = deque()
    proc_data = {}

    # 建立行程資料
    for p in processes:
        pid, arrival, burst = p
        proc_data[pid] = {
            'arrival': arrival,
            'burst': burst,
            'remaining': burst,
            'start_time': -1,      # 首次執行時間 (用於計算回應時間)
            'completion_time': 0,   # 完成時間
            'waiting_time': 0,      # 等待時間
            'turnaround_time': 0,   # 周轉時間
            'response_time': 0      # 回應時間
        }

    # 依到達時間排序
    processes_sorted = sorted(processes, key=lambda x: (x[1], x[0]))

    current_time = 0
    completed_count = 0
    n = len(processes)
    proc_index = 0
    gantt_chart = []

    # 目前執行的行程
    current_proc = None

    # 模擬執行
    while completed_count < n:
        # 將到達的行程加入就緒佇列
        while proc_index < n and processes_sorted[proc_index][1] <= current_time:
            pid = processes_sorted[proc_index][0]
            if proc_data[pid]['remaining'] > 0:
                ready_queue.append(pid)
            proc_index += 1

        # 如果有行程正在執行完時間量子，將其放回就緒佇列
        if current_proc is not None:
            ready_queue.append(current_proc)
            current_proc = None

        # 從就緒佇列取出行程執行
        if ready_queue:
            current_proc = ready_queue.popleft()

            # 記錄首次執行時間（回應時間）
            if proc_data[current_proc]['start_time'] == -1:
                proc_data[current_proc]['start_time'] = current_time
                proc_data[current_proc]['response_time'] = current_time - proc_data[current_proc]['arrival']

            # 計算執行時間（不超過時間量子和剩餘時間）
            exec_time = min(time_quantum, proc_data[current_proc]['remaining'])

            # 記錄甘特圖
            gantt_chart.append((current_proc, current_time, current_time + exec_time))

            # 更新時間和剩餘時間
            current_time += exec_time
            proc_data[current_proc]['remaining'] -= exec_time

            # 將在執行期間到達的行程加入就緒佇列
            while proc_index < n and processes_sorted[proc_index][1] <= current_time:
                pid = processes_sorted[proc_index][0]
                if proc_data[pid]['remaining'] > 0:
                    ready_queue.append(pid)
                proc_index += 1

            # 檢查行程是否完成
            if proc_data[current_proc]['remaining'] == 0:
                proc_data[current_proc]['completion_time'] = current_time
                proc_data[current_proc]['turnaround_time'] = current_time - proc_data[current_proc]['arrival']
                proc_data[current_proc]['waiting_time'] = (proc_data[current_proc]['turnaround_time'] -
                                                           proc_data[current_proc]['burst'])
                completed_count += 1
                current_proc = None

        else:
            # CPU 閒置，跳到下一個行程到達時間
            if proc_index < n:
                idle_until = processes_sorted[proc_index][1]
                if idle_until > current_time:
                    gantt_chart.append(('IDLE', current_time, idle_until))
                    current_time = idle_until
            else:
                break

    # 計算效能指標
    total_waiting_time = sum(proc_data[pid]['waiting_time'] for pid in proc_data)
    total_turnaround_time = sum(proc_data[pid]['turnaround_time'] for pid in proc_data)
    total_response_time = sum(proc_data[pid]['response_time'] for pid in proc_data)

    avg_waiting_time = total_waiting_time / n
    avg_turnaround_time = total_turnaround_time / n
    avg_response_time = total_response_time / n

    # 計算 CPU 使用率
    total_burst_time = sum(p[2] for p in processes)
    cpu_utilization = (total_burst_time / current_time) * 100

    # 計算吞吐量 (processes per unit time)
    throughput = n / current_time

    # 輸出結果
    print(f"{'Process':<10} {'Arrival':<10} {'Burst':<10} {'Completion':<12} "
          f"{'Waiting':<10} {'Turnaround':<12} {'Response':<10}")
    print("-" * 80)

    for pid in sorted(proc_data.keys()):
        p = proc_data[pid]
        print(f"{pid:<10} {p['arrival']:<10} {p['burst']:<10} {p['completion_time']:<12} "
              f"{p['waiting_time']:<10} {p['turnaround_time']:<12} {p['response_time']:<10}")

    print("\n" + "="*60)
    print(f"平均等待時間 (Average Waiting Time):        {avg_waiting_time:.2f}")
    print(f"平均周轉時間 (Average Turnaround Time):     {avg_turnaround_time:.2f}")
    print(f"平均回應時間 (Average Response Time):       {avg_response_time:.2f}")
    print(f"CPU 使用率 (CPU Utilization):              {cpu_utilization:.2f}%")
    print(f"吞吐量 (Throughput):                       {throughput:.4f} processes/unit time")
    print("="*60)

    # 繪製甘特圖
    draw_gantt_chart(gantt_chart, f"Round Robin Scheduling (TQ={time_quantum})", time_quantum)

    return {
        'process_data': proc_data,
        'gantt_chart': gantt_chart,
        'avg_waiting_time': avg_waiting_time,
        'avg_turnaround_time': avg_turnaround_time,
        'avg_response_time': avg_response_time,
        'cpu_utilization': cpu_utilization,
        'throughput': throughput
    }


def draw_gantt_chart(gantt_chart, title, time_quantum=None):
    """
    繪製甘特圖

    參數:
        gantt_chart: list of tuples (process_id, start_time, end_time)
        title: 圖表標題
        time_quantum: 時間量子（用於檔名）
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # 獲取所有唯一的行程 ID
    process_ids = list(set([item[0] for item in gantt_chart if item[0] != 'IDLE']))
    process_ids.sort()

    # 為每個行程分配顏色
    colors = list(mcolors.TABLEAU_COLORS.values())
    color_map = {pid: colors[i % len(colors)] for i, pid in enumerate(process_ids)}
    color_map['IDLE'] = 'lightgray'

    # 繪製甘特圖
    for item in gantt_chart:
        pid, start, end = item
        duration = end - start
        ax.barh(0, duration, left=start, height=0.5,
                color=color_map[pid], edgecolor='black', linewidth=1)

        # 在中間添加標籤
        mid = start + duration / 2
        ax.text(mid, 0, str(pid), ha='center', va='center',
                fontweight='bold', fontsize=9, color='white' if pid != 'IDLE' else 'black')

    # 添加時間刻度
    time_points = sorted(set([item[1] for item in gantt_chart] + [gantt_chart[-1][2]]))
    ax.set_xticks(time_points)
    ax.set_xticklabels(time_points, fontsize=9)

    # 設定圖表樣式
    ax.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax.set_yticks([])
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    # 添加圖例
    legend_elements = [plt.Rectangle((0, 0), 1, 1, fc=color_map[pid], edgecolor='black')
                      for pid in process_ids]
    ax.legend(legend_elements, process_ids, loc='upper right',
              title='Processes', framealpha=0.9)

    plt.tight_layout()

    # 儲存圖表
    filename = f"RR_TQ_{time_quantum}_gantt.png" if time_quantum is not None else "RR_gantt.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath)
    print(f"甘特圖已儲存至: {filepath}")

    plt.show()


def compare_time_quantums(processes, quantums):
    """
    比較不同時間量子的效能

    參數:
        processes: list of tuples (process_id, arrival_time, burst_time)
        quantums: list of time quantum values to compare
    """
    results = {}

    for tq in quantums:
        print(f"\n\n{'#'*60}")
        print(f"Testing with Time Quantum = {tq}")
        print(f"{'#'*60}")
        result = round_robin_scheduling(processes, tq)
        results[tq] = result

    # 繪製比較圖
    draw_comparison_chart(results)

    return results


def draw_comparison_chart(results):
    """
    繪製不同時間量子的效能比較圖

    參數:
        results: dictionary with time_quantum as key and metrics as value
    """
    quantums = list(results.keys())
    avg_waiting = [results[tq]['avg_waiting_time'] for tq in quantums]
    avg_turnaround = [results[tq]['avg_turnaround_time'] for tq in quantums]
    avg_response = [results[tq]['avg_response_time'] for tq in quantums]
    cpu_util = [results[tq]['cpu_utilization'] for tq in quantums]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # 平均等待時間
    ax1.plot(quantums, avg_waiting, marker='o', linewidth=2, markersize=8, color='blue')
    ax1.set_xlabel('Time Quantum', fontweight='bold')
    ax1.set_ylabel('Average Waiting Time', fontweight='bold')
    ax1.set_title('Average Waiting Time vs Time Quantum', fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # 平均周轉時間
    ax2.plot(quantums, avg_turnaround, marker='s', linewidth=2, markersize=8, color='green')
    ax2.set_xlabel('Time Quantum', fontweight='bold')
    ax2.set_ylabel('Average Turnaround Time', fontweight='bold')
    ax2.set_title('Average Turnaround Time vs Time Quantum', fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 平均回應時間
    ax3.plot(quantums, avg_response, marker='^', linewidth=2, markersize=8, color='red')
    ax3.set_xlabel('Time Quantum', fontweight='bold')
    ax3.set_ylabel('Average Response Time', fontweight='bold')
    ax3.set_title('Average Response Time vs Time Quantum', fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # CPU 使用率

    # 儲存圖表
    filename = "RR_comparison_chart.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath)
    print(f"比較圖已儲存至: {filepath}")

    ax4.plot(quantums, cpu_util, marker='D', linewidth=2, markersize=8, color='purple')
    ax4.set_xlabel('Time Quantum', fontweight='bold')
    ax4.set_ylabel('CPU Utilization (%)', fontweight='bold')
    ax4.set_title('CPU Utilization vs Time Quantum', fontweight='bold')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 範例測試資料
    # 格式: (Process ID, Arrival Time, Burst Time)
    test_processes = [
        ('P1', 0, 24),
        ('P2', 1, 3),
        ('P3', 2, 3),
        ('P4', 3, 5),
        ('P5', 4, 8)
    ]

    print("="*60)
    print("Round Robin CPU Scheduling Algorithm 模擬器")
    print("="*60)
    print("\n測試資料:")
    print(f"{'Process':<10} {'Arrival Time':<15} {'Burst Time':<15}")
    print("-" * 40)
    for p in test_processes:
        print(f"{p[0]:<10} {p[1]:<15} {p[2]:<15}")

    # 單一時間量子測試
    time_quantum = 4
    result = round_robin_scheduling(test_processes, time_quantum)

    # 比較不同時間量子
    print("\n\n" + "="*60)
    print("比較不同時間量子的效能")
    print("="*60)
    quantums_to_compare = [2, 4, 6, 8]
    comparison_results = compare_time_quantums(test_processes, quantums_to_compare)
