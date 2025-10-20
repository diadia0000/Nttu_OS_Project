import pandas as pd
from collections import deque
import matplotlib.pyplot as plt
import os

def mlfq_scheduling_with_history(processes, quantums, aging_period):
    print(f"--- MLFQ Scheduling (Q0={quantums[0]}, Q1={quantums[1]}, Q2=FCFS, Aging={aging_period}) ---")

    queues = [deque(), deque(), deque()]
    proc_data = {}
    proc_list_by_arrival = []

    for p in processes:
        pid, arrival, burst = p
        proc_data[pid] = {
            'arrival': arrival, 'burst': burst, 'remaining': burst,
            'queue': 0, 'time_in_quantum': 0,
            'start_time': -1, 'completion_time': 0,
        }
        proc_list_by_arrival.append((arrival, pid))

    proc_list_by_arrival.sort()

    current_time = 0
    completed_count = 0
    n = len(processes)
    proc_arrival_idx = 0
    running_proc_pid = None

    gantt_history = []

    while completed_count < n:

        # --- 規則 5: 優先權提升 (Aging) ---
        if current_time > 0 and current_time % aging_period == 0:
            print(f"[Time {current_time}] *** PRIORITY BOOST (AGING) ***")
            if running_proc_pid is not None:
                pid = running_proc_pid
                proc_data[pid]['queue'] = 0
                proc_data[pid]['time_in_quantum'] = 0
                queues[0].append(pid)
                running_proc_pid = None

            for q_idx in [1, 2]:
                while queues[q_idx]:
                    pid = queues[q_idx].popleft()
                    proc_data[pid]['queue'] = 0
                    proc_data[pid]['time_in_quantum'] = 0
                    queues[0].append(pid)

        # --- 規則 2: 新行程抵達 ---
        while proc_arrival_idx < n and proc_list_by_arrival[proc_arrival_idx][0] == current_time:
            arrival, pid = proc_list_by_arrival[proc_arrival_idx]
            print(f"[Time {current_time}] Process {pid} ARRIVES. Added to Q0.")
            proc_data[pid]['queue'] = 0
            queues[0].append(pid)
            proc_arrival_idx += 1

        # --- 規則 3: 檢查先佔 (Preemption) ---
        if running_proc_pid is not None:
            current_q = proc_data[running_proc_pid]['queue']
            if (current_q > 0 and queues[0]) or (current_q > 1 and queues[1]):
                print(f"[Time {current_time}] Process {running_proc_pid} (Q{current_q}) PREEMPTED.")
                queues[current_q].appendleft(running_proc_pid)
                running_proc_pid = None

        # --- 規則 4: 檢查執行完畢或降級 ---
        if running_proc_pid is not None:
            pid = running_proc_pid
            data = proc_data[pid]

            # A. 檢查是否執行完畢
            if data['remaining'] == 0:
                print(f"[Time {current_time}] Process {pid} FINISHED.")
                data['completion_time'] = current_time
                completed_count += 1
                running_proc_pid = None

            # B. 檢查 Quantum 是否用完 (Q0, Q1)
            elif data['queue'] < 2 and data['time_in_quantum'] == quantums[data['queue']]:
                print(
                    f"[Time {current_time}] Process {pid} (Q{data['queue']}) quantum expired. Demoting to Q{data['queue'] + 1}.")
                data['queue'] += 1
                data['time_in_quantum'] = 0
                queues[data['queue']].append(pid)
                running_proc_pid = None

        # --- 規則 3: 選擇新行程執行 ---
        if running_proc_pid is None:
            if queues[0]:
                running_proc_pid = queues[0].popleft()
                print(f"[Time {current_time}] CPU picks {running_proc_pid} from Q0.")
            elif queues[1]:
                running_proc_pid = queues[1].popleft()
                print(f"[Time {current_time}] CPU picks {running_proc_pid} from Q1.")
            elif queues[2]:
                running_proc_pid = queues[2].popleft()
                print(f"[Time {current_time}] CPU picks {running_proc_pid} from Q2.")
            else:
                # CPU 閒置
                gantt_history.append("Idle")  # <--- 記錄 Idle
                current_time += 1
                continue

                # --- 執行 1 個時間單位 ---
        pid = running_proc_pid
        data = proc_data[pid]

        if data['start_time'] == -1:
            data['start_time'] = current_time

        data['remaining'] -= 1
        data['time_in_quantum'] += 1

        gantt_history.append(f"P{pid}")  # <--- 記錄正在執行的 PID

        # --- 時鐘前進 ---
        current_time += 1

    # 返回結果和執行歷史
    return proc_data, gantt_history


# (這是之前的結果印出函數，保持不變)
def print_mlfq_results(proc_data, processes):
    results = []
    for p_orig in processes:
        pid = p_orig[0]
        data = proc_data[pid]
        arrival, burst, ct, start = data['arrival'], data['burst'], data['completion_time'], data['start_time']
        tat = ct - arrival
        wt = tat - burst
        rt = start - arrival
        results.append([pid, arrival, burst, ct, tat, wt, rt])

    df = pd.DataFrame(results, columns=["PID", "Arrival", "Burst", "Completion", "TAT", "WT", "Response"])
    print("\n--- MLFQ Final Results ---")
    print(df)
    print(f"\n平均周轉時間 (Average TAT): {df['TAT'].mean():.2f}")
    print(f"平均等待時間 (Average WT): {df['WT'].mean():.2f}")
    print(f"平均回應時間 (Average RT): {df['Response'].mean():.2f}\n")


def plot_gantt_chart(gantt_history, processes, save_path=None):
    """
    使用 matplotlib 繪製甘特圖
    """

    # 1. 解析 history: 將 ['P1', 'P1', 'P2'] 轉換為 [('P1', 0, 2), ('P2', 2, 3)]
    blocks = []
    if not gantt_history:
        print("Gantt history is empty.")
        return

    last_pid = gantt_history[0]
    start_time = 0
    for t, pid in enumerate(gantt_history):
        if pid != last_pid:
            # 一個區塊結束
            blocks.append({'pid': last_pid, 'start': start_time, 'end': t})
            # 新區塊開始
            last_pid = pid
            start_time = t
    # 加入最後一個區塊
    blocks.append({'pid': last_pid, 'start': start_time, 'end': len(gantt_history)})

    # 2. 設置畫布
    fig, ax = plt.subplots(figsize=(16, 5))

    # 3. 建立 PID 到 Y 軸位置 和 顏色的映射
    pids = sorted(list(set(f"P{p[0]}" for p in processes)))
    if "Idle" in [b['pid'] for b in blocks]:
        pids.append("Idle")

    y_map = {pid: i for i, pid in enumerate(pids)}

    # 使用 Pastel1 調色盤
    colors = plt.cm.Pastel1(range(len(pids)))
    color_map = {pid: colors[i] for i, pid in enumerate(pids)}
    color_map['Idle'] = '#D3D3D3'  # 灰色

    # 4. 繪製每一個區塊
    for block in blocks:
        pid = block['pid']
        start = block['start']
        end = block['end']
        duration = end - start

        if duration == 0: continue

        # 繪製長條
        ax.barh(y=y_map[pid], width=duration, left=start, height=0.7,
                color=color_map[pid], edgecolor='black', alpha=0.9)

        # 在長條中間加上文字
        label = f"{pid}\n({duration})" if duration > 1 else pid
        if duration > 0:
            ax.text(start + duration / 2, y_map[pid], label,
                    ha='center', va='center', color='black', fontsize=9)

    # 5. 美化圖表
    ax.set_xlabel("Time Units", fontsize=12)
    ax.set_ylabel("Process", fontsize=12)
    ax.set_title("MLFQ Scheduling Gantt Chart", fontsize=16)

    # 設置 Y 軸刻度
    ax.set_yticks(range(len(pids)))
    ax.set_yticklabels(pids)

    # 設置 X 軸刻度 (顯示每個時間單位)
    max_time = len(gantt_history)
    ax.set_xticks(range(0, max_time + 1, 1))
    ax.set_xlim(0, max_time)

    # 加上網格
    ax.grid(axis='x', which='major', linestyle='--', linewidth=0.5)

    # 翻轉 Y 軸 (讓 P1 在最上面)
    ax.invert_yaxis()

    plt.tight_layout()
    if save_path:
        # 確保資料夾存在
        output_dir = os.path.dirname(save_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(save_path)
        print(f"Gantt chart saved to: {save_path}")
        plt.close(fig)  # 關閉圖表以釋放記憶體
    else:
        plt.show()

# --- 主程式 ---

# [PID, Arrival Time, Burst Time]
processes_mlfq = [
    [1, 0, 5],   # 中短工作 (Q0 -> Q1)
    [2, 2, 2],   # 極短工作 (Q0 finish)
    [3, 4, 15],  # 長工作 (Q0 -> Q1 -> Q2)
    [4, 8, 3]    # 晚到的短工作 (Q0 finish, preempts P3)
]

# Q0 = 4, Q1 = 8, Q2 = FCFS
quantums = [4, 8, None]
aging_period = 40

# 執行 MLFQ 並取得執行歷史
final_proc_data, history = mlfq_scheduling_with_history(processes_mlfq, quantums, aging_period)

# 印出摘要表
print_mlfq_results(final_proc_data, processes_mlfq)

# 繪製並儲存甘特圖
save_file_path = "picture_output/MLFQ/MLFQ_Gantt_Chart.png"
plot_gantt_chart(history, processes_mlfq, save_path=save_file_path)