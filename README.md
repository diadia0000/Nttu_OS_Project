OS 期中專案 - CPU 排程演算法模擬器本專案旨在使用 Python 實作並比較三種常見的 CPU 排程演算法。透過模擬一組行程（Process）的執行，我們將計算並分析各演算法的效能指標，包含平均等待時間（Average Waiting Time）與平均周轉時間（Average Turnaround Time），並將結果視覺化。🚀 專案特色語言: Python 3視覺化: 使用 matplotlib 產生效能比較圖表。易於擴充: 程式碼結構清晰，方便未來新增更多排程演算法。🤖 實作的演算法本專案實作了以下三種演算法：先到先服務 (First-Come, First-Served - FCFS)最短工作優先 (Shortest Job First - SJF)循環式排程 (Round Robin - RR)🛠️ 環境需求與安裝確認你已安裝 Python 3.x。複製此專案：git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
安裝所需的套件：pip install matplotlib
▶️ 如何執行直接執行主程式即可啟動模擬：python main.py
程式執行後，將會在終端機（Terminal）印出各演算法的效能分析數據，並自動跳出一個視覺化的比較圖表。📊 輸出結果範例終端機輸出：*** CPU Scheduling Performance Analysis ***

[FCFS]
Average Waiting Time: 17.00
Average Turnaround Time: 27.20

[SJF]
Average Waiting Time: 8.20
Average Turnaround Time: 18.40

[Round Robin (Quantum=4)]
Average Waiting Time: 19.60
Average Turnaround Time: 29.80
產生的圖表 (performance_chart.png):👥 組員[你的名字][組員 B 的名字][組員 C 的名字]
