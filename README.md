# OS 期中專案 - CPU 排程演算法模擬器

本專案旨在使用 **Python** 實作並比較三種常見的 CPU 排程演算法。  
透過模擬一組行程（Process）的執行，我們將計算並分析各演算法的效能指標，包含：  
- 平均等待時間（Average Waiting Time）  
- 平均周轉時間（Average Turnaround Time）
- 回應時間 (Response Time)
- 吞吐量 (Throughput)
並將結果以圖表方式視覺化呈現。

---

## 專案特色

- **語言**: Python 3  
- **視覺化**: 使用 `matplotlib` 產生效能比較圖表  
- **易於擴充**: 程式碼結構清晰，方便未來新增更多排程演算法  

---

## 實作的演算法

本專案實作了以下三種演算法：

1. **先到先服務 (First-Come, First-Served - FCFS)**
3. **循環式排程 (Round Robin - RR)**
4. **多層回饋佇列排程 (Multilevel Feedback Queue Scheduling - MLFQ)**
---

## 環境需求與安裝

Python 3.10.11。

```bash
# 複製此專案
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

# 安裝所需套件
pip install matplotlib
```

---

## 如何執行

直接執行主程式即可啟動模擬：

```bash
python [your_file_name].py
```

程式執行後，將會在終端機（Terminal）印出各演算法的效能分析數據，  
並自動跳出一個視覺化的比較圖表。

---


**產生的圖表：**
`performance_chart.png`

---

## 組員

- 蔡昌諭(組長)
- 謝尚哲
- 鍾承翰
- 羅宜茜
- 謝秉倫