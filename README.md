# OpenCV Haar Cascade + LBPH 即時人臉辨識系統 (LBPH_Face)

本專案使用 Visual Studio Code 開發，基於 OpenCV 的 Haar Cascade 分類器進行人臉偵測，並結合 LBPH (Local Binary Patterns Histograms)演算法實現即時人臉特徵提取與身份比對。

---

## 📌 專案架構 (Project Structure)

```text
LBPH_Face/
├── collect_faces.py          # 1. 影像收集：透過 Webcam 實時偵測並裁切人臉樣本
├── train_lbph.py             # 2. 模型訓練：讀取人臉集並生成 LBPH 模型與標籤對應檔
├── recognize_lbph.py         # 3. 即時辨識：結合 Webcam 實時進行 LBPH 特徵比對與標示
├── haarcascade_frontalface_alt.xml  # Haar Cascade 人臉偵測分類器模型檔
├── data/                     # 人臉圖像資料庫
│   ├── student01/            # 學生/人員 01 的灰階人臉照片
│   └── student02/            # 學生/人員 02 的灰階人臉照片
└── models/                   # 訓練結果儲存目錄
    ├── lbph_model.yml        # LBPH 權重與特徵模型檔
    └── labels.json           # 類別標籤與對應姓名檔
```

---

## 🔄 作業流程 (Workflow)

系統整體辨識與處理邏輯如下：

```text
Webcam 影像
    │
    ▼
Haar 人臉偵測
    │
    ▼
裁切灰階人臉
    │
    ▼
LBPH 特徵分析
    │
    ▼
與訓練資料比對
    │
    ▼
距離 ≤ 閾值 ？
  ├── 是 ──► 顯示姓名
  └── 否 ──► 顯示 Unknown
```

---

## 🛠️ 環境需求與安裝 (Prerequisites)

* Python 3.8+
* Visual Studio Code
* OpenCV 套件 (包含 contrib 模組以使用 LBPH)

請使用以下指令安裝必要套件：

```bash
pip install opencv-python opencv-contrib-python numpy
```

---

## 🚀 使用步驟 (Usage)

### 1. 收集人臉樣本
執行 `collect_faces.py`，輸入人員名稱或學號 (例如 `student01`)，面向攝影機擷取人臉數據：
```bash
python collect_faces.py
```

### 2. 訓練辨識模型
資料收集完成後，執行 `train_lbph.py` 讀取 `data/` 資料夾並自動產出模型至 `models/` 目錄：
```bash
python train_lbph.py
```

### 3. 執行即時人臉辨識
開啟 `recognize_lbph.py` 啟動 Webcam 進行實時辨識：
```bash
python recognize_lbph.py
```
> 按下 **'q'** 鍵即可退出視窗。