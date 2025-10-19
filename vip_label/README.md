# 自動標註系統 Auto-Labeling System

自動使用 YOLO 模型對影片進行標註，並上傳到 Roboflow 的簡化工作流程。

## 目錄結構

```
vip_label/
├── datasets/
│   ├── videos/              # 放置原始影片 (*.mp4, *.avi, *.mov, *.mkv)
│   └── roboflow_sync/       # Roboflow 上傳區 (auto_label.py 自動生成)
│       ├── images/          # 自動標註的圖片
│       └── labels/          # 自動標註的標籤
├── models/
│   └── best.pt              # YOLO 訓練好的模型
├── config.py                # ⭐ 配置文件 - 所有設定都在這裡
├── auto_label.py            # 步驟 1: 自動標註
└── sync_to_roboflow.py      # 步驟 2: 上傳到 Roboflow
```

## 安裝依賴

```bash
pip install ultralytics roboflow opencv-python
```

**依賴套件說明:**
- `ultralytics`: YOLO 模型推論
- `roboflow`: 上傳資料到 Roboflow 平台
- `opencv-python`: 圖片處理和影片讀取

## 使用流程

### 準備工作

1. **放置模型檔案**
   - 將訓練好的 YOLO 模型 `best.pt` 放到 `models/` 目錄

2. **放置影片檔案**
   - 將要標註的影片放到 `datasets/videos/` 目錄
   - 支援格式: `.mp4`, `.avi`, `.mov`, `.mkv`

3. **設定配置** (重要!)
   - 開啟 [`config.py`](config.py) 檔案
   - 修改以下關鍵參數:

```python
# YOLO 模型設定
MODEL_PATH = MODELS_DIR / "best.pt"        # 模型路徑
VID_STRIDE = 10                             # 每 10 幀取 1 幀
CONFIDENCE_THRESHOLD = 0.25                 # 信心閾值
IOU_THRESHOLD = 0.45                        # IoU 閾值

# Roboflow 設定
ROBOFLOW_WORKSPACE = "your-workspace"       # ⭐ 改成你的工作區名稱
ROBOFLOW_PROJECT = "drowning-detection"     # ⭐ 改成你的專案名稱
ROBOFLOW_API_KEY = ""                       # ⭐ 填入你的 API Key

# 批次處理設定
MAX_IMAGES_PER_BATCH = 1000                 # 每批次最多處理圖片數 (0=不限制)
MIN_LABELS_PER_IMAGE = 0                    # 過濾掉沒有標註的圖片 (0=保留所有)
```

### 步驟 1: 自動標註

執行自動標註腳本，使用 YOLO 模型對影片進行預測並直接儲存到上傳區:

```bash
python auto_label.py
```

**功能說明:**
- 讀取 `datasets/videos/` 中的所有影片
- 使用 YOLO 模型進行物件偵測
- 按照 `VID_STRIDE` 設定抽取影格
- 自動生成 YOLO 格式標註檔案 (`.txt`)
- 過濾掉不符合條件的圖片 (依據 `MIN_LABELS_PER_IMAGE`)
- **直接儲存到 `datasets/roboflow_sync/` 上傳區**

**輸出範例:**
```
Found 2 video file(s)
Batch name: auto_labeled_20251018_224229
Loading model: best.pt

[1/2] Processing: video1.mp4
   Saved 120 frames

[2/2] Processing: video2.mp4
   Saved 85 frames

Auto-labeling complete!
Total saved: 205 frames
Output location: datasets\roboflow_sync
   Images: datasets\roboflow_sync\images
   Labels: datasets\roboflow_sync\labels

Next step:
   Run sync_to_roboflow.py to upload to Roboflow
```

**目錄結構 (自動標註後):**
```
datasets/roboflow_sync/
├── images/
│   ├── auto_labeled_20251018_224229_video1_frame000000_20251018224530.jpg
│   ├── auto_labeled_20251018_224229_video1_frame000001_20251018224530.jpg
│   └── ...
└── labels/
    ├── auto_labeled_20251018_224229_video1_frame000000_20251018224530.txt
    ├── auto_labeled_20251018_224229_video1_frame000001_20251018224530.txt
    └── ...
```

### 步驟 2: 上傳到 Roboflow

將自動標註的資料上傳到 Roboflow:

```bash
python sync_to_roboflow.py
```

**功能說明:**
- 檢查 Roboflow CLI 安裝狀態
- 驗證圖片和標註數量
- 使用 Roboflow CLI 上傳資料
- 自動生成批次名稱 (格式: `auto_labeled_YYYYMMDD`)
- 上傳完成後可選擇清空上傳目錄

**輸出範例:**
```
📊 準備上傳:
   圖片數量: 205
   標註數量: 205

🚀 開始上傳到 Roboflow...
   工作區/專案: your-workspace/drowning-detection
   批次名稱: auto_labeled_20250118

執行指令:
   roboflow upload --project your-workspace/drowning-detection --batch auto_labeled_20250118 --images datasets\roboflow_sync\images --annotations datasets\roboflow_sync\labels

✅ 上傳完成!
   已上傳 205 張圖片到 Roboflow
```

## 配置參數說明

所有可調整的參數都集中在 [`config.py`](config.py) 中，主要分為以下幾類:

### 路徑配置
- `VIDEOS_DIR`: 原始影片目錄
- `AUTO_LABELED_DIR`: 自動標註暫存區
- `ROBOFLOW_SYNC_DIR`: Roboflow 上傳區
- `MODEL_PATH`: YOLO 模型路徑

### YOLO 模型參數
- `VID_STRIDE`: 影片抽幀間隔 (數值越大，處理速度越快，但標註數量越少)
- `CONFIDENCE_THRESHOLD`: 信心閾值 (0-1，越高越嚴格)
- `IOU_THRESHOLD`: IoU 閾值 (用於 NMS，0-1)
- `MAX_DET`: 每張圖片最大檢測數量

### 資料處理參數
- `BATCH_NAME_PREFIX`: 批次名稱前綴
- `MIN_LABELS_PER_IMAGE`: 每張圖片最少標註數 (過濾用，0=保留所有)
- `MAX_IMAGES_PER_BATCH`: 每批次最大圖片數 (限制上傳量)

### Roboflow 參數
- `ROBOFLOW_WORKSPACE`: Roboflow 工作區名稱 ⭐ 必填
- `ROBOFLOW_PROJECT`: Roboflow 專案名稱 ⭐ 必填
- `ROBOFLOW_API_KEY`: Roboflow API Key (或使用環境變數)

### 標註審核參數
- `ENABLE_LABEL_REVIEW`: 是否啟用互動式標註審核 (True/False)
- `CLASS_NAMES`: 類別名稱列表，例如 `['person', 'car']` 或 `None`

### 其他參數
- `VERBOSE`: 是否顯示 YOLO 詳細訊息

## 常見問題

### 1. 找不到模型檔案
```
❌ 錯誤: 找不到模型檔案 models\best.pt
```
**解決方案:** 將訓練好的 YOLO 模型放到 `models/` 目錄下，檔名為 `best.pt`

### 2. 找不到影片檔案
```
❌ 錯誤: 在 datasets\videos 中找不到影片檔案
```
**解決方案:** 將影片放到 `datasets/videos/` 目錄，支援格式: `.mp4`, `.avi`, `.mov`, `.mkv`

### 3. Roboflow CLI 未安裝
```
Error: Roboflow CLI not installed
```
**解決方案:** 執行 `pip install roboflow`

### 4. 上傳失敗
**可能原因:**
- API Key 錯誤或未設定
- 工作區/專案名稱錯誤
- 網路連線問題

**解決方案:**
1. 檢查 [`config.py`](config.py) 中的 `ROBOFLOW_WORKSPACE` 和 `ROBOFLOW_PROJECT`
2. 設定正確的 `ROBOFLOW_API_KEY`
3. 或使用環境變數: `set ROBOFLOW_API_KEY=your_api_key`

## 進階使用

### 調整抽幀間隔

如果影片太長，可以增加 `VID_STRIDE` 來減少標註數量:

```python
# config.py
VID_STRIDE = 30  # 每 30 幀取 1 幀 (原本是 10)
```

### 過濾無標註圖片

如果只想上傳有標註的圖片:

```python
# config.py
MIN_LABELS_PER_IMAGE = 1  # 至少要有 1 個標註
```

### 限制每批次處理數量

避免一次處理太多資料:

```python
# config.py
MAX_IMAGES_PER_BATCH = 500  # 每批次最多 500 張 (0=不限制)
```

## 工作流程圖

```
原始影片 (datasets/videos/)
    ↓
[auto_label.py] 使用 YOLO 自動標註
    ↓ (直接儲存到 roboflow_sync/)
上傳準備區 (datasets/roboflow_sync/)
    ├── images/  (標註好的圖片)
    └── labels/  (YOLO 格式標註)
    ↓
[sync_to_roboflow.py] 上傳到 Roboflow
    ↓
Roboflow 平台 ✅
```

## 授權

MIT License

## 聯絡資訊

如有問題或建議，請開 Issue 或聯絡開發者。
