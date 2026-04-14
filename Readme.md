# 等待優化架構


現在的程式碼確實有幾個痛點：`shallow_3D_new.py` 負責了太多流程控制，而 `SWE_func2.py` 裡面的一個 `SWE_functions` 類別包山包海（這在物件導向中被稱為 "God Object"），包含了物理方程、數值方法（RK4）、空間濾波（FFT）、甚至繪圖與檔案輸出。

為了讓你更好理解，我們可以用一個**「大氣流體實驗室」**的比喻：
現在的架構就像是實驗室裡只有一位超級研究員（`SWE_functions`），他不僅要懂流體力學公式，還要自己手動轉動機器的旋鈕（時間積分）、自己拿尺量網格、還要兼職做數據報表和畫圖。當你想引進新設備（優化演算法）時，這個研究員就會手忙腳亂。

我們現在要做的，就是**將這個實驗室「部門化」**。每一個模組只負責一件事情（單一職責原則，Single Responsibility Principle）。

以下是我為你設計的、具備高擴充性與優化潛力的全新架構藍圖。我們一步一步來看看各個部門該負責什麼：

### 📁 建議的專案目錄架構

```text
swe_model/
│
├── main.py                  # 實驗室總監：程式的唯一入口，負責啟動整個模擬
├── config.py                # 實驗室規章：純粹儲存參數，不進行任何運算
│
├── core/                    # 核心運算部門
│   ├── __init__.py
│   ├── grid.py              # 網格專家：處理 X, Y, kx, ky 等空間與頻域網格
│   ├── math_tools.py        # 數學專家：專注於純數學運算（FFT微分、Laplace、RK4）
│   └── physics.py           # 物理專家：只放 SWE 與 N-S 方程的實作
│
├── models/                  # 模型策略部門
│   ├── __init__.py
│   ├── base_model.py        # 定義模型的通用介面
│   ├── one_way.py           # 實作 One-way 模型 ('OW')
│   ├── mass_sink.py         # 實作 Mass sink 模型 ('MS')
│   └── momentum_flux.py     # 實作 Momentum flux 模型 ('MF')
│
├── io_utils/                # 輸出入與視覺化部門
│   ├── __init__.py
│   ├── writer.py            # 檔案總管：專職將陣列寫入 NetCDF
│   └── plotter.py           # 繪圖部門：專職用 matplotlib 畫圖
│
└── initial_conditions.py    # 初始狀態部門：產生渦旋、初始風場等
```

---

### 步驟解析與設計理念

#### 第一步：抽離設定與網格運算 (分離 `config.py` 與 `grid.py`)
在目前的 `setting.py` 中，你直接計算了 `X, Y = np.meshgrid(x,y)`。這會導致每次 import 這個檔案時，系統都會執行運算。
* **作法：** `config.py` 應該只包含純數值（如 `Lx = 700000`, `Nx = 512`, `dt = 5`）。
* **網格獨立：** 我們建立一個 `Grid` 類別放在 `grid.py` 中。這樣未來如果你想優化演算法（例如引入自適應網格 AMR），只需修改這個類別即可。

#### 第二步：純粹化數學算子 (建立 `math_tools.py`)
目前的 `Spatial_diff` 和 `Laplace` 是綁在物理類別裡的。但從數學定理來看，散度或拉普拉斯算子只是一個純粹的數學運算，它不需要知道自己正在處理的是「風速」還是「水深」。
* **推導與設計：** 在頻域中，空間微分可以表示為 $\frac{\partial f}{\partial x} \xrightarrow{\mathcal{F}} ik_x \hat{f}$。這個操作只需要傳入變數 $f$ 和頻率矩陣 $k_x$。
* **作法：** 將 RK4、頻譜濾波器（Wave filter）、以及空間微分全部移到 `math_tools.py`。這樣做的好處是，未來如果你想把 FFT 的計算從 CPU (NumPy/SciPy) 優化為 GPU (CuPy)，你只需要改這個檔案，而完全不用動到物理方程式。

#### 第三步：物理方程與控制邏輯解耦 (建立 `physics.py` 與 `models/`)
你現在有三個不同的實驗設定（'OW', 'MS', 'MF'），用 `if-elif` 寫在主迴圈裡。隨著邏輯變複雜，這會讓迴圈變得非常難閱讀。
* **作法：** 應用「策略模式 (Strategy Pattern)」。你可以定義一個父類別，然後讓三個模型繼承它並實作自己獨特的項（例如 $Q$ 質量匯或動量通量）。這樣主迴圈就不用寫一堆判斷式，只要無腦呼叫 `model.step()` 即可。

#### 第四步：解救主迴圈 (建立 `io_utils/`)
`shallow_3D_new.py` 裡面充滿了印出文字、判斷時間寫入 NetCDF 以及畫圖的程式碼。
* **作法：** 把 `write_single_data` 和 `plot_uvp` 移出去。主程式 `main.py` 的迴圈應該乾淨到只剩下：計算下一步 $\rightarrow$ 更新時間 $\rightarrow$ 呼叫 IO 管理器看需不需要輸出。
