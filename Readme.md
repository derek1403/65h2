# 雙層淺水方程－邊界層耦合模型
**Shallow Water – Boundary Layer Coupled Model (SWE-BL)**

---

## 專案簡介

本模型以**偽頻譜法 (Pseudo-spectral Method)** 為數值核心，模擬自由大氣（Free Atmosphere）與行星邊界層（Planetary Boundary Layer, PBL）之間的動力耦合過程。空間微分運算在頻域中進行（$\partial/\partial x \rightarrow ik_x$），時間積分採用四階 Runge-Kutta（RK4）方法。

模型的研究對象為一**雙層系統**：

- **自由大氣層**：以淺水方程（Shallow Water Equations, SWE）描述，追蹤水平風速 $(u, v)$ 與自由液面位移 $h$。
- **邊界層**：以含有地表摩擦、科氏力與壓力梯度力的簡化 Navier-Stokes 方程描述，追蹤近地面風速 $(u_{sfc}, v_{sfc})$ 及垂直速度 $w_{sfc}$。

透過切換實驗模式（OW / MS / MF），可系統性地比較不同邊界層回饋機制對自由大氣渦旋演化的影響。

---

## 環境需求

```bash
pip install numpy scipy numexpr matplotlib netCDF4
```

---

## 執行方式

```bash
python main.py
```

模擬結果（NetCDF 格式）輸出至 `data/`；圖像輸出至 `plot/uvp/` 與 `plot/vor/`。

所有參數（網格大小、物理常數、輸出頻率、實驗模式）皆在 `config.py` 中設定，**無需修改其他檔案**。

---

## 專案結構

```
swe_model/
│
├── main.py                   # 程式唯一入口：初始化、時間迴圈、輸出排程
├── config.py                 # 所有可調參數集中於此
├── initial_conditions.py     # 初始渦旋場設定
│
├── core/
│   ├── grid.py               # 空間與頻域網格建立（X, Y, kx, ky, k²）
│   ├── math_tools.py         # 數學算子：wave_filter、Spatial_diff、Laplace、RK4
│   └── physics.py            # 物理方程：SWE、N_S_EQ、ini_wind、damping
│
├── models/
│   ├── base_model.py         # 共用時間步進介面（BaseModel）
│   ├── one_way.py            # 實驗模式：One-way (OW)
│   ├── mass_sink.py          # 實驗模式：Mass Sink (MS)
│   └── momentum_flux.py      # 實驗模式：Momentum Flux (MF)
│
└── io_utils/
    ├── writer.py             # NetCDF 檔案輸出
    └── plotter.py            # 風場、高度場與渦度場繪圖
```

---

## 控制方程

### 自由大氣：淺水方程

$$\frac{\partial u}{\partial t} = -g\frac{\partial h}{\partial x} + fv - \mathbf{u}\cdot\nabla u - w_p\frac{u - u_{sfc}}{H} + \nu_1 \nabla^2 u$$

$$\frac{\partial v}{\partial t} = -g\frac{\partial h}{\partial y} - fu - \mathbf{u}\cdot\nabla v - w_p\frac{v - v_{sfc}}{H} + \nu_1 \nabla^2 v$$

$$\frac{\partial h}{\partial t} = -(H+h)\left(\frac{\partial u}{\partial x}+\frac{\partial v}{\partial y}\right) - \mathbf{u}\cdot\nabla h - (H+h)Q$$

其中 $w_p = \frac{1}{2}(|w_{sfc}| + w_{sfc})$ 為向上輸送的垂直速度篩選項（僅保留 $w_{sfc}>0$ 的部分）； $Q$ 為質量匯項（由實驗模式決定）。

### 邊界層：簡化 Navier-Stokes 方程

$$\frac{\partial u_{sfc}}{\partial t} = -\frac{1}{\rho}\frac{\partial P}{\partial x} + fv_{sfc} - \mathbf{u}_{sfc}\cdot\nabla u_{sfc} - w_p^{-}\frac{u_{sfc} - u}{H} + \nu_2\nabla^2 u_{sfc} - \frac{C_D |\mathbf{u}_{sfc}| u_{sfc}}{H}$$

$$\frac{\partial v_{sfc}}{\partial t} = -\frac{1}{\rho}\frac{\partial P}{\partial y} - fu_{sfc} - \mathbf{u}_{sfc}\cdot\nabla v_{sfc} - w_p^{-}\frac{v_{sfc} - v}{H} + \nu_2\nabla^2 v_{sfc} - \frac{C_D |\mathbf{u}_{sfc}| v_{sfc}}{H}$$

其中 $w_p^{-} = \frac{1}{2}(|w_{sfc}| - w_{sfc})$ 為向下輸送篩選項；地表拖曳係數 $C_D$ 依風速分段給定（仿 Large & Pond 1981 參數化方案）。

邊界層垂直速度由連續方程診斷：

$$w_{sfc} = -H\left(\frac{\partial u_{sfc}}{\partial x} + \frac{\partial v_{sfc}}{\partial y}\right)$$

---

## 三種實驗模式

在 `config.py` 中設定 `mm = 0 / 1 / 2` 即可切換。

### One-way (OW)：`mm = 0`

邊界層**完全不回饋**自由大氣，作為對照組。

$$Q = 0, \quad w_{in} = 0, \quad u_{in} = 0, \quad v_{in} = 0$$

自由大氣方程退化為無邊界層強迫的標準 SWE。此模式用於評估純大氣動力過程的基準演化。

---

### Mass Sink (MS)：`mm = 1`

邊界層向上的質量輸送被等效為自由大氣的**質量匯 (Mass Sink)**。

$$Q = w_{sfc} \times Q_0, \quad w_{in} = 0, \quad u_{in} = 0, \quad v_{in} = 0$$

邊界層 Ekman 抽吸效應透過 $h$ 的傾向方程中的 $-(H+h)Q$ 項作用於自由大氣，但不攜帶動量資訊。

---

### Momentum Flux (MF)：`mm = 2`

邊界層向上的質量通量**同時攜帶動量**，對自由大氣施加完整的動量強迫。

$$Q = 0, \quad w_{in} = w_{sfc}, \quad u_{in} = u_{sfc}, \quad v_{in} = v_{sfc}$$

$u$ 與 $v$ 方程中的 $-w_p(u - u_{sfc})/H$ 項完整啟動，代表邊界層流體夾卷至自由大氣時攜入的水平動量。此為最完整的耦合模式。

---

## 數值方法

### 偽頻譜法空間微分

所有空間微分在頻域中進行，以避免有限差分的截斷誤差：

$$\frac{\partial f}{\partial x} = \mathcal{F}^{-1}\left(i k_x \hat{f}\right), \qquad \nabla^2 f = \mathcal{F}^{-1}\left(-k^2 \hat{f}\right)$$

### 頻譜濾波器（`wave_filter`）

為抑制非線性交互作用產生的混疊誤差（Aliasing）並維持數值穩定性，每次空間微分前均施加頻譜濾波：

$$\hat{f}_{filtered}(k_x, k_y) = \hat{f}(k_x, k_y) \cdot \underbrace{\text{sinc}\!\left(\frac{k_x}{2k_{x,max}}\right)}_{\text{Lanczos 平滑}} \cdot \underbrace{\text{sinc}\!\left(\frac{k_y}{2k_{y,max}}\right)}_{\text{Lanczos 平滑}}$$

超出截斷波數 $k_{max} = N/3$（對應 **2/3 規則**去混疊）的所有模態直接設為零。

### 時間積分

採用標準四階 Runge-Kutta 方法：

$$y^{n+1} = y^n + \frac{\Delta t}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$

---

## 參數設定（`config.py`）

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `Lx`, `Ly` | 700,000 m | 計算域大小 |
| `Nx`, `Ny` | 512 | 網格點數 |
| `dt` | 5 s | 時間步長 |
| `hours` | 36 | 模擬總時數 |
| `SP` | 3.0 hr | Spin-up 時間 |
| `H` | 1000 m | 等效水深 |
| `g` | 9.81 m s⁻² | 重力加速度 |
| `f` | 5×10⁻⁵ s⁻¹ | 科氏力參數 |
| `nu1` | 100 m² s⁻¹ | 自由大氣黏滯係數 |
| `nu2` | 5000 m² s⁻¹ | 邊界層黏滯係數 |
| `Q0` | 2×10⁻⁵ | 質量通量係數 |
| `mm` | 2 | 實驗模式（0=OW, 1=MS, 2=MF）|
| `OT_data` | 1800 s | NetCDF 輸出間隔 |
| `OT_plot` | 600 s | 圖像輸出間隔 |

---

## 初始場設定（`initial_conditions.py`）

預設初始渦旋為**橢圓形渦旋（Ellipse Vortex）**，以平滑函數 $S(s) = 1 - 3s^2 + 2s^3$ 建立連續的渦度邊界：

$$
\zeta(x,y) = \begin{cases} 
\zeta_0 & r_1 < 1 \\ 
\zeta_0 \  S\left(\dfrac{1-r_1}{r_2-r_1}\right) & 1 \leq r_1 \leq r_2^{-1} \\
0 & r_2 > 1 
\end{cases}
$$

其中 $r_1, r_2$ 為橢圓無因次半徑，定義如下：

$$r_{1} = \sqrt{\left(\frac{x-x_0}{a_{1}}\right)^2 + \left(\frac{y-y_0}{b_{1}}\right)^2}$$

$$r_{2} = \sqrt{\left(\frac{x-x_0}{a_{2}}\right)^2 + \left(\frac{y-y_0}{b_{2}}\right)^2}$$

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `zeta0` | 2×10⁻³ s⁻¹ | 渦旋核心最大渦度 |
| `a1`, `b1` | 20 km, 40 km | 渦旋核心半軸（短軸/長軸）|
| `a2`, `b2` | 24 km, 44 km | 渦旋平滑過渡區外緣半軸 |

如需自訂初始場（如多個渦旋、不同形狀或由觀測資料初始化），只需修改 `initial_conditions.py` 中的 `make_initial_vorticity()` 函數，並回傳相同維度的渦度場陣列即可，其餘程式碼無需更動。

---

## 輸出格式

### NetCDF（`data/`）

每個輸出時刻儲存一個 `.nc` 檔案，包含以下變數：

| 變數 | 說明 |
|------|------|
| `u`, `v` | 自由大氣水平風速 (m s⁻¹) |
| `w` | 自由大氣垂直速度 (m s⁻¹) |
| `h` | 自由液面位移 (m) |
| `P` | 自由大氣壓力 (Pa) |
| `u_sfc`, `v_sfc` | 邊界層水平風速 (m s⁻¹) |
| `w_sfc` | 邊界層頂垂直速度 (m s⁻¹) |

### 圖像（`plot/`）

- `plot/uvp/`：風場向量圖（自由大氣 + 邊界層）疊加 $h$ 與 $P$ 填色圖
- `plot/vor/`：自由大氣與邊界層渦度場（聚焦渦旋中心 ±70 km 範圍）

---

## 數值穩定性注意事項

- **時間步長**： $\Delta t$ 須滿足 CFL 條件。當 `H` 或 `nu` 增大時，建議適當縮小 `dt`。
- **Spin-up 時間**：前 `SP` 小時邊界層不對自由大氣施加強迫，避免初始不平衡造成的數值衝擊。
- **頻譜濾波**：`wave_filter` 中的 Lanczos 平滑與 2/3 去混疊規則對長時間積分的穩定性至關重要，**請勿移除**。

# 示意圖
```json?chameleon
{"component":"LlmGeneratedComponent","props":{"height":"600px","prompt":"建立一個交互式橢圓渦旋 (Ellipse Vortex) 生成模擬器。這對應使用者程式碼中的 initial_conditions.py。\n\n1. 設定網格：512x512，顯示範圍約 700km x 700km。\n2. 提供 Slider 控制項：\n   - a1 (短軸長度): 範圍 10km 到 50km。\n   - b1 (長軸長度): 範圍 20km 到 100km。\n   - zeta0 (最大渦度值): 範圍 5e-4 到 5e-3。\n3. 即時渲染：\n   - 使用 Canvas 繪製渦度的 Heatmap。\n   - 呈現橢圓形狀隨參數變化的過程。\n4. 數學參考：顯示 r1 = sqrt(((X-x0)/a1)^2 + ((Y-y0)/b1)^2) 的公式。\n5. 語言要求：繁體中文。幫助使用者理解如何自定義初始條件。","id":"im_951888f86fb5d38b"}}
```