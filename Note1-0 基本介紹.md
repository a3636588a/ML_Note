# 機器學習 筆記

## 前言
感測器訊號通常夾雜雜訊、環境波動且具有複雜的非線性。單純的 `if-else` 邏輯難以定義何謂「正常」與「異常」。

**MLP（多層感知機）的價值**：
它是一個**「萬能函數擬合器」**。它能夠學習訊號之間複雜的**特徵交互作用**（例如：只有當「壓力」與「溫度」同時上升時才被判定為異常），並在吵雜的訊號中「看到」隱藏的模式。

---

## 基礎定義與神經網路公式

在機器學習中，單個神經元的基礎計算公式為：
$$Z = \left( \sum_{i=1}^{n} w_i \cdot x_i \right) + b$$
當我們以矩陣（多個神經元與多筆資料）來表示時，公式為：
$$\mathbf{Z} = \mathbf{X} \cdot \mathbf{W} + \mathbf{B}$$

神經網路的基本結構包含三個層面，分別對應到公式中的變數：
1. **輸入層 (Input Layer)**：對應到向量或矩陣 $\mathbf{X}$。
2. **隱藏層 (Hidden Layer) / 權重**：對應到權重矩陣 $\mathbf{W}$。
3. **輸出層 (Output Layer)**：對應到輸出矩陣 $\mathbf{Z}$（在進入下一層前通常會再通過激活函數 $a = \sigma(\mathbf{Z})$）。
4. **偏移量 (Bias)**：對應到偏移向量 $\mathbf{B}$（或單個偏置值 $b$），代表偏移。

### 1. 輸入矩陣 $\mathbf{X}$ 的舉例 ($1 \times 4$ 矩陣)
輸入矩陣 $\mathbf{X}$ 代表輸入特徵。例如，我們有 4 個感測器特徵（例如：溫度、壓力、流量、振動頻率）：
$$\mathbf{X} = \begin{bmatrix} x_1 & x_2 & x_3 & x_4 \end{bmatrix}$$
這是一個 $1 \times 4$ 的矩陣（1 列 4 行），代表一筆包含 4 個特徵的觀測資料。
例如實際數值可以是：
$$\mathbf{X} = \begin{bmatrix} 25.5 & 101.3 & 5.2 & 0.8 \end{bmatrix}$$

### 2. 權重矩陣 $\mathbf{W}$ 的維度轉換 ($4 \times 3$ 矩陣)
$\mathbf{W}$ 矩陣決定了資料要從**「幾維空間」**轉換到**「幾維空間」**。
* **$\mathbf{W}$ 矩陣的列數 (Row)**：代表**「上一層的特徵數」**。
* **$\mathbf{W}$ 矩陣的行數 (Column)**：代表**「這一層想產生的神經元（特徵）數」**。

若上一層有 4 個特徵（即輸入 $\mathbf{X}$ 的維度為 4），且我們希望這一層產生 3 個神經元，則 $\mathbf{W}$ 必須是一個 $4 \times 3$ 的矩陣（4 列 3 行）：
$$\mathbf{W} = \begin{bmatrix} w_{11} & w_{12} & w_{13} \\ w_{21} & w_{22} & w_{23} \\ w_{31} & w_{32} & w_{33} \\ w_{41} & w_{42} & w_{43} \end{bmatrix}$$

當我們進行矩陣乘法 $\mathbf{X} \cdot \mathbf{W}$（即一個 $1 \times 4$ 矩陣與一個 $4 \times 3$ 矩陣相乘）時：
$$(1 \times 4) \times (4 \times 3) = (1 \times 3)$$
相乘後的結果會是一個 $1 \times 3$ 的矩陣，成功將資料從 4 維空間轉換到了 3 維空間！

### 3. 偏移矩陣 $\mathbf{B}$ (Bias)
$\mathbf{B}$ 是當你算完 $\mathbf{X} \cdot \mathbf{W}$ 後，會加上的一個偏移量。它的維度必須與輸出的維度一致（在此例中為 $1 \times 3$）：
$$\mathbf{B} = \begin{bmatrix} b_1 & b_2 & b_3 \end{bmatrix}$$

因此完整的計算過程為：
$$
\begin{aligned}
\mathbf{Z} &= \mathbf{X} \cdot \mathbf{W} + \mathbf{B} \\
\mathbf{Z} &= \begin{bmatrix} x_1 & x_2 & x_3 & x_4 \end{bmatrix} \begin{bmatrix} w_{11} & w_{12} & w_{13} \\ w_{21} & w_{22} & w_{23} \\ w_{31} & w_{32} & w_{33} \\ w_{41} & w_{42} & w_{43} \end{bmatrix} + \begin{bmatrix} b_1 & b_2 & b_3 \end{bmatrix} \\
\mathbf{Z} &= \begin{bmatrix} (x_1 w_{11} + x_2 w_{21} + x_3 w_{31} + x_4 w_{41} + b_1) & (x_1 w_{12} + x_2 w_{22} + x_3 w_{32} + x_4 w_{42} + b_2) & (x_1 w_{13} + x_2 w_{23} + x_3 w_{33} + x_4 w_{43} + b_3) \end{bmatrix}
\end{aligned}
$$
輸出 $\mathbf{Z}$ 即為一個 $1 \times 3$ 的矩陣，代表這一層 3 個神經元的輸出值。

---

## FNN vs CNN 的權重算法與比較

在不同的神經網路架構中，權重矩陣 $\mathbf{W}$ 扮演了不同的角色，其計算方式與物理意義也有所不同。以下為全連接神經網路 (FNN) 與卷積神經網路 (CNN) 的比較：

![FNN vs CNN 權重定義比較](image/media__1781143897392.png)

### 1. 全連接層 (FNN - Fully Connected Neural Network)
* **權重名稱**：Weights (權重)
* **在程式中的任務**：負責**全局的維度轉換**（例如將 $5 \times 4$ 的維度進行變換）。
* **物理意義（白話）**：「綜合所有人的意見，算出得分。」
* **數學公式**：
  對於第 $j$ 個輸出神經元 $z_j$：
  $$z_j = \left( \sum_{i=1}^{D_{in}} x_i \cdot w_{ij} \right) + b_j$$
  其中 $D_{in}$ 是輸入特徵的總數，每個輸入特徵 $x_i$ 都與一個專屬權重 $w_{ij}$ 相乘。

### 2. 卷積網路 (CNN - Convolutional Neural Network)
* **權重名稱**：Kernel / Filter (卷積核 / 濾波器)
* **在程式中的任務**：在輸入資料（如時間軸或空間維度）上**滑動計算特徵**。
* **物理意義（白話）**：「像放大鏡一樣，不管特徵出現在哪都抓得出來。」
* **數學公式**：
  以二維圖像卷積為例，對於輸出特徵圖的特定位置 $(i, j)$，其計算公式為：
  $$z_{i, j} = \left( \sum_{m=0}^{K_H-1} \sum_{n=0}^{K_W-1} x_{i+m, \, j+n} \cdot w_{m, n} \right) + b$$
  其中 $K_H$ 和 $K_W$ 分別是卷積核（Kernel）的高度與寬度，$w_{m, n}$ 是卷積核在位置 $(m, n)$ 的權重。

#### 卷積運算過程示意圖
![CNN 卷積運算示意圖](image/1_ZCjPUFrB6eHPRi4eyP6aaA.gif)

如上圖所示：
* **輸入 (Image)**：一個 $5 \times 5$ 的矩陣。
* **卷積核 (Kernel/Filter)**：一個 $3 \times 3$ 的矩陣（圖中黃色遮罩對應的乘數，如底部的紅字 $\times 1$、$\times 0$、$\times 1$）。
* **運算方式**：卷積核在輸入矩陣上由左至右、由上至下滑動。每次覆蓋一個 $3 \times 3$ 的區域，並將對應位置的元素相乘後相加：
  $$z_{0, 0} = (1 \times 1) + (1 \times 0) + (1 \times 1) + (0 \times 0) + (1 \times 1) + (1 \times 0) + (0 \times 1) + (0 \times 0) + (1 \times 1) = 1 + 0 + 1 + 0 + 1 + 0 + 0 + 0 + 1 = 4$$
* **特徵圖 (Convolved Feature)**：滑動計算後得到的輸出，此例中為一個 $3 \times 3$ 的矩陣，第一個計算結果為 **4**。
