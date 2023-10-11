# 深度學習

## 1.神經網絡

### 1.1 活化函數

#### Sigmoid 函數

```py
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
```

#### ReLU 函數

```py
def relu(x):
    return np.maximum(0, x)
```

#### 恆等函數

```py
def identity_function(x):
    return x
```

#### Softmax 函數

```py
def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x) # 防範溢位 ex.100,99,98 -> 0,-1,-2
    return np.exp(x) / np.sum(np.exp(x))
```

`迴歸問題`使用恆等函數
`分類問題`使用 Softmax 函數

#### 安裝 Git Large File Storage

git lfs install
cd C:\Users\ASUS\Documents\GitHub\deep-learning
git lfs track "\*.pth"
