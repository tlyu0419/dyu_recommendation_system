from sklearn.datasets import make_regression
import numpy as np
X, y = make_regression(n_samples=5, n_features=2, noise=1, random_state=42)

X1 = X[:,0]
X2 = X[:,1]
y

w1 = 0
w2 = 0
b=0
lr = 0.1
epochs = 50

for epoch in range(epochs):
    y_pred = w1 * X1 + w2*X2 + b
    error = y_pred - y
    loss = np.mean(error ** 2)

    # 計算梯度
    dw1 = 2 * np.mean(error * X1)
    dw2 = 2 * np.mean(error * X2)
    db = 2 * np.mean(error)

    # 更新參數
    w1 -= lr * dw1
    w2 -= lr * dw2
    b -= lr * db

    print(f"Epoch {epoch}: Loss={loss:.4f}, w={w1:.4f}, w={w2:.4f}, b={b:.4f}")
