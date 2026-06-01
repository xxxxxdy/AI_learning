"""
简单的逻辑回归神经网络（单神经元 + Sigmoid 激活）

逻辑回归可以看作只有一层的神经网络：
  输入 x -> 线性变换 z = w·x + b -> Sigmoid -> 输出概率 p
"""

import numpy as np


class LogisticRegressionNN:
    """单神经元逻辑回归，使用梯度下降训练。"""

    def __init__(self, n_features: int, learning_rate: float = 0.1):
        self.learning_rate = learning_rate
        # 权重和偏置（随机初始化）
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0.0

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        """Sigmoid 激活函数，将线性输出映射到 (0, 1)。"""
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def forward(self, X: np.ndarray) -> np.ndarray:
        """前向传播：计算预测概率。"""
        z = X @ self.weights + self.bias
        return self.sigmoid(z)

    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """二元交叉熵损失。"""
        eps = 1e-8
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def backward(self, X: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray) -> None:
        """反向传播：计算梯度并更新参数。"""
        n_samples = X.shape[0]
        error = y_pred - y_true

        dw = (X.T @ error) / n_samples
        db = np.mean(error)

        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 1000,
        verbose: bool = False,
    ) -> list[float]:
        """训练模型，返回每个 epoch 的损失。"""
        losses = []
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = self.compute_loss(y_pred, y)
            self.backward(X, y_pred, y)
            losses.append(loss)

            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

        return losses

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """返回属于正类的概率。"""
        return self.forward(X)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """返回类别预测（0 或 1）。"""
        return (self.predict_proba(X) >= threshold).astype(int)


def load_german_credit(
    path: str = "german.data-numeric",
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    加载 German Credit 数值版数据集。

    最后一列为标签：1=好客户，2=坏客户；转换为 0/1（坏客户=1）。
    特征做标准化，并按比例划分训练集/测试集。
    """
    data = np.loadtxt(path)
    X = data[:, :-1]
    y = (data[:, -1] == 2).astype(float)

    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    X = (X - mean) / std

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(y))
    split = int(len(y) * (1 - test_ratio))

    train_idx, test_idx = indices[:split], indices[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_german_credit()

    print(f"训练集: {X_train.shape[0]} 样本, {X_train.shape[1]} 特征")
    print(f"测试集: {X_test.shape[0]} 样本")
    print(f"坏客户比例: 训练 {y_train.mean():.1%}, 测试 {y_test.mean():.1%}\n")

    model = LogisticRegressionNN(n_features=X_train.shape[1], learning_rate=0.1)
    model.fit(X_train, y_train, epochs=1000, verbose=True)

    train_acc = np.mean(model.predict(X_train) == y_train)
    test_acc = np.mean(model.predict(X_test) == y_test)

    print(f"\n训练集准确率: {train_acc:.2%}")
    print(f"测试集准确率: {test_acc:.2%}")
