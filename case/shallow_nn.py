"""
浅层神经网络（单隐层）

结构：输入 -> 隐层(tanh) -> 输出(sigmoid)
"""

import numpy as np

from logistic_regression import load_german_credit


class ShallowNN:
    """单隐层前馈神经网络，用于二分类。"""

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 16,
        learning_rate: float = 0.01,
        l2: float = 0.01,
    ):
        self.learning_rate = learning_rate
        self.l2 = l2
        self.hidden_size = hidden_size

        # Xavier 初始化，适配 tanh 隐层
        self.W1 = np.random.randn(n_features, hidden_size) * np.sqrt(1.0 / n_features)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size) * np.sqrt(1.0 / hidden_size)
        self.b2 = 0.0

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    @staticmethod
    def tanh(z: np.ndarray) -> np.ndarray:
        return np.tanh(z)

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """前向传播，返回 z1, a1, z2, a2。"""
        z1 = X @ self.W1 + self.b1
        a1 = self.tanh(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = self.sigmoid(z2)
        return z1, a1, z2, a2

    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        eps = 1e-8
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def backward(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        z1: np.ndarray,
        a1: np.ndarray,
        a2: np.ndarray,
    ) -> None:
        n_samples = X.shape[0]

        delta_z2 = a2 - y_true
        dW2 = a1.T @ delta_z2 / n_samples + self.l2 * self.W2
        db2 = np.mean(delta_z2)

        delta_a1 = delta_z2[:, np.newaxis] * self.W2
        delta_z1 = delta_a1 * (1.0 - a1 ** 2)

        dW1 = X.T @ delta_z1 / n_samples + self.l2 * self.W1
        db1 = np.mean(delta_z1, axis=0)

        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 500,
        batch_size: int = 32,
        verbose: bool = False,
    ) -> list[float]:
        losses = []
        n_samples = X.shape[0]

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                X_batch = X[indices[start:end]]
                y_batch = y[indices[start:end]]

                z1, a1, _, a2 = self.forward(X_batch)
                self.backward(X_batch, y_batch, z1, a1, a2)

            _, _, _, a2 = self.forward(X)
            loss = self.compute_loss(a2, y)
            losses.append(loss)

            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

        return losses

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        _, _, _, a2 = self.forward(X)
        return a2

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


if __name__ == "__main__":
    np.random.seed(42)

    X_train, X_test, y_train, y_test = load_german_credit()

    hidden_size = 16
    learning_rate = 0.01
    l2 = 0.01
    epochs = 500

    print(f"训练集: {X_train.shape[0]} 样本, {X_train.shape[1]} 特征")
    print(f"测试集: {X_test.shape[0]} 样本")
    print(
        f"隐层: {hidden_size} (tanh) | 输出: sigmoid | "
        f"lr={learning_rate} | l2={l2} | epochs={epochs} | batch=32\n"
    )

    model = ShallowNN(
        n_features=X_train.shape[1],
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        l2=l2,
    )
    model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=True)

    train_acc = np.mean(model.predict(X_train) == y_train)
    test_acc = np.mean(model.predict(X_test) == y_test)

    print(f"\n训练集准确率: {train_acc:.2%}")
    print(f"测试集准确率: {test_acc:.2%}")
