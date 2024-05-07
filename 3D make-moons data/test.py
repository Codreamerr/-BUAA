import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def make_moons_3d(n_samples=500, noise=0.1):
    t = np.linspace(0, 2 * np.pi, n_samples)
    x = 1.5 * np.cos(t)
    y = np.sin(t)
    z = np.sin(2 * t)
    X = np.vstack([np.column_stack([x, y, z]), np.column_stack([-x, y - 1, -z])])
    y = np.hstack([np.zeros(n_samples), np.ones(n_samples)])
    X += np.random.normal(scale=noise, size=X.shape)
    return X, y

# 生成训练数据和测试数据
X_train, y_train = make_moons_3d(n_samples=1000, noise=0.2)
X_test, y_test = make_moons_3d(n_samples=500, noise=0.2)

# 数据标准化处理
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 可视化函数
def visualize_results(X, y_true, y_pred, title):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    correct = y_true == y_pred
    ax.scatter(X[correct & (y_true == 0), 0], X[correct & (y_true == 0), 1], X[correct & (y_true == 0), 2], color='green', label='Class 0 Correct', s=20)
    ax.scatter(X[correct & (y_true == 1), 0], X[correct & (y_true == 1), 1], X[correct & (y_true == 1), 2], color='blue', label='Class 1 Correct', s=20)
    ax.scatter(X[~correct & (y_true == 0), 0], X[~correct & (y_true == 0), 1], X[~correct & (y_true == 0), 2], color='red', label='Class 0 Incorrect', s=20)
    ax.scatter(X[~correct & (y_true == 1), 0], X[~correct & (y_true == 1), 1], X[~correct & (y_true == 1), 2], color='orange', label='Class 1 Incorrect', s=20)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    plt.show()

# 逻辑回归模型
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train_scaled, y_train)
y_pred_logistic = logistic_model.predict(X_test_scaled)
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
visualize_results(X_test, y_test, y_pred_logistic, f'Logistic Regression (Accuracy: {accuracy_logistic:.2f})')

# SVM (线性核)
svm_linear_model = SVC(kernel='linear')
svm_linear_model.fit(X_train_scaled, y_train)
y_pred_svm_linear = svm_linear_model.predict(X_test_scaled)
accuracy_svm_linear = accuracy_score(y_test, y_pred_svm_linear)
visualize_results(X_test, y_test, y_pred_svm_linear, f'SVM (Linear Kernel) (Accuracy: {accuracy_svm_linear:.2f})')

# SVM (多项式核)
svm_poly_model = SVC(kernel='poly', degree=3)
svm_poly_model.fit(X_train_scaled, y_train)
y_pred_svm_poly = svm_poly_model.predict(X_test_scaled)
accuracy_svm_poly = accuracy_score(y_test, y_pred_svm_poly)
visualize_results(X_test, y_test, y_pred_svm_poly, f'SVM (Polynomial Kernel) (Accuracy: {accuracy_svm_poly:.2f})')

# SVM (RBF核)
svm_rbf_model = SVC(kernel='rbf')
svm_rbf_model.fit(X_train_scaled, y_train)
y_pred_svm_rbf = svm_rbf_model.predict(X_test_scaled)
accuracy_svm_rbf = accuracy_score(y_test, y_pred_svm_rbf)
visualize_results(X_test, y_test, y_pred_svm_rbf, f'SVM (RBF Kernel) (Accuracy: {accuracy_svm_rbf:.2f})')

# XGBoost模型
xgboost_model = XGBClassifier(n_estimators=100, max_depth=5, use_label_encoder=False, eval_metric='logloss')
xgboost_model.fit(X_train_scaled, y_train)
y_pred_xgboost = xgboost_model.predict(X_test_scaled)
accuracy_xgboost = accuracy_score(y_test, y_pred_xgboost)
visualize_results(X_test, y_test, y_pred_xgboost, f'XGBoost (Accuracy: {accuracy_xgboost:.2f})')
# 
# def visualize_z_projection(X, y_true, y_pred, title):
#     fig, ax = plt.subplots(figsize=(10, 8))
#     correct = y_true == y_pred
#
#     # 正确分类的点
#     ax.scatter(X[correct & (y_true == 0), 0], X[correct & (y_true == 0), 1], color='green', label='Class 0 Correct',
#                s=50, edgecolors='k')
#     ax.scatter(X[correct & (y_true == 1), 0], X[correct & (y_true == 1), 1], color='blue', label='Class 1 Correct',
#                s=50, edgecolors='k')
#
#     # 错误分类的点
#     ax.scatter(X[~correct & (y_true == 0), 0], X[~correct & (y_true == 0), 1], color='red', label='Class 0 Incorrect',
#                s=50, edgecolors='k')
#     ax.scatter(X[~correct & (y_true == 1), 0], X[~correct & (y_true == 1), 1], color='orange',
#                label='Class 1 Incorrect', s=50, edgecolors='k')
#
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_title(title)
#     ax.legend()
#     plt.grid(True)
#     plt.show()
#
#
# # 使用修改后的可视化函数展示z方向投影的分类结果
# visualize_z_projection(X_test, y_test, y_pred_logistic, f'Logistic Regression (Accuracy: {accuracy_logistic:.2f})')
# visualize_z_projection(X_test, y_test, y_pred_svm_linear, f'SVM (Linear Kernel) (Accuracy: {accuracy_svm_linear:.2f})')
# visualize_z_projection(X_test, y_test, y_pred_svm_poly, f'SVM (Polynomial Kernel) (Accuracy: {accuracy_svm_poly:.2f})')
# visualize_z_projection(X_test, y_test, y_pred_svm_rbf, f'SVM (RBF Kernel) (Accuracy: {accuracy_svm_rbf:.2f})')
# visualize_z_projection(X_test, y_test, y_pred_xgboost, f'XGBoost (Accuracy: {accuracy_xgboost:.2f})')
