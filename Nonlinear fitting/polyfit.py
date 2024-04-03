#多项式拟合
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

file_path_train = 'G:\\machine learning\\complex_nonlinear_data.xlsx'
file_path_test = 'G:\\machine learning\\new_complex_nonlinear_data.xlsx'

# 读取训练和测试数据
df_train = pd.read_excel(file_path_train)
df_test = pd.read_excel(file_path_test)

X_train = df_train['Feature'].values.reshape(-1, 1)
y_train = df_train['Target'].values
X_test = df_test['Feature'].values.reshape(-1, 1)
y_test = df_test['Target'].values

# 初始化最低MSE和最佳degree
min_mse = float('inf')
best_degree = 0
best_model = None

# 尝试不同的degree值
polynomial_features = None
best_polynomial_features = None
for degree in range(1, 30):
    # 在每次迭代中更新多项式特征变换器的degree
    current_polynomial_features = PolynomialFeatures(degree=degree)
    X_poly_train = current_polynomial_features.fit_transform(X_train)
    X_poly_test = current_polynomial_features.transform(X_test)

    model = LinearRegression()
    model.fit(X_poly_train, y_train)

    y_pred_train = model.predict(X_poly_train)
    mse_train = mean_squared_error(y_train, y_pred_train)

    # 更新最低MSE、最佳degree和最佳模型
    if mse_train < min_mse:
        min_mse = mse_train
        best_degree = degree
        best_model = model
        best_polynomial_features = current_polynomial_features

X_poly_test = best_polynomial_features.transform(X_test)

# 使用最佳模型对测试集进行预测
y_pred_test = best_model.predict(X_poly_test)
mse_test = mean_squared_error(y_test, y_pred_test)
# 绘制训练集上的图形
plt.figure(figsize=(5, 5))
plt.scatter(X_train, y_train, color='blue', label='Training data points')
sorted_order = np.argsort(X_train.flatten())
X_train_sorted = X_train[sorted_order]
best_y_pred_train = best_model.predict(best_polynomial_features.transform(X_train_sorted))
plt.plot(X_train_sorted[:, 0], best_y_pred_train, color='red', label=f'Best fit curve (degree={best_degree}) on train set')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Polynomial fit with lowest MSE on Training Set')
plt.legend()
plt.show()

# 绘制测试集上的图形
plt.figure(figsize=(5, 5))
plt.scatter(X_test, y_test, color='green', label='Test data points', alpha=0.6)
X_test_sorted = X_test[np.argsort(X_test.flatten())]
best_y_pred_test = best_model.predict(best_polynomial_features.transform(X_test_sorted))
plt.plot(X_test_sorted[:, 0], best_y_pred_test, color='purple', linestyle='--', label=f'Prediction on test set (degree={best_degree})')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Prediction on Test Set')
plt.legend()
plt.show()

# 输出训练集和测试集的最佳degree及对应的MSE
print('Best degree:', best_degree)
print('Minimum MSE on training set:', min_mse)
print('MSE on test set:', mse_test)
