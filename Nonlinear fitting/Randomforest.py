import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# 读取训练和测试数据
file_path_train = 'G:\\machine learning\\complex_nonlinear_data.xlsx'
file_path_test = 'G:\\machine learning\\new_complex_nonlinear_data.xlsx'

# 读取训练和测试数据集
df_train = pd.read_excel(file_path_train)
df_test = pd.read_excel(file_path_test)

X_train = df_train['Feature'].values.reshape(-1, 1)
y_train = df_train['Target'].values
X_test = df_test['Feature'].values.reshape(-1, 1)
y_test = df_test['Target'].values

# 初始化随机森林回归模型
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 使用模型进行预测
y_pred_train = model.predict(X_train)
mse_train = mean_squared_error(y_train, y_pred_train)

# 使用模型对测试集进行预测
y_pred_test = model.predict(X_test)
mse_test = mean_squared_error(y_test, y_pred_test)

# 输出训练集和测试集的MSE
print('MSE on train set:', mse_train)
print('MSE on test set:', mse_test)

# 绘制训练集数据点和拟合曲线
plt.scatter(X_train, y_train, color='blue', label='Training data points')
X_train_sorted = sorted(X_train)
y_pred_train_sorted = model.predict(X_train_sorted)
plt.plot(X_train_sorted, y_pred_train_sorted, color='red', label='Random Forest Fit')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Random Forest Fitting on Training Set')
plt.legend()
plt.show()

# 绘制测试集数据点和预测曲线
plt.scatter(X_test, y_test, color='green', label='Test data points')
X_test_sorted = sorted(X_test)
y_pred_test_sorted = model.predict(X_test_sorted)
plt.plot(X_test_sorted, y_pred_test_sorted, color='purple', label='Random Forest Prediction')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Random Forest Prediction on Test Set')
plt.legend()
plt.show()
