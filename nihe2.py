import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


# 读取
file_path_train = 'G:/machine learning/complex_nonlinear_data.xlsx'
file_path_test = 'G:/machine learning/new_complex_nonlinear_data.xlsx'
df_train = pd.read_excel(file_path_train)
df_test = pd.read_excel(file_path_test)

X_train = df_train['Feature'].values.reshape(-1, 1)
y_train = df_train['Target'].values
X_test = df_test['Feature'].values.reshape(-1, 1)
y_test = df_test['Target'].values

# 定义超参数
param_grid = {
    'gamma': [3, 5, 9],
    'C': [0.7, 0.9, 2],
    'epsilon': [0.02, 0.05, 0.09]
}

# 最佳超参数
grid_search = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=10)
grid_search.fit(X_train, y_train)

best_svr = grid_search.best_estimator_
best_params = grid_search.best_params_

y_pred_train = best_svr.predict(X_train)
y_pred_test = best_svr.predict(X_test)

# 输出MSE
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)

print('Best Parameters:', best_params)
print('MSE on train set:', mse_train)
print('MSE on test set:', mse_test)

# 绘制拟合曲线
plt.scatter(X_train, y_train, color='blue', label='Training data points')
plt.plot(X_train, y_pred_train, color='red', label='SVR Fit on train set')
plt.scatter(X_test, y_test, color='green', label='Test data points')
plt.plot(X_test, y_pred_test, color='purple', label='SVR Fit on test set')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('SVR Fitting')
plt.legend()
plt.show()
