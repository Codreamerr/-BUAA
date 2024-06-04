import numpy as np
import matplotlib.pyplot as plt


# 生成数据函数
def generate_heights(N, mu_M, sigma_M, mu_F, sigma_F):
    num_males = int(3 / 5 * N)
    num_females = N - num_males

    male_heights = np.random.normal(mu_M, sigma_M, num_males)
    female_heights = np.random.normal(mu_F, sigma_F, num_females)

    heights = np.concatenate([male_heights, female_heights])
    labels = np.array([1] * num_males + [0] * num_females)

    indices = np.arange(N)
    np.random.shuffle(indices)

    return heights[indices], labels[indices]


# 混合高斯模型的EM算法
def gaussian(x, mu, sigma):
    return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def em_algorithm(heights, max_iter=500, tol=1e-6):
    N = len(heights)

    # 初始化参数
    pi = 0.3  # 初始男生比例
    mu_M = np.mean(heights) + 5
    sigma_M = np.std(heights)
    mu_F = np.mean(heights) - 5
    sigma_F = np.std(heights)

    for _ in range(max_iter):
        # E步：计算后验概率
        gamma_M = pi * gaussian(heights, mu_M, sigma_M)
        gamma_F = (1 - pi) * gaussian(heights, mu_F, sigma_F)
        gamma_sum = gamma_M + gamma_F
        gamma_M /= gamma_sum
        gamma_F /= gamma_sum

        # M步：更新参数
        N_M = np.sum(gamma_M)
        N_F = np.sum(gamma_F)

        mu_M_new = np.sum(gamma_M * heights) / N_M
        sigma_M_new = np.sqrt(np.sum(gamma_M * (heights - mu_M_new) ** 2) / N_M)

        mu_F_new = np.sum(gamma_F * heights) / N_F
        sigma_F_new = np.sqrt(np.sum(gamma_F * (heights - mu_F_new) ** 2) / N_F)

        pi_new = N_M / N

        # 检查收敛性
        if np.abs(mu_M - mu_M_new) < tol and np.abs(sigma_M - sigma_M_new) < tol and \
                np.abs(mu_F - mu_F_new) < tol and np.abs(sigma_F - sigma_F_new) < tol and \
                np.abs(pi - pi_new) < tol:
            break

        mu_M, sigma_M, mu_F, sigma_F, pi = mu_M_new, sigma_M_new, mu_F_new, sigma_F_new, pi_new

    return pi, mu_M, sigma_M, mu_F, sigma_F


# 第一步：生成数据并进行数据可视化
N = 2500
mu_M = 176
sigma_M = 8
mu_F = 164
sigma_F = 6

heights, labels = generate_heights(N, mu_M, sigma_M, mu_F, sigma_F)

plt.figure(figsize=(12, 6))
plt.hist(heights, bins=30, alpha=0.6, color='g', label='Heights Distribution')
plt.title('University Students Heights Distribution')
plt.xlabel('Height (cm)')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# 第二步：执行EM算法并打印估计的参数
pi, mu_M_est, sigma_M_est, mu_F_est, sigma_F_est = em_algorithm(heights, max_iter=200)

print(f"Estimated parameters:")
print(f"pi (male proportion): {pi:.3f}")
print(f"mu_M (male mean height): {mu_M_est:.3f}")
print(f"sigma_M (male height std): {sigma_M_est:.3f}")
print(f"mu_F (female mean height): {mu_F_est:.3f}")
print(f"sigma_F (female height std): {sigma_F_est:.3f}")

# 第三步：对最终结果进行可视化处理
plt.figure(figsize=(12, 6))

# 直方图
count, bins, ignored = plt.hist(heights, bins=30, alpha=0.6, color='g', label='Heights Distribution')

# 绘制估计的高斯分布
x = np.linspace(heights.min(), heights.max(), 1000)
bin_width = bins[1] - bins[0]
p_M = pi * gaussian(x, mu_M_est, sigma_M_est) * N * bin_width
p_F = (1 - pi) * gaussian(x, mu_F_est, sigma_F_est) * N * bin_width
plt.plot(x, p_M, label='Estimated Male Distribution', linestyle='--')
plt.plot(x, p_F, label='Estimated Female Distribution', linestyle='--')

# 绘制真实的高斯分布
p_M_real = (3 / 5) * gaussian(x, mu_M, sigma_M) * N * bin_width
p_F_real = (2 / 5) * gaussian(x, mu_F, sigma_F) * N * bin_width
plt.plot(x, p_M_real, label='True Male Distribution', linestyle=':')
plt.plot(x, p_F_real, label='True Female Distribution', linestyle=':')

# 显示均值
plt.axvline(mu_M_est, color='b', linestyle='-', linewidth=2, label='Estimated Male Mean')
plt.axvline(mu_F_est, color='r', linestyle='-', linewidth=2, label='Estimated Female Mean')
plt.axvline(mu_M, color='b', linestyle='-.', linewidth=2, label='True Male Mean')
plt.axvline(mu_F, color='r', linestyle='-.', linewidth=2, label='True Female Mean')

plt.title('University Students Heights Distribution and Estimated Gaussian Mixtures')
plt.xlabel('Height (cm)')
plt.ylabel('Frequency')
plt.legend()
plt.show()
