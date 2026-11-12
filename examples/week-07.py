import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
# 一个玩具正演函数；在真实项目中换成光栅仿真结果
forward = lambda theta: np.sin(theta)
train_x = np.linspace(0, 1.5, 6).reshape(-1, 1)
gp = GaussianProcessRegressor(kernel=ConstantKernel(1)*RBF(.5), alpha=1e-8).fit(train_x, forward(train_x).ravel())
theta = np.linspace(0, 1.5, 300).reshape(-1, 1)
mu, std = gp.predict(theta, return_std=True)
y_obs, sigma_obs = 0.65, 0.08
# 假设代理误差与测量误差独立高斯；均匀参数先验
variance = sigma_obs**2 + std**2
posterior = np.exp(-0.5*(mu-y_obs)**2/variance)/np.sqrt(variance)
posterior /= np.trapz(posterior, theta.ravel())
plt.plot(theta, posterior); plt.xlabel('theta'); plt.ylabel('posterior density'); plt.show()
# 使用真实前向模型前，先用留出数据检验 GP 预测误差。
