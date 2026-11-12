import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
rng = np.random.default_rng(3)
# 合成二值构型；真实研究请替换成独立模拟采样的 Ising 配置
X = np.vstack([rng.binomial(1, p, size=(80, 16)) for p in (.15, .5, .85)])
rng.shuffle(X); train, test = X[:180], X[180:]
# 16 → 2 → 16 的自编码器：输出目标与输入相同
model = MLPRegressor(hidden_layer_sizes=(2,), activation='tanh',
                     max_iter=1000, random_state=1).fit(train, train)
z = np.tanh(test @ model.coefs_[0] + model.intercepts_[0])
recon = model.predict(test)
print('测试集重建 MSE:', np.mean((test-recon)**2))
plt.scatter(z[:,0], z[:,1], c=test.mean(axis=1), cmap='coolwarm')
plt.xlabel('latent 1'); plt.ylabel('latent 2'); plt.colorbar(label='mean spin proxy'); plt.show()
# 与相同数据上的 PCA 基线比较；构型独立性需要另行验证。
