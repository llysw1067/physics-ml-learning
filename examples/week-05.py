import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
rng = np.random.default_rng(7)
L = 12
# Metropolis 更新：演示用小样本，不能直接作临界指数估计
def sample(T):
    s = rng.choice([-1, 1], size=(L, L))
    samples = []
    for sweep in range(900):
        for _ in range(L*L):
            i, j = rng.integers(0, L, size=2)
            neighbors = s[(i+1)%L,j]+s[(i-1)%L,j]+s[i,(j+1)%L]+s[i,(j-1)%L]
            dE = 2*s[i,j]*neighbors
            if dE <= 0 or rng.random() < np.exp(-dE/T): s[i,j] *= -1
        if sweep >= 400 and sweep % 25 == 0: samples.append(s.ravel().copy())
    return samples
X = np.array(sample(1.6) + sample(3.5))
Z = PCA(n_components=2).fit_transform(X)
plt.scatter(Z[:,0], Z[:,1], c=[0]*20+[1]*20, cmap='coolwarm')
plt.xlabel('PC1'); plt.ylabel('PC2'); plt.show()
# 增加独立链和采样量；检查自相关、平衡和有限尺寸效应。
