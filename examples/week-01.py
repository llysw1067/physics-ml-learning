import numpy as np
import matplotlib.pyplot as plt

# 每一行是一条合成色散曲线；每一列是同一个波矢点
k = np.linspace(0, np.pi, 100)
contrast = np.array([0.1, 0.2, 0.3])
X = np.array([1 + 0.4*np.cos(k) + c for c in contrast])
print('样本数、特征数：', X.shape)
for curve in X:
    plt.plot(k, curve)
plt.xlabel('wave vector k')
plt.ylabel('normalized frequency (toy)')
plt.show()
