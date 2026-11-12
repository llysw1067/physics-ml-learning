import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(4)
t = rng.normal(size=150)
X = np.column_stack([t, 0.8*t + 0.4*rng.normal(size=150)])
Xc = X - X.mean(axis=0)  # 先中心化
U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
score = Xc @ Vt.T         # 投影到主轴
print('解释方差比：', s**2 / np.sum(s**2))
plt.scatter(score[:, 0], score[:, 1], s=10)
plt.xlabel('PC1'); plt.ylabel('PC2'); plt.axis('equal'); plt.show()
