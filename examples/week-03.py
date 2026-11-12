import numpy as np
import matplotlib.pyplot as plt
k = np.linspace(-1, 1, 200)
g = 0.2  # 耦合强度，仅用于示意
# 对每个波矢求实对称矩阵的本征值
levels = np.array([np.linalg.eigvalsh([[q, g], [g, -q]]) for q in k])
plt.plot(k, levels[:, 0]); plt.plot(k, levels[:, 1])
plt.xlabel('wave vector k'); plt.ylabel('eigenvalue (toy)'); plt.show()
