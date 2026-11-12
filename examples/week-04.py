import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv
from sklearn.cluster import KMeans
rng = np.random.default_rng(2)
# 构造 RGB 示例图；替换 rgb 为自己的校园图像（范围 0 到 1）
rgb = np.zeros((50, 90, 3))
rgb[:, :30] = [0.2, 0.6, 0.25]   # 植被示意
rgb[:, 30:60] = [0.6, 0.6, 0.6] # 建筑 / 道路示意
rgb[:, 60:] = [0.25, 0.4, 0.7]  # 水体 / 阴影示意
rgb = np.clip(rgb + rng.normal(0, .04, rgb.shape), 0, 1)
hsv = rgb_to_hsv(rgb)
# 色相编码为圆上的坐标，避免 0 和 1 实际相近却被算远
h, sat, value = np.moveaxis(hsv, -1, 0)
features = np.stack([sat*np.cos(2*np.pi*h), sat*np.sin(2*np.pi*h), value], axis=-1)
labels = KMeans(n_clusters=3, random_state=0, n_init=10).fit_predict(features.reshape(-1, 3))
fig, ax = plt.subplots(1, 2); ax[0].imshow(rgb); ax[1].imshow(labels.reshape(rgb.shape[:2]), cmap='viridis')
ax[0].set_title('synthetic RGB'); ax[1].set_title('HSV clustering'); plt.show()
