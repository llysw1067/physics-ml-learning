import numpy as np
import matplotlib.pyplot as plt
# 一维双层介质正入射，非磁性；u = omega*a/(2*pi*c)
def trace(u, n1, n2, f=.5):
    d1, d2 = 2*np.pi*u*n1*f, 2*np.pi*u*n2*(1-f)
    return np.cos(d1)*np.cos(d2) - .5*(n1/n2+n2/n1)*np.sin(d1)*np.sin(d2)
u = np.linspace(.000001, 1.2, 20000); ks = np.linspace(.02, np.pi-.02, 31)  # 避开端点切触根
curves = []
for n2 in np.linspace(1.5, 2.5, 15):
    F = trace(u, 1.0, n2); row = []
    for k in ks:
        y = F-np.cos(k); idx = np.where(y[:-1]*y[1:] < 0)[0]
        roots = u[idx] - y[idx]*(u[idx+1]-u[idx])/(y[idx+1]-y[idx])
        row.append(roots[:2])
    bands = np.array(row).T  # 2 条带 × k 点
    curves.append(bands.ravel())
X = np.array(curves); Xc = X-X.mean(axis=0)
_, s, vt = np.linalg.svd(Xc, full_matrices=False)
scores = Xc @ vt[:2].T
print('最后一组参数的采样带隙候选:', bands[0].max(), bands[1].min())
plt.scatter(scores[:,0], scores[:,1], c=np.linspace(1.5,2.5,15))
plt.xlabel('PC1'); plt.ylabel('PC2'); plt.colorbar(label='n2'); plt.show()
# 加密频率和 k 网格并单独检查端点切触根后，才可作定量结论。
