// 每个单元独立保存内容；修改此数组即可扩充后续课程。
const lessons = [
{title:'向量、矩阵与物理数据',en:'Vectors as data',theme:'线性组合 · 数据形状 · NumPy',concept:'把一次测量或一条色散曲线写成向量；把多次测量堆叠成矩阵 X。约定行是样本、列是特征，矩阵乘法可以表示对全部样本做同一个线性变换。',formula:'X ∈ ℝⁿˣᵈ  ·  y = Xw',example:'例如，一条光子晶体色散曲线在 d 个波矢处取样，就成为 d 维向量。不同结构参数产生多条曲线；比较前要确认波矢网格与频率单位一致。',demo:'photonic',demoTitle:'双层介质折射率如何改变色散？',code:`import numpy as np
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
plt.show()`,questions:['如果每条曲线的采样网格不同，直接堆叠有什么问题？','改变介电常数会如何影响真实色散？这个示意模型没有包含哪些物理？'],tasks:'读向量与矩阵基础 2h；NumPy 数组与绘图 2h；光子晶体数据格式检查 2h；复盘 1h；计算物理：熟悉旧作业 1–2h',reading:'Strang：向量、线性组合和矩阵乘法；计算物理作业：数组、绘图与结果标注。'},
{title:'投影、协方差与 PCA',en:'Projection & PCA',theme:'内积 · 方差 · SVD',concept:'PCA 寻找中心化数据中方差最大的正交方向。把数据投影到少数方向，可以保留主要变化，但最大方差不必然对应最重要的物理机制。',formula:'Xc = X − mean(X)  ·  Xc = UΣVᵀ',example:'对多条色散曲线按统一波矢采样，PCA 的每个分量是一种随 k 变化的曲线形状；得分表示每个结构沿该变化方向的投影。',demo:'pca',demoTitle:'拖动相关性，观察第一主方向',code:`import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(4)
t = rng.normal(size=150)
X = np.column_stack([t, 0.8*t + 0.4*rng.normal(size=150)])
Xc = X - X.mean(axis=0)  # 先中心化
U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
score = Xc @ Vt.T         # 投影到主轴
print('解释方差比：', s**2 / np.sum(s**2))
plt.scatter(score[:, 0], score[:, 1], s=10)
plt.xlabel('PC1'); plt.ylabel('PC2'); plt.axis('equal'); plt.show()`,questions:['为什么中心化会改变主轴的含义？','如果噪声幅度最大，PC1 还能代表物理信号吗？'],tasks:'Strang 内积与投影 2h；协方差与 SVD 2h；曲线 PCA 练习 2h；复盘 1h；计算物理：绘图及单位 1–2h',reading:'Strang：正交与最小二乘、特征值及 SVD；《机器学习》降维相关章节。'},
{title:'特征值与耦合模',en:'Eigenvectors in physics',theme:'本征问题 · 耦合 · 能带',concept:'矩阵的本征向量在变换下方向不变。对实对称矩阵，本征值可表示耦合模式的特征频率；模型的形式和参数必须来自具体物理系统。',formula:'H(k)v = λ(k)v',example:'用两条相交的示意色散与耦合项构成 2×2 矩阵；耦合使交叉附近产生反交叉。这是教学玩具模型，不是实际光子晶体求解。',demo:'bands',demoTitle:'改变耦合强度，观察反交叉',code:`import numpy as np
import matplotlib.pyplot as plt
k = np.linspace(-1, 1, 200)
g = 0.2  # 耦合强度，仅用于示意
# 对每个波矢求实对称矩阵的本征值
levels = np.array([np.linalg.eigvalsh([[q, g], [g, -q]]) for q in k])
plt.plot(k, levels[:, 0]); plt.plot(k, levels[:, 1])
plt.xlabel('wave vector k'); plt.ylabel('eigenvalue (toy)'); plt.show()`,questions:['耦合趋近于零时交叉附近发生什么？','模型中的矩阵与真实 Maxwell 本征问题有哪些差别？'],tasks:'特征值与对称矩阵 2h；二能级耦合模型 2h；绘制参数扫描 2h；复盘 1h；计算物理：数值误差 1–2h',reading:'Strang：特征值、特征向量和对称矩阵；Photonic Crystals + COMSOL 的实际模型作为延伸。'},
{title:'聚类：从颜色到结构',en:'Clustering with k-means',theme:'距离 · 特征 · 无监督学习',concept:'k-means 交替执行最近中心分配和中心更新，最小化簇内平方距离。结果依赖特征尺度、k、初始化及数据分布。',formula:'min Σᵢ ||xᵢ − μ₍cᵢ₎||²',example:'校园遥感图片的每个像素可写成 RGB 向量，也可变换到 HSV；色相是周期变量，直接使用普通欧氏距离会在色相边界产生问题。',demo:'kmeans',demoTitle:'调整 k，观察二维示意像素聚类',code:`import numpy as np
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
ax[0].set_title('synthetic RGB'); ax[1].set_title('HSV clustering'); plt.show()`,questions:['为什么颜色相近的屋顶和道路可能被分到同一类？','RGB 和 HSV 的尺度、色相周期如何影响距离？'],tasks:'《机器学习》聚类 2h；图像数组与颜色空间 2h；校园影像聚类 2h；复盘 1h；计算物理：向量化 1–2h',reading:'《机器学习》聚类章节；参考原讲义 tokenization / coarse-graining。'},
{title:'Ising：从构型到相空间',en:'Ising configurations',theme:'自旋 · 磁化强度 · PCA',concept:'把二维自旋构型展平成向量，多个构型形成样本矩阵。PCA 可以揭示主要变化，仍须检查温度采样、对称性、系统尺寸和磁化强度。',formula:'m = (1/N) Σᵢ sᵢ  ·  sᵢ ∈ {−1, +1}',example:'低温铁磁态存在正、负两种对称方向，高温构型更无序。简单随机生成的图像只是示意；定量研究需要正确采样的 Ising 构型。',demo:'ising',demoTitle:'观察合成自旋构型与磁化强度',code:`import numpy as np
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
# 增加独立链和采样量；检查自相关、平衡和有限尺寸效应。`,questions:['为什么 m 和 −m 可能代表同一个对称破缺相？','PCA 散点分离能否单独证明临界温度？'],tasks:'Ising 和序参量 2h；构型矩阵与 PCA 2h；阅读 Hu et al. 的问题设置 2h；复盘 1h；计算物理：随机数 1–2h',reading:'Hu, Singh & Scalettar (2017), Discovering phases, phase transitions, and crossovers through unsupervised machine learning。'},
{title:'Autoencoder 与潜在表征',en:'Latent representations',theme:'编码器 · 重建误差 · 比较',concept:'Autoencoder 将输入压缩到 latent space，再尝试重建；重建成功说明模型保留了某些信息，不自动证明潜变量是物理序参量。',formula:'z = encoder(x)  ·  x̂ = decoder(z)',example:'将 Ising 构型作为输入，比较 PCA 与非线性自编码器的低维表示；按独立模拟或时间段划分训练与测试，避免相关样本泄漏。',demo:'ising',demoTitle:'不同构型的压缩：哪些细节可以丢失？',code:`import numpy as np
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
# 与相同数据上的 PCA 基线比较；构型独立性需要另行验证。`,questions:['重建误差更小是否意味着相变识别更准确？','如何验证 latent variable 与磁化强度的关系？'],tasks:'神经网络与损失函数 2h；实现小型 autoencoder 2h；与 PCA 比较 2h；复盘 1h；计算物理：测试与复现 1–2h',reading:'《机器学习》神经网络；过冷水论文是 VAE + 矢量量化的延伸阅读，并非本周复现。'},
{title:'高斯过程与参数反演',en:'Gaussian processes',theme:'代理模型 · 不确定度 · 反演',concept:'Gaussian process (GP) 为未知函数给出联合概率模型。代理模型预测的不确定度与测量噪声应区分；参数反演还需似然、先验和可辨识性分析。',formula:'p(θ | y) ∝ p(y | θ) p(θ)',example:'可先用衍射光栅前向模型计算不同参数的强度，再拟合 GP 代理，结合观测值估计参数。本页交互只展示简化似然而非完整 GP。',demo:'inverse',demoTitle:'观测噪声如何改变参数反演？',code:`import numpy as np
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
# 使用真实前向模型前，先用留出数据检验 GP 预测误差。`,questions:['如果多个参数产生几乎相同强度，后验会怎样？','GP 预测不确定度为什么不能直接等同于实验误差？'],tasks:'概率、似然与贝叶斯 2h；理解 GP 回归 2h；光栅前向模型与反演草图 2h；复盘 1h；计算物理：误差传播 1–2h',reading:'《机器学习》回归与概率模型；先从简化光栅正演入手，GP 与真实数据反演待实践。'},
{title:'研究闭环与 FYP 雏形',en:'From notebook to research',theme:'验证 · 可复现 · 科学问题',concept:'把一个研究问题写成可验证闭环：问题与基线、数据与单位、方法、误差与对照、解释和下一步。性能提升需要与物理基线及数据划分一起报告。',formula:'Question → Data → Baseline → Test → Interpretation',example:'对一维光子晶体：生成或导入不同参数的色散曲线，检查数值收敛，识别频带范围和可能带隙，再把曲线特征输入 PCA。PCA 本身不等于带隙检测。',demo:'photonic',demoTitle:'观察一维周期结构的允许带',code:`import numpy as np
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
# 加密频率和 k 网格并单独检查端点切触根后，才可作定量结论。`,questions:['有限波矢采样可能漏掉什么？','如何区分项目假设、已经计算的结果和下一步计划？'],tasks:'选定 FYP 问题 1h；文献和数据清单 2h；基线和复现 2h；写一页研究备忘 2h；计算物理：误差检查 1–2h',reading:'Photonic Crystals + COMSOL 研究记录；Hu et al. 2017；Cartography 案例作为延伸。'}
];
