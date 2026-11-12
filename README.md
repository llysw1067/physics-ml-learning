# 物理 × 机器学习：八周学习工作台

面向初学者的线性代数、Python、机器学习与计算物理学习记录。项目正在建设中，网页中的进度由学习者自行记录。

## 开始学习

1. 用浏览器打开 `website/index.html`，选择一个 60 分钟学习单元。
2. 每周还可安排约 6 小时科研学习和 1–2 小时计算物理练习。
3. `examples/` 提供网页中的独立 Python 示例；请先检查注释和依赖。
4. `notes/` 可记录练习、卡点和复盘；请勿提交不想公开的记录。

## Python 环境

目前使用已有 Conda 环境 `calculatephys`（Python 3.12）。已检查 NumPy、Matplotlib、SciPy 可用；第 4–7 周示例需要 scikit-learn，当前未安装。运行示例：

```powershell
conda activate calculatephys
python examples/week-01.py
```

后续会补充经过验证的环境配置。

## 内容与边界

网页受 Token Cartography 交互讲义视觉组织启发，正文与演示为本项目教学内容。原讲义、论文 PDF、原始数据和图片未包含在仓库中。示例使用简化模型或合成数据，不能直接视为研究结论。网页记录保存在当前浏览器，建议定期导出备份。

路线：向量与矩阵 → PCA → 特征值与耦合模 → 聚类 → Ising → Autoencoder → GP 与反演 → 科研问题和复现。学习状态由本人维护。

本项目尚在整理，许可协议待选择；请勿加入未获再分发许可的教材、原讲义、论文全文或个人数据。
