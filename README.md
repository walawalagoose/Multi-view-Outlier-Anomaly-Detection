# Multi-View-Outlier-Anomaly-Detection

A collection of baseline models and benchmark datasets for Multi-View Outlier/Anomaly Detection tasks

# Introduction

This is a repository focusing on Anomaly Detection/Outlier Detection task on Multi-View data, based on datasets/baseline models collected that summarized from 30+ related papers ***(still under update)***. For copyright reasons, this repo only provides links rather than original resources.

This repository is continuously being improved. We will provide a more comprehensive summary, more detailed descriptions, more systematic classification, and a more readable layout design in the future. 

Any positive contribution to issues/pull requests is welcomed!

**Keywords: Multi-View Learning; Anomaly Detection; Outlier Detection**

------

这是一个专注于多视图数据上anomaly detection/outlier detection任务的repo，根据从20+篇相关论文（更新中）中收集和总结的数据集/基线模型总结得出。基于版权原因，本repo不提供原始资源。

该repo正在不断完善中，后续将提供更为全面的总结、更加细致的描述、更加系统的分类与更加已读的排版设计。

欢迎大家贡献issues/pull requests！

**关键词：多视图学习 异常检测 离群值检测**

# Baselines

## Mainstream Methods

1. **HOAD**
   - From: "A Spectral Framework for Detecting Inconsistency across Multi-Source Object Relationships", ICDM 2011 (CCF-B)
   - Description: 
     - 核心思想：在每个视图数据上建立图并构建邻接矩阵，并为所有视图构建了一个带有约束的组合谱聚类（连接所有图）
     - 异常度量：样本在所有视图谱嵌入的两两余弦距离之和
     - 限制：依赖成对计算；基于聚类；处理一种异常值
   - Keywords/ Contributions: 谱聚类；图表示
   - Link: [[Paper]](https://ieeexplore.ieee.org/abstract/document/6137313), [Code] (Not-Found yet)
2. **AP**
   - From: "Clustering-Based Anomaly Detection in Multi-View Data", CIKM 2013 (CCF-B)
   - Description: 
     - 核心思想：在每个视图数据上执行亲和传播聚类，基于亲和传播矩阵构造亲和性向量（数据点和其他点聚类中心与该点亲和度之和，which可由距离或核函数表示）
     - 异常度量：样本在所有视图亲和性向量的两两相似性（距离、相似性 or HSIC)之和
     - 限制/优点：依赖成对计算；基于聚类；处理一种异常值
   - Keywords/ Contributions: 亲和传播聚类
   - Link: [[Paper]](https://dl.acm.org/doi/abs/10.1145/2505515.2507840), [Code] (Not-Found yet)
3. **DMOD**
   - From: "Dual-Regularized Multi-View Outlier Detection", IJCAI 2015 (CCF-A)
   - Description: 
     - 核心思想：进行K-Means聚类后将数据分解为聚类中心矩阵、聚类指示矩阵和残差矩阵，并构造对每个视图残差矩阵稀疏性和聚类结构的成对约束
     - 异常度量：样本在所有视图内聚类指示矩阵的成对内积（类异常）与残差矩阵内积（属性异常）之差
     - 限制/优点：依赖成对计算；基于聚类；处理两种异常值
   - Keywords/ Contributions: K-Means聚类；稀疏表示；凸优化
   - Link: [[Paper]](https://www.ijcai.org/Abstract/15/572), [[Code]](https://github.com/nilde/GABD/blob/aaf0101361dece3f720e3f4f3f0e0c9caa4246ad/mongoDBScripts/codiOriol/src/OutlierDetector/DMOD.py)
4. **CRMOD**
   - From: "Consensus Regularized Multi-View Outlier Detection", TIP 2018 (CCF-A, JCR Q1)
   - Description: DOMD的优化版
     - 核心思想：进行K-Means聚类后将数据分解为共识聚类中心矩阵、聚类指示矩阵和残差矩阵，并构造对每个视图残差矩阵稀疏性和聚类结构的共识约束
     - 异常度量：同DMOD，仍为pair-wise
     - 限制/优点：依赖成对计算；基于聚类；处理两种异常值
   - Keywords/ Contributions: K-Means聚类；稀疏表示；凸优化
   - Link: [[Paper]](https://ieeexplore.ieee.org/abstract/document/8047342), [[Code]](https://github.com/nilde/GABD/blob/aaf0101361dece3f720e3f4f3f0e0c9caa4246ad/mongoDBScripts/codiOriol/src/OutlierDetector/CMOD.py)
5. **MLRA**
   - From: "Multi-View Low-Rank Analysis with Applications to Outlier Detection", TKDD 2018 (CCF-B, JCR Q1)
   - Description: SIAM 2015版本的优化
     - 核心思想：将每个视图数据分解为低秩自表示项与稀疏残差矩阵之和，并构造系数矩阵低秩性、残差矩阵稀疏性与系数矩阵间的对齐约束
     - 异常度量：样本在所有视图内低秩系数矩阵的成对内积（类异常）与残差矩阵内积（属性异常）之差
     - 限制/优点：依赖成对计算；基于子空间（自表示）；处理两种异常值 (first ever)
   - Keywords/ Contributions: 子空间学习；稀疏表示；低秩表示；凸优化
   - Link: [[Paper]](https://dl.acm.org/doi/abs/10.1145/3168363), [[Code]](https://sheng-li.org/Codes/SDM15_MLRA_Code.zip)
6. **LDSR**
   - From: "Latent Discriminant Subspace Representations for Multi-View Outlier Detection", AAAI 2018 (CCF-A)
   - Description: MLRA的优化版
     - 核心思想：将每个视图数据分解为低秩自表示项（包括共识表示和特定表示）与稀疏残差矩阵之和，并构造共识系数矩阵低秩性、特定系数矩阵和残差矩阵稀疏性的约束
     - 异常度量：样本在所有视图内特定系数矩阵（类异常）和残差矩阵（属性异常）误差之和
     - 限制/优点：不依赖成对计算 (first ever)；基于子空间（自表示）；处理三种异常值 (first ever)
   - Keywords/ Contributions: 子空间学习；自表示学习；稀疏表示；低秩表示；凸优化
   - Link: [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/11826), [[Code]](https://github.com/kailigo/mvod)
7. **MODDIS**
   - From: "Multi-view Outlier Detection in Deep Intact Space", ICDM 2019 (CCF-B)
   - Description:
     - 核心思想：建立多个view-specific自编码器与一个view-consensus自编码器，前者重构残差项最小化残差，后者重构共识项并最小化不同样本输入与输出的差异
     - 异常度量：样本在所有视图内的合并latent嵌入的knn距离（类异常）和与平均嵌入之间的误差（属性异常）之和
     - 限制/优点：深度方法 (first ever)；不依赖成对计算；基于邻域；处理三种异常值
   - Keywords/ Contributions: 深度学习；knn；intact space learning；表示学习；自编码器
   - Link: [[Paper]](https://ieeexplore.ieee.org/abstract/document/8970937), [[Code]](https://github.com/sigerma/ICDM-2019-MODDIS)
8. **MUVAD**
   - From: "Multi-View Anomaly Detection: Neighborhood in Locality Matters", AAAI 2019 (CCF-A)
   - Description: 
     - 核心思想：
     - 异常度量：
     - 限制/优点：依赖成对计算；基于邻域；处理三种异常值
   - Keywords/ Contributions: 
   - Link: [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/4418), [Code] (Not-Found yet)
9. **NCMOD**
   - From: "Neighborhood Consensus Networks for Unsupervised Multi-view Outlier Detection", AAAI 2021 (CCF-A)
   - Description: 
     - 核心思想：使用自编码器重构样本；并在每个视图数据上建立图、构建knn邻接矩阵与共识项，施加对共识邻接矩阵的优化
     - 异常度量：样本重构误差（属性异常）与latent embedding经共识邻接矩阵优化后的knn距离（类异常）之和
     - 限制/优点：半深度方法；不依赖成对计算；基于邻域；处理三种异常值
   - Keywords/ Contributions: 深度学习；knn ；intact space learning；表示学习；图表示；凸优化
   - Link: [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/16873), [[Code]](https://github.com/auguscl/NCMOD)
10. **SRLSP**
    - From: "A Self-Representation Method with Local Similarity Preserving for Fast Multi-View Outlier Detection", TKDD 2023 (CCF-B, JCR Q1)
    - Description: 
      - 核心思想：
      - 异常度量：
      - 限制/优点：
    - Keywords/ Contributions: 
    - Link: [[Paper]](https://dl.acm.org/doi/abs/10.1145/3532191), [[Code]](https://github.com/wy54224/SRLSP)
11. **IAMOD**
    - From: "Information-aware Multi-view Outlier Detection", TKDD 2024 (CCF-B, JCR Q1)
    - Description: 
      - 核心思想：
      - 异常度量：
      - 限制/优点：
    - Keywords/ Contributions: 深度学习；信息论
    - Link: [[Paper]](https://dl.acm.org/doi/abs/10.1145/3638354), [[Code]](https://github.com/MaybeLL/IAMOD)
12. **MODGD**
    - From: "Multi-view Outlier Detection via Graphs Denoising", Information Fusion 2024 (JCR Q1)
    - Description: 
      - 核心思想：
      - 异常度量：
      - 限制/优点：
    - Keywords/ Contributions: 
    - Link: [[Paper]](https://www.sciencedirect.com/science/article/abs/pii/S1566253523003287), [[Code]]( http://Doctor-Nobody.github.io/codes/MODGD.zip)
13. TODO...



## Partial Multi-View OD/AD

1. **CL**

   - From: "Partial Multi-View Outlier Detection Based on Collective Learning", AAAI 2018 (CCF-A)
   - Description: 

   - Keywords/ Contributions: 
   - Link: [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/11278), [[Code]](https://github.com/eeGuoJun/AAAI2018_CL)

2. **RCPMOD**

   - From: "Regularized Contrastive Partial Multi-view Outlier Detection", ACMMM 2024 (CCF-A)

   - Description: 
   - Keywords/ Contributions: 
   - Link: [[Paper]](https://arxiv.org/abs/2408.07819), [[Code]](https://github.com/180400518/RCPMOD)

3. TODO...

## New Methods/Pure AD/Under Exploration...

1. CC

   - From: "Using Consensus Clustering for Multi-view Anomaly Detection", S&P Workshops 2012 (CCF-A of main conference)

   - Description: 
   - Keywords/ Contributions: 
   - Link: [[Paper]](https://ieeexplore.ieee.org/abstract/document/6227694), [Code] (Not-Found yet)

2. PLVM

   - From: "Multi-view Anomaly Detection via Robust Probabilistic Latent Variable Models", NeurIPS 2016 (CCF-A)

   - Description: 
   - Keywords/ Contributions: 
   - Link: [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2016/hash/0f96613235062963ccde717b18f97592-Abstract.html), [Code] (Not-Found yet)

3. MGAD

   - From: "Multi-View Group Anomaly Detection", CIKM 2018 (CCF-B)

   - Description: 
   - Keywords/ Contributions: 
   - Link: [[Paper]](https://dl.acm.org/doi/abs/10.1145/3269206.3271770), [Code] (Not-Found yet)

4. IMVSAD

   - From: "Inductive Multi-view Semi-Supervised Anomaly Detection via Probabilistic Modeling", ICBK 2019

   - Description: 
   - Keywords/ Contributions: 
   - Link: [[Paper]](https://link.springer.com/chapter/10.1007/978-3-031-30678-5_9), [Code] (Not-Found yet)

5. Bayesian-MVAD

   - From: "Towards a Hierarchical Bayesian Model of Multi-View Anomaly Detection", IJCAI 2020 (CCF-A)

   - Description: 
   - Keywords/ Contributions: 
   - Link: [[Paper]](https://par.nsf.gov/servlets/purl/10171437), [[Code]](https://github.com/zwang-datascience/MVAD_Bayesian/)

6. CGAEs

   - From: "Cross-aligned and Gumbel-refactored Autoencoders for Multi-view Anomaly Detection", ICTAI 2021 (CCF-C)

   - Description: 
   - Keywords/ Contributions: 
   - Link: [[Paper]](https://ieeexplore.ieee.org/abstract/document/9643362), [Code] (Not-Found yet)

7. PLSVD (Unofficial name)

   - From: "Learning Probabilistic Latent Structure for Outlier Detection from Multi-view Data", PAKDD 2021 (CCF-C)

   - Description: 
   - Keywords/ Contributions: 
   - Link: [[Paper]](https://ieeexplore.ieee.org/abstract/document/8944679/), [Code] (Not-Found yet)

8. Deep ADAN (Unofficial name)

   - From: "A Deep Multi-View Framework for Anomaly Detection on Attributed Networks", TKDE 2022 (CCF-A, JCR Q1)

   - Description: 
   - Keywords/ Contributions: 
   - Link: [[Paper]](https://ieeexplore.ieee.org/abstract/document/9162509), [Code] (Not-Found yet)

9. Fast ODDE (Unofficial name)

   - From: "Fast Multi-View Outlier Detection via Deep Encoder", TBD 2022 (CCF-C, JCR Q1)

   - Description: 
   - Keywords/ Contributions: 
   - Link: [[Paper]](https://ieeexplore.ieee.org/abstract/document/9122431), [Code] (Not-Found yet)

10. ECMOD

    - From: "Learning Enhanced Representations via Contrasting for Multi-view Outlier Detection", DASFAA 2023 (CCF-B)

    - Description: 
    - Keywords/ Contributions: 
    - Link: [[Paper]](https://link.springer.com/chapter/10.1007/978-3-031-30678-5_9), [[Code]](https://github.com/scu-kdde/OAM-ECMOD-2023)

11. SeeM

    - From: "SeeM: A Shared Latent Variable Model for Unsupervised Multi-view Anomaly Detection", PAKDD 2024  (CCF-C)

    - Description: 
    - Keywords/ Contributions: 
    - Link: [[Paper]](https://link.springer.com/chapter/10.1007/978-981-97-2242-6_7), [[Code]](https://github.com/thanhphuong163/SeeM)

12. MGAD-SGCC (Unofficial name)

    - From: "Towards Multi-view Graph Anomaly Detection with Similarity-Guided Contrastive Clustering", *arxiv* 2024

    - Description: 
    - Keywords/ Contributions: 
    - Link: [[Paper]](https://arxiv.org/abs/2409.09770), [Code] (Not-Found yet)

13. Multi-view AD exploration

    - From: "Multiview Deep Anomaly Detection: A Systematic Exploration", TNNLS 2024 (CCF-B, JCR Q1)

    - Description: 
    - Keywords/ Contributions: 
    - Link: [[Paper]](https://ieeexplore.ieee.org/abstract/document/9810850), [Code] (Not-Found yet)

14. TODO...

## Non-typical Multi-View Tasks/ Non-AD

TO BE DONE

# Datasets

