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
     - 核心思想：进行K-Means聚类后将数据分解为聚类中心矩阵、聚类指示矩阵和残差矩阵的线性组合，并构造对每个视图残差矩阵稀疏性和聚类结构的成对约束
     - 异常度量：样本在所有视图内聚类指示矩阵的成对内积（类异常）与残差矩阵内积（属性异常）之差
     - 限制/优点：依赖成对计算；基于聚类；处理两种异常值
   - Keywords/ Contributions: K-Means聚类；稀疏表示；凸优化
   - Link: [[Paper]](https://www.ijcai.org/Abstract/15/572), [[Code]](https://github.com/nilde/GABD/blob/aaf0101361dece3f720e3f4f3f0e0c9caa4246ad/mongoDBScripts/codiOriol/src/OutlierDetector/DMOD.py)
4. **CRMOD**
   - From: "Consensus Regularized Multi-View Outlier Detection", TIP 2018 (CCF-A, JCR Q1)
   - Description: DOMD的优化版
     - 核心思想：进行K-Means聚类后将数据分解为共识聚类中心矩阵、聚类指示矩阵和残差矩阵的线性组合，并构造对每个视图残差矩阵稀疏性和聚类结构的共识约束
     - 异常度量：同DMOD，仍为pair-wise
     - 限制/优点：依赖成对计算；基于聚类；处理两种异常值
   - Keywords/ Contributions: K-Means聚类；稀疏表示；凸优化
   - Link: [[Paper]](https://ieeexplore.ieee.org/abstract/document/8047342), [[Code]](https://github.com/nilde/GABD/blob/aaf0101361dece3f720e3f4f3f0e0c9caa4246ad/mongoDBScripts/codiOriol/src/OutlierDetector/CMOD.py)
5. **MLRA**
   - From: "Multi-View Low-Rank Analysis with Applications to Outlier Detection", TKDD 2018 (CCF-B, JCR Q1)
   - Description: SIAM 2015版本的优化
     - 核心思想：将数据分解为低秩自表示项与稀疏残差项之和，并构造系数矩阵低秩性、残差矩阵稀疏性与系数矩阵间的对齐约束
     - 异常度量：样本在所有视图内低秩系数矩阵的成对内积（类异常）与残差矩阵内积（属性异常）之差
     - 限制/优点：依赖成对计算；基于子空间（自表示）；处理两种异常值 (first ever)
   - Keywords/ Contributions: 子空间学习；稀疏表示；低秩表示；凸优化
   - Link: [[Paper]](https://dl.acm.org/doi/abs/10.1145/3168363), [[Code]](https://sheng-li.org/Codes/SDM15_MLRA_Code.zip)
6. **LDSR**
   - From: "Latent Discriminant Subspace Representations for Multi-View Outlier Detection", AAAI 2018 (CCF-A)
   - Description: MLRA的优化版
     - 核心思想：将数据分解为低秩自表示项（包括共识表示和特定表示）与稀疏残差项之和，并构造共识系数矩阵低秩性、特定系数矩阵和残差矩阵稀疏性的约束
     - 异常度量：样本在所有视图内特定系数矩阵（类异常）和残差矩阵（属性异常）$\mathcal{l}_2$-norm之和
     - 限制/优点：不依赖成对计算 (first ever)；基于子空间（自表示）；处理三种异常值 (first ever)
   - Keywords/ Contributions: 子空间学习；自表示学习；稀疏表示；低秩表示；凸优化
   - Link: [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/11826), [[Code]](https://github.com/kailigo/mvod)
7. **MODDIS**
   - From: "Multi-view Outlier Detection in Deep Intact Space", ICDM 2019 (CCF-B)
   - Description:
     - 核心思想：建立两类编码器执行重构：多个view-specific Enc与一个view-consensus Enc。前者重构残差项并最小化残差，后者重构共识项并最小化输入与输入表示空间的密度差异
     - 异常度量：样本在所有视图内的合并latent嵌入的knn距离（类异常）和与平均嵌入之间的差（属性异常）之和
     - 限制/优点：深度方法 (first ever)；不依赖成对计算；基于邻域；处理三种异常值
   - Keywords/ Contributions: 深度学习；knn；intact space learning；表示学习；自编码器
   - Link: [[Paper]](https://ieeexplore.ieee.org/abstract/document/8970937), [[Code]](https://github.com/sigerma/ICDM-2019-MODDIS)
8. **MUVAD**
   - From: "Multi-View Anomaly Detection: Neighborhood in Locality Matters", AAAI 2019 (CCF-A)
   - Description: 
     - 核心思想：在每个视图上构建图邻接矩阵，并以近似正常样本和邻接样本集为目标进行优化
     - 异常度量：样本在所有视图内邻接矩阵的knn正常邻域相似度的和
     - 限制/优点：不依赖成对计算；基于邻域；处理三种异常值
   - Keywords/ Contributions: knn；凸优化；图表示
   - Link: [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/4418), [Code] (Not-Found yet)
9. **NCMOD**
   - From: "Neighborhood Consensus Networks for Unsupervised Multi-view Outlier Detection", AAAI 2021 (CCF-A)
   - Description: 
     - 核心思想：使用自编码器重构样本，同时为每个视图的图knn邻接矩阵构建共识项，并施加最小距离优化
     - 异常度量：样本重构误差（属性异常）与latent embedding的共识knn距离（类异常）之和
     - 限制/优点：深度方法；不依赖成对计算；基于邻域；处理三种异常值
   - Keywords/ Contributions: 深度学习；knn ；intact space learning；表示学习；自编码器；图表示
   - Link: [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/16873), [[Code]](https://github.com/auguscl/NCMOD)
10. **SRLSP**
    - From: "A Self-Representation Method with Local Similarity Preserving for Fast Multi-View Outlier Detection", TKDD 2023 (CCF-B, JCR Q1)
    - Description: 
      - 核心思想：提出了两个创新模块：邻域自表示与融合自适应相似度学习模块。前者使用邻域的线性组合表示样本点并约束残差项与系数，后者构建跨视图共识系数矩阵并施加最小距离优化 (similar to NCMOD)
      - 异常度量：邻域表示残差项（属性异常）与各视图系数矩阵同共识矩阵的差（类异常）之和
      - 限制/优点：不依赖成对计算；基于子空间（邻域）；处理三种异常值
    - Keywords/ Contributions: knn；子空间学习；自适应相似度学习；凸优化；图表示
    - Link: [[Paper]](https://dl.acm.org/doi/abs/10.1145/3532191), [[Code]](https://github.com/wy54224/SRLSP)
11. **IAMOD**
    - From: "Information-aware Multi-view Outlier Detection", TKDD 2024 (CCF-B, JCR Q1)
    - Description: 
      - 核心思想：基于信息论思想构建两组神经网络：一组编码器压缩数据，并基于对比学习最大化同一样本点不同视图间相似度并最小化不同样本相似度；一组预测器（AE架构）执行跨视图重构（预测）并最小化预测误差。
      - 异常度量：latent embedding的knn距离（属性异常）与跨视图预测误差（类异常）之和
      - 限制/优点：深度方法；依赖成对计算；基于邻域；处理三种异常值
    - Keywords/ Contributions: 深度学习；knn；信息论；对比学习；intact space learning；表示学习
    - Link: [[Paper]](https://dl.acm.org/doi/abs/10.1145/3638354), [[Code]](https://github.com/MaybeLL/IAMOD)
12. **MODGD**
    - From: "Multi-view Outlier Detection via Graphs Denoising", Information Fusion 2024 (JCR Q1)
    - Description: 
      - 核心思想：在每个视图上构建图与邻接矩阵；随后将矩阵分解为共识低秩项与稀疏残差项，并引入权重矩阵和低秩约束以减少属性异常对类异常检测的影响并优化共识图结果
      - 异常度量：邻接矩阵的knn距离（属性异常）与分解后的残差$\mathcal{l}_2$-norm（类异常）之和
      - 限制/优点：基于子空间与邻域；处理三种异常值
    - Keywords/ Contributions: knn；子空间学习；矩阵分解；低秩表示；稀疏表示；凸优化；图表示
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

1. 20newsgroup

   - Description: The 20 Newsgroups data set is a collection of approximately 20,000 newsgroup documents, partitioned (nearly) evenly across 20 different newsgroups. To the best of my knowledge, it was originally collected by Ken Lang, probably for his *[Newsweeder: Learning to filter netnews](http://qwone.com/~jason/20Newsgroups/lang95.bib)* paper, though he does not explicitly mention this collection. The 20 newsgroups collection has become a popular data set for experiments in text applications of machine learning techniques, such as text classification and text clustering.
   - Features: Multi-view (); With anomaly ()
   - Being used in: MUVAD(spilitted into newsm&newsng); MODGD
   - Link: [[20newsgroup-homepage]](https://lig-membres.imag.fr/grimal/data.html); [[20newsgroup-datapage]](http://qwone.com/~jason/20Newsgroups/)

2. AWA2 (AWA-10)

   - Description: This dataset provides a platform to benchmark transfer-learning algorithms, in particular *attribute base classification* and *zero-shot learning* [1]. It can act as a drop-in replacement to the original *Animals with Attributes (AwA)* dataset [2,3], as it has the same class structure and almost the same characteristics.
     It consists of 37322 images of 50 animals classes with pre-extracted feature representations for each image. The classes are aligned with Osherson's classical class/attribute matrix [3,4], thereby providing 85 numeric attribute values for each class. Using the shared attributes, it is possible to transfer information between different classes. The image data was collected from public sources, such as Flickr, in 2016.
   - Features: Multi-view (); With anomaly ()
   - Being used in: SRLSP; IAMOD
   - Link: [[AWA2]](https://cvml.ista.ac.at/AwA2/)

3. BUAA VisNir (BUAA NIR-VIS)

   - Description: 该数据集共有150人，分为红外、彩色和一种本人不了解的图像
   - Features: Multi-view (); With anomaly ()
   - Being used in: DMOD; CRMOD; LDSR
   - Link: [[BUAA VisNir]](https://blog.csdn.net/weixin_42078490/article/details/123065435) *(more related datasets can be viewed from this link!)*

4. Caltech 101

   - Description: Pictures of objects belonging to 101 categories. About 40 to 800 images per category. Most categories have about 50 images. Collected in September 2003 by Fei-Fei Li, Marco Andreetto, and Marc'Aurelio Ranzato. The size of each image is roughly 300 x 200 pixels. We have carefully clicked outlines of each object in these pictures, these are included under the 'Annotations.tar'. There is also a MATLAB script to view the annotations, 'show_annotations.m'.
   - Features: Multi-view (); With anomaly ()
   - Being used in: SRLSP(Caltech-7); IAMOD(Caltech-7); MODGD
   - Link: [[Caltech 101-homepage]](http://www.vision.caltech.edu/datasets/) *(more related datasets can be viewed from this link!)*; [[Caltech 101-datapage]](https://data.caltech.edu/records/mzrjq-6wc02)

5. COIL20

   - Description: Three-view version of COIL-20 (Columbia University Object Image Library, 20 classes), view0 is original grayscale image, view1 is horizontal flip version, and view2 is vertical flip version of original image. They are all 128x128 tensor type. Can be used for multi-view learning.
   - Features: Multi-view (); With anomaly ()
   - Being used in: MODGD
   - Link: [[COIL20]](https://kaggle.com/datasets/yupanliu999/coil20-3v)

6. DBLP

   - Description: The dblp computer science bibliography, subsequently referred to as dblp, is copyright by Schloss Dagstuhl - Leibniz Center for Informatics (German: Schloss Dagstuhl - Leibniz-Zentrum fÃ¼r Informatik GmbH). The metadata provided by dblp on its webpages, as well as their XML, JSON, RDF, RIS, BibTeX, and text export formats available at our website, is released under the CC0 1.0 Public Domain Dedication license. That is, you are free to copy, distribute, use, modify, transform, build upon, and produce derived works from our data, even for commercial purposes, all without asking permission. Of course, we are always happy if you provide a link to us as the source of the data.
   - Features: Multi-view (); With anomaly ()
   - Being used in: HOAD
   - Link: [[DBLP-original]](https://dblp.uni-trier.de/); [[DBLP-kaggle]](https://www.kaggle.com/search?q=DBLP+in%3Adatasets)

7. KDD-Cup 1999

   - Description: This is the data set used for The Third International Knowledge Discovery and Data Mining Tools Competition, which was held in conjunction with KDD-99 The Fifth International Conference on Knowledge Discovery and Data Mining. The competition task was to build a network intrusion detector, a predictive model capable of distinguishing between "bad'' connections, called intrusions or attacks, and "good'' normal connections. This database contains a standard set of data to be audited, which includes a wide variety of intrusions simulated in a military network environment.
   - Features: Multi-view (); With anomaly ()
   - Being used in: CRMOD
   - Link: [[KDD-Cup 1999-original]](https://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html); [[KDD-Cup 1999-kaggle]](https://www.kaggle.com/datasets/galaxyh/kdd-cup-1999-data/data)

8. LandUse-21

   - Description: This is a 21 class land use image dataset meant for research purposes. There are 100 images for each of the following classes， each image measures 256x256 pixels. The images were manually extracted from large images from the USGS National Map Urban Area Imagery collection for various urban areas around the country. The pixel resolution of this public domain imagery is 1 foot.
   - Features: Multi-view (); With anomaly ()
   - Being used in: MODGD
   - Link: [[LandUse-21-original]](http://weegee.vision.ucmerced.edu/datasets/landuse.html); [[LandUse-21-availablenow]](https://www.kaggle.com/datasets/apollo2506/landuse-scene-classification?select=images)

9. MNIST & USPS

   - Description: 
   - Features: Multi-view (); With anomaly ()

   1. MNIST
      - Description: One of the most famous image dataset. MNIST dataset, which is a set of 70,000 small images of digits handwritten by high school students and employees of the US Census Bureau. Each image is labeled with the digit it represents. This set has been studied so much that it is often called the “Hello World” of Machine Learning.
      - Being used in: NCMOD
      - Link: [[MNIST]](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)
   2. USPS
      - Description: An image database for handwritten text recognition research is described. Digital images of approximately 5000 city names, 5000 state names, 10000 ZIP Codes, and 50000 alphanumeric characters are included. Each image was scanned from mail in a working post office at 300 pixels/in in 8-bit gray scale on a high-quality flat bed digitizer. The data were unconstrained for the writer, style, and method of preparation. These characteristics help overcome the limitations of earlier databases that contained only isolated characters or were prepared in a laboratory setting under prescribed circumstances. Also, the database is divided into explicit training and testing sets to facilitate the sharing of results among researchers as well as performance comparisons.
      - Being used in: 
      - Link: [[USPS]](https://www.kaggle.com/datasets/bistaumanga/usps-dataset)
   3. USPS-MNIST
      - Description: 
      - Being used in: CL; MLRA

10. MovieLens-1M

    - Description: GroupLens Research has collected and made available rating data sets from the MovieLens web site ([https://movielens.org](https://movielens.org/)). MovieLens 1M movie ratings. Stable benchmark dataset. 1 million ratings from 6000 users on 4000 movies. Released 2/2003.
    - Features: Multi-view (); With anomaly ()
    - Being used in: HOAD (unknown for specific subdataset); MLRA
    - Link: [[MovieLens-1M]](https://grouplens.org/datasets/movielens/1m/)

11. MSRC-v1

    - Description: The Microsoft Research Cambridge Object Recognition Image Database contains a set of images (digital photographs) grouped into categories. Its intended use is research, in particular object recognition research. Last published: May 18, 2005.
    - Features: Multi-view (); With anomaly ()
    - Being used in: SRLSP; IAMOD
    - Link: [[MSRC-v1]](https://www.microsoft.com/en-us/research/project/image-understanding/downloads/)

12. Oxford Flowers

    - Description: 
    - Features: Multi-view (); With anomaly ()

    1. Oxford 17 Flowers
       - Description: Oxford 17 category flower dataset with 80 images for each class. The flowers chosen are some common flowers in the UK. The images have large scale, pose and light variations and there are also classes with large variations of images within the class and close similarity to other classes.
       - Being used in: CL
       - Link: [[Oxford 17 Flowers]](https://www.kaggle.com/datasets/datajameson/oxford-17-flowers-dataset)
    2. Oxford 102 Flowers
       - Description: We have created a 102 category dataset, consisting of 102 flower categories. The flowers chosen to be flower commonly occuring in the United Kingdom. Each class consists of between 40 and 258 images.
       - Being used in:
       - Link: [[Oxford 102 Flowers]](https://www.kaggle.com/datasets/yousefmohamed20/oxford-102-flower-dataset)

13. REUTERS

    - Description: Reuters Multilingual dataset containing 6 samples of 1200 documents over 6 labels, and desribed by 5 views of 2000 words each.
    - Features: Multi-view (); With anomaly ()
    - Being used in: NCMOD
    - Link: [[REUTERS-original]](https://lig-membres.imag.fr/grimal/data.html); [[REUTERS-kaggle]](https://www.kaggle.com/datasets/nltkdata/reuters) (don't know if the two versions are the same)

14. Titanic

    - Description: The famous dataset for the beginer competition "Titanic - Machine Learning from Disaster" in Kaggle.
    - Features: Multi-view (); With anomaly ()
    - Being used in: NCMOD
    - Link: [[Titanic]](https://www.kaggle.com/competitions/titanic/data)

15. UCI Machine Learning Repository

    - Description: Maintaining 670 datasets as a service to the machine learning community.
    - Features: Multi-view (); With anomaly ()
    - Link: [[UCI Machine Learning Repository]](https://archive.ics.uci.edu/dataset)

    1. credit
       - Description: This research aimed at the case of customers' default payments in Taiwan (of China) and compares the predictive accuracy of probability of default among six data mining methods.
       - Being used in: CRMOD
       - Link: [[UCI-credit-original]](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients); [[UCI-credit-availablenow]](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
    2. digit (Handwritten/Multiple Features)
       - Description: This dataset consists of features of handwritten numerals ('0'--'9') extracted from a collection of Dutch utility maps. 200 patterns per class (for a total of 2,000 patterns) have been digitized in  binary images.
       - Being used in: MODGD
       - Link: [[UCI-digit-original]](https://archive.ics.uci.edu/ml/datasets/Multiple+Features)
    3. ionospere
       - Description: This radar data was collected by a system in Goose Bay, Labrador.  This system consists of a phased array of 16 high-frequency antennas with a total transmitted power on the order of 6.4 kilowatts.
       - Being used in: DMOD; CRMOD; MLRA; MUVAD; SRLSP; IAMOD
       - Link: [[UCI-ionospere-original]](https://archive.ics.uci.edu/dataset/52/ionosphere); [[UCI-ionospere-availablenow]](https://www.kaggle.com/datasets/prashant111/ionosphere)
    4. iris
       - Description: This is one of the earliest datasets used in the literature on classification methods and widely used in statistics and machine learning.  The data set contains 3 classes of 50 instances each, where each class refers to a type of iris plant.  One class is linearly separable from the other 2; the latter are not linearly separable from each other.
       - Being used in: AP; DMOD; CRMOD; MLRA; SRLSP; IAMOD
       - Link: [[UCI-iris-original]](https://archive.ics.uci.edu/dataset/53/iris); [[UCI-iris-availablenow]](https://www.kaggle.com/datasets/uciml/iris)
    5. leaf
       - Description: This dataset consists in a collection of shape and texture features extracted from digital images of leaf specimens originating from a total of 40 different plant species.
       - Being used in: SRLSP
       - Link: [[UCI-leaf-original]](https://archive.ics.uci.edu/dataset/288/leaf)
    6. letter
       - Description: The objective is to identify each of a large number of black-and-white rectangular pixel displays as one of the 26 capital letters in the English alphabet.  The character images were based on 20 different fonts and each letter within these 20 fonts was randomly distorted to produce a file of 20,000 unique stimuli.  Each stimulus was converted into 16 primitive numerical attributes (statistical moments and edge counts) which were then scaled to fit into a range of integer values from 0 through 15.  We typically train on the first 16000 items and then use the resulting model to predict the letter category for the remaining 4000.
       - Being used in: AP; DMOD; CRMOD; MLRA; LDSR; MODDIS; SRLSP; IAMOD
       - Link: [[UCI-letter-original]](https://archive.ics.uci.edu/dataset/59/letter+recognition)
    7. pima
       - Description: This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset. Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.
       - Being used in: MLRA; LDSR; MODDIS; SRLSP; IAMOD
       - Link: [UCI-pima-original]; [[UCI-pima-availablenow]](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
    8. vowel
       - Description: Speaker independent recognition of the eleven steady state vowels of British English using a specified training set of lpc derived log area ratios. The problem is specified by the accompanying data file, "vowel.data".  This consists of a three dimensional array: voweldata [speaker, vowel, input]. The speakers are indexed by integers 0-89.  (Actually, there are fifteen individual speakers, each saying each vowel six times.)  The vowels are indexed by integers 0-10.  For each utterance, there are ten floating-point input values, with array indices 0-9.
       - Being used in: MUVAD
       - Link: [[UCI-vowel-original]](https://archive.ics.uci.edu/dataset/152/connectionist+bench+vowel+recognition+deterding+data)
    9. waveform
       - Description: CART book's waveform domains.
       - Being used in: AP; MLRA (v1)
       - Link: [[UCI-waveform-original-v1];](https://archive.ics.uci.edu/dataset/107/waveform+database+generator+version+1
         ) [[UCI-waveform-original-v2]](https://archive.ics.uci.edu/dataset/108/waveform+database+generator+version+2)
    10. wdbc (Breast Cancer Wisconsin Diagnostic)
        - Description: Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image. n the 3-dimensional space is that described in: [K. P. Bennett and O. L. Mangasarian: "Robust Linear Programming Discrimination of Two Linearly Inseparable Sets", Optimization Methods and Software 1, 1992, 23-34].
        - Being used in: MLRA; LDSR; MODDIS
        - Link: [[UCI-wdbc-original]](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic); [[UCI-wdbc-availablenow]](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
    11. wine
        - Description: These data are the results of a chemical analysis of wines grown in the same region in Italy but derived from three different cultivars. The analysis determined the quantities of 13 constituents found in each of the three types of wines. 
        - Being used in: LDSR; MODDIS
        - Link: [[UCI-wine-original]](https://archive.ics.uci.edu/dataset/109/wine); [[UCI-wine-availablenow]](https://www.kaggle.com/datasets/tawfikelmetwally/wine-dataset)
    12. wobc (breast/Breast Cancer Wisconsin Original)
        - Description: Breast cancer Wisconsin (original) dataset contains real data of 699 observations with independent variables that allows you to classify dependent variable into malignant or benign. The dataset is perfect for logistic regression analysis.
        - Being used in: DMOD; CRMOD
        - Link: [[UCI-breast-original]](https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original); [[UCI-breast-availablenow]](https://www.kaggle.com/datasets/marshuu/breast-cancer)
    13. zoo
        - Description: A simple database containing 17 Boolean-valued attributes.  The "type" attribute appears to be the class attribute.
        - Being used in: AP; MLRA; LDSR; MODDIS; MUVAD; SRLSP; IAMOD
        - Link: [[UCI-zoo-original]](https://archive.ics.uci.edu/dataset/111/zoo); [[UCI-zoo-availablenow]](https://www.kaggle.com/datasets/uciml/zoo-animal-classification)

16. WebKB

    - Description: WebKB datasets containing 4 subsets of documents over 6 labels, and desribed by 2 views (content and citations).
    - Features: Multi-view (); With anomaly ()
    - Being used in: MLRA (Wisconsin)
    - Link: [[WebKB]](https://lig-membres.imag.fr/grimal/data.html)

17. YaleB

    - Description: The cropped dataset only contains the single P00 pose. Same as cropped images here, just converted to PNG instead http://vision.ucsd.edu/~leekc/ExtYaleDatabase/ExtYaleB.html
    - Features: Multi-view (); With anomaly ()
    - Being used in: MODGD
    - Link: [[YaleB-original]](http://vision.ucsd.edu/~leekc/ExtYaleDatabase/ExtYaleB.html) (already unavailable); [[YaleB-availablenow]](https://www.kaggle.com/datasets/tbourton/extyalebcroppedpng)

18. TODO...
