本项目是kaggle上房价预测，包括以下三部分内容
（1）数据分析，因属性数量较多，重点挑选与房价相关性排名前10的属性进行分析，通过散点图矩阵和相关系数热度图分析特征与房价的关系，使用直方图矩阵观察数据分布。
（2）数据预处理，主要进行缺失值处理，数据变换，属性构造和离群点处理。对缺失属性，基于缺失属性的数据类别采取中位数/众数填补和回归填补。对于时间特征，进行变换处理；对存在偏度的数值特征进行对数变换；对有序的类别特征，分别采用有序映射和独热处理两种方法进行对比；对于离群点，建立线性回归，基于3σ原则判定离群点并删除。
（3）建模，首先基于gridsearchCV寻找性能较佳的基分类器，包括linearRegression,SVR,基于树模型的 bagging和boosting方法，然后分析分类器的相关性，对性能较优且相关度低的分类器进行集成，对比不同组合的voting和stacking方法的预测效果。
