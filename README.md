# MachineLearning
## First part
学习了机器学习的一些经典的模型，主要分为监督学习（classification，regression），无监督学习（clustering）
### classification
including:LogisticRegression, KNN, NaiveBayes, SVM, DecisionTree, RandomTree
#### **LogisticRgression**
- 可以实现在线梯度下降算法，SGD算法，用于大规模数据处理
#### **KNN**
- 精度高，对异常值不敏感
缺点：计算复杂度高，空间复杂度高
#### **NaiveBayes**:generative learning algorithms
- 简单，在数据较少的情况下仍有效
- 比discriminative收敛更快
- 可以进行多类处理
缺点：学习不了特征间的关系
- 对输入数据敏感，不可缺失特征
#### **Decision Tree**
- 处理特征间的交互关系，
- 不担心异常值，可以处理非线性
缺点：不支持在线学习，
- 易过拟合
### **SVMs**
- 高准确率，更好的避免过拟合，处理数据不可分（适当的核函数）
缺点：内存消耗大，难调参