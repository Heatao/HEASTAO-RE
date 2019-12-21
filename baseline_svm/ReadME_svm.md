# svm做分类

本次实验选择svm做为baseline进行对比

选择了两个特征做对比，一个是采用的tfidf，一个是用glove向量，前者是将训练集和测试集的文本全部做一次tfidf的向量变换，后者的处理是将句子的每一个词查glove词表得到向量，再求平均。

**参数选择**

```python
clf = svm.SVC(kernel='rbf', decision_function_shape='ovr', gamma='scale', class_weight='balanced')
```

词向量选择的是*glove.68.100d*

**预测**

- 调用*get_relation_svm.py*

- ```python
  test = 'The most common <e1>audits</e1> were about <e2>waste</e2> and recycling.'
  print(svm_pre(test))
  ```

**结果**

|           | macro-f1 |
| --------: | -------: |
| glove+svm |     0.41 |
| tfidf+svm |     0.49 |

**分析**

采用句子的平均词向量反而不如tfidf，原因可能是因为取平均后句子的语义会有一定的丢失



> 参考资料：
>
> https://scikit-learn.org/stable/