## 用bert做关系抽取

采用的数据集是SemEval2010，将任务抽象为文本分类任务，共10个类别，[具体任务介绍][https://docs.google.com/document/d/1QO_CnmvNRnYwNWu1-QCAeR5ToQYkXUqFeAJbdEhsq7w/preview]

同时也是对[[Enriching Pre-trained Language Model with Entity Information for Relation Classification](https://arxiv.org/abs/1905.08284)]的简易版复现，个人论文主要创新点是在输出层中加入实体标识符，这在关系抽取中是常见的方式

预训练模型采用的是bert_uncased_L-12_H-768_A-12/1

环境要求：

- python 3.x
- tensorflow 1.x

**运行单次预测**

```python
# predict
test = "The <e1>company</e1> fabricates plastic <e2>chairs</e2>."
print(predict_single_relation(test))
```

**超参数设置**

```python
MAX_SEQ_LENGTH = 128,
BATCH_SIZE = 32,
LEARNING_RATE = 2e-5,
NUM_TRAIN_EPOCHS = 3.0,
WARMUP_PROPORTION = 0.1,
SAVE_SUMMARY_STEPS = 100,
SAVE_CHECKPOINTS_STEPS = 10000,
bert_model_hub = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1")

"LEARNING_RATE": 2e-5,
"NUM_TRAIN_EPOCHS":3
```

**结果**

|               |     values |
| ------------: | ---------: |
|      micro-f1 |   0.840265 |
|          loss |   0.571484 |
|   global_step | 750.000000 |

**分析**

世纪效果效果不如paper中写的89.25，一方面是训练次数不够，第二是所用的bert模型相对较小，第三是在计算f1值的时候未去除“Other”的影响



> 参考资料：
>
> https://zhuanlan.zhihu.com/p/78445887
>
> https://zhuanlan.zhihu.com/p/61671334