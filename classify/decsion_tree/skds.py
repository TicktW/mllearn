#!/usr/bin/env python
# coding: utf-8
# In[1]:
# imports
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import pydotplus
from sklearn.externals.six import StringIO
import numpy as np
# In[1]

# In[2]:
# readfiles and format
# 年龄 症状 是否散光 眼泪数量 分类标签
with open("./lenses.txt") as f:
    lenses = [line.strip().split("\t") for line in f.readlines()]
lenses_labels = ['age', 'prescript', 'astigmatic', 'tearRate', 'labels']
lenses
df = pd.DataFrame(lenses, columns=lenses_labels)
le = LabelEncoder()
for col in df.columns:
    df[col] = le.fit_transform(df[col])
df_label = df['labels'].to_list()
df_data_frame = df.drop("labels", axis=1)
df_data = df_data_frame.values.tolist()

# In[2]

# In[3]:
# cluster func
clf = tree.DecisionTreeClassifier()
clf.fit(df_data, df_label)
dot_data = StringIO()
tree.export_graphviz(
    clf,
    out_file=dot_data,  # 绘制决策树
    feature_names=df_data_frame.keys(),
    class_names=clf.classes_,
    rounded=True,
    special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("tree.pdf")
# In[3]

# In[4]:
# trans to pandas

clf.predict([[1, 2, 0, 1], [1, 0, 1, 0]])
# In[4]
