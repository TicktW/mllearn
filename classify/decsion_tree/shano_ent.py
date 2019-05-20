import math
import os
import pandas as pd
import numpy as np
import pickle


def createDataSet():
    '''
    ["年龄", "工作", "房子", "信贷", "类别"]
    '''
    dataSet = [
        [0, 0, 0, 0, 'no'],  #数据集
        [0, 0, 0, 1, 'no'],
        [0, 1, 0, 1, 'yes'],
        [0, 1, 1, 0, 'yes'],
        [0, 0, 0, 0, 'no'],
        [1, 0, 0, 0, 'no'],
        [1, 0, 0, 1, 'no'],
        [1, 1, 1, 1, 'yes'],
        [1, 0, 1, 2, 'yes'],
        [1, 0, 1, 2, 'yes'],
        [2, 0, 1, 2, 'yes'],
        [2, 0, 1, 1, 'yes'],
        [2, 1, 0, 1, 'yes'],
        [2, 1, 0, 2, 'yes'],
        [2, 0, 0, 0, 'no']
    ]
    labels = ['不放贷', '放贷']  #分类属性
    return dataSet, labels  #返回数据集和分类属性


def shanno_ent(data_set, label_index):
    # yes_count = 0
    # all_count = len(data_set)
    # for data in data_set:
    #     if data[-1] == "yes":
    #         yes_count += 1
    data_cnt = len(data_set)
    label_cnt = {}

    for data in data_set:
        cur_label = data[label_index]
        if cur_label not in label_cnt.keys():
            label_cnt[cur_label] = 1
        else:
            label_cnt[cur_label] += 1

    shanno_ent = 0.0
    for key in label_cnt:
        prob = float(label_cnt[key] / data_cnt)
        shanno_ent -= prob * math.log(prob, 2)

    print(shanno_ent)
    return shanno_ent


def choose_label(df):
    '''
    由信息增益判断选择哪个label作为选择依据
    '''
    # calculate the info_gain
    # info_gain = sum(P(one_info)H(label))
    info_gains = []

    for col in df.columns[:-1]:
        group_df = df.groupby(by=col)
        size_df = group_df.size().reset_index(name="times")
        all_cnt = size_df.apply(lambda x: x.sum())["times"]
        # print(size_df)
        # print(size_df[col].values)
        shanno_li = []
        for val in size_df[col].values:
            group = group_df.get_group(val)
            print(group)
            shanno_li.append(size_df.loc[val]["times"] / all_cnt *
                             shanno_ent(group.as_matrix(), -1))
        print(shanno_li)
        data_set_shano = shanno_ent(df.as_matrix(), -1)
        info_gain = data_set_shano - sum(shanno_li)
        print(info_gain)
        info_gains.append(info_gain)
    selecte_freature = df.columns[info_gains.index(max(info_gains))]
    print("The selected spec is '{}'".format(selecte_freature))
    return selecte_freature


def create_tree(df, ds_tree, labels):
    '''
    create a decision tree
    决策树
    '''
    # when exit recursion
    # if len( set(df["class"]) ) <= 1:
    #     ds_tree = df["class"]
    #
    #return
    if not labels:
        return
    # select a feature whose info gain max
    feature = choose_label(df)

    # init ds_tree
    ds_tree[feature] = {}

    # group by data_set the feature
    group_df = df.groupby(by=feature)
    size_df = group_df.size().reset_index(name="times")

    # create tree for child group
    for val in size_df[feature].values:
        group = group_df.get_group(val)
        gval_set = set(group["class"])
        if len(gval_set) <= 1:
            ds_tree[feature][val] = list(gval_set)[0]
        else:
            ds_tree[feature][val] = {}
            create_tree(group, ds_tree[feature][val], labels)
    else:
        labels.remove(feature)

##.1
print("ok")
print("ok")

def cache_tree():
    if os.path.exists("ds_tree"):
        with open("ds_tree", "rb") as f:
            return pickle.load(f)

    data_set, labels = createDataSet()
    columns = ["age", "work", "house", "borrow", "class"]
    df = pd.DataFrame(data=data_set, columns=columns)
    # choose_label(df)
    ds_tree = {}
    columns.remove("class")
    create_tree(df, ds_tree, columns)
    with open("ds_tree", "wb") as f:
        pickle.dump(ds_tree, f)
        return ds_tree



def predict(df, ds_tree):
    for i in range(df.shape[0]):
        print(df)
        row = df.iloc[i, :]
        print(row)
        while 1:
            try:
                label, child_tree = ds_tree.popitem()
            except FileNotFoundError as ffe:
                return ds_tree
            except AttributeError as ae:
                return ds_tree
            print(child_tree)
            print(row[label])
            ds_tree = child_tree[row[label]]
            print(ds_tree)
            # row[ds_tree.keys()[0]]


def main():
    # data_set,labels =  createDataSet()

    # shanno_ent(data_set, -1)
    ds_tree = cache_tree()
    ds, la = createDataSet()

    columns = ["age", "work", "house", "borrow", "class"]
    df = pd.DataFrame(data=ds, columns=columns)
    res = predict(df, ds_tree)
    print("predict result is '{}'".format(res))


if __name__ == "__main__":
    main()
