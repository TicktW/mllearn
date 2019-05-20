import numpy as np
import operator


def create_data_set():
    '''
    create data set for knn
    '''
    datas = np.array([[1, 101], [5, 89], [108, 5], [115, 8]])
    labels = [0, 0, 1, 1]  # 0->emotion 1->action
    return datas, labels


def knn_classify(predict, datas, labels, k):
    '''
    the algorithm of knn
    predict: the data need to be classified
    '''
    # datas = np.array()
    datas_size = datas.shape[0]
    diff_mat = np.tile(predict, (datas_size, 1)) - datas  # repeat the matrix
    sq_diff_mat = diff_mat**2
    sq_distance = sq_diff_mat.sum(1)  # add elements every row
    distance = sq_distance**0.5
    sort_index = distance.argsort()
    class_count = {}
    for i in range(k):
        label = labels[sort_index[i]]
        class_count[label] = class_count.get(label, 0) + 1
    print(diff_mat)
    sorted_class_count = sorted(class_count.items(),
                                key=operator.itemgetter(1),
                                reverse=True)
    return sorted_class_count[0][0]


def main():
    datas, labels = create_data_set()
    tdata = [101, 20]
    result = knn_classify(tdata, datas, labels, 3)
    print("result is")
    print(result)


if __name__ == "__main__":
    main()
