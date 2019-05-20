import numpy as np
import operator
import os
from sklearn.neighbors import KNeighborsClassifier as knn


def mat2vector(file_name):
    vect = np.zeros((1, 1024))

    with open(file_name) as f:
        for num, line in enumerate(f):
            for index, char in enumerate(line[:-1]):
                # print(line, char)
                # vect[num*32 + index] = char error, numpy array 需要显示声明行列
                vect[0, num * 32 + index] = int(char)
    return vect

def sk_knn():
    train_labels = []
    train_flist = os.listdir("./digit/trainingDigits")
    train_len = len(train_flist)
    train_mat = np.zeros((train_len, 1024))

    for i,fname in enumerate(train_flist):
        # print(fname)
        flabel = int(fname.split("_")[0])
        train_mat[i,:] = mat2vector("./digit/trainingDigits/{}".format(fname))
        train_labels.append(flabel)
        # break

    print(train_labels)
    print(train_mat)
    knn_instance = knn(n_neighbors=3)  #TODO neighbors <= 5效果最好
    knn_instance.fit(train_mat, train_labels)

    test_flist = os.listdir("./digit/testDigits")  # test file list
    err_count = 0
    for fname in test_flist:
        test_label = int(fname.split("_")[0])
        test_mat = mat2vector("./digit/testDigits/{}".format(fname))
        res = knn_instance.predict(test_mat)
        if res != test_label:
            err_count += 1
            print("error,the predict res is {}, the real label is {}".format(res, test_label))
            print("./digit/testDigits/{}".format(fname))
    print("the error rate is {}%".format(err_count / len(test_flist) * 100))



def main():
    # print(mat2vector("./digit/testDigits/0_0.txt"))
    sk_knn()


if __name__ == "__main__":
    main()
