import operator
import numpy as np

from matplotlib.font_manager import FontProperties
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

label_map = {
    'didntLike': 1,
    'smallDoses': 2,
    'largeDoses': 3,
}


def file2matrix():
    # open and read file
    with open("./datingTestSet.txt") as f:
        array_lines = f.readlines()
        line_nums = len(array_lines)

    # trans to numpy array
    ret_mat = np.zeros((line_nums, 3))
    label_vector = []
    for line_nu, array_line in enumerate(array_lines):
        # print(line_nu,array_line)
        line_list = array_line.split()
        ret_mat[line_nu, 0:3] = line_list[0:3]
        label_vector.append(label_map[line_list[-1]])
    # print(ret_mat[1,:])
    # print(label_vector)
    return ret_mat, label_vector


def show_data(mat, labels):
    fig, axs = plt.subplots(nrows=2,
                            ncols=2,
                            sharex=False,
                            sharey=False,
                            figsize=(13, 8))
    # number_labels = len(labels)
    label_colors = [
        "black" if label == 1 else ("orange" if label == 2 else "red")
        for label in labels
    ]

    axs[0][0].scatter(x=mat[:, 0],
                      y=mat[:, 1],
                      color=label_colors,
                      s=5,
                      alpha=0.5)
    # print(label_colors)
    axs[0][0].set_title(u"fly distance and play games")
    axs[0][0].set_xlabel(u"fly")
    axs[0][0].set_ylabel(u"games")

    axs[0][1].scatter(x=mat[:, 0],
                      y=mat[:, 2],
                      color=label_colors,
                      s=5,
                      alpha=0.5)
    # print1label_colors)
    axs[0][1].set_title(u"fly distance and ice cream")
    axs[0][1].set_xlabel(u"fly")
    axs[0][1].set_ylabel(u"cream")

    axs[1][0].scatter(x=mat[:, 1],
                      y=mat[:, 2],
                      color=label_colors,
                      s=5,
                      alpha=0.5)
    # pr1nt(label_colors)
    axs[1][0].set_title(u"play games and ice cream")
    axs[1][0].set_xlabel(u"games")
    axs[1][0].set_ylabel(u"cream")

    # add legend
    plt.show()


def norm(datas):
    min_vals = datas.min(0)
    max_val = datas.max(0)
    print(datas.max(0))
    ranges = max_val - min_vals
    norm_datas = np.zeros(np.shape(datas))
    row_len = datas.shape[0]
    norm_datas = datas - np.tile(min_vals, (row_len, 1))
    norm_datas = norm_datas / np.tile(ranges, (row_len, 1))
    print(norm_datas)
    return norm_datas, ranges, min_vals


def knn_classify(predict, datas, labels, k):
    '''
    the algorithm of knn
    predict: the data need to be classified
    datas: dataset
    labels: labels
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


def date_class_test():
    mat, labels = file2matrix()

    norm_mat, ranges, minvals = norm(mat)
    m = norm_mat.shape[0]

    ho_ratio = 0.1
    nums_test = int(m * ho_ratio)
    err_count = 0.0
    for i in range(nums_test):
        res = knn_classify(norm_mat[i, :], norm_mat[nums_test:m, :],
                           labels[nums_test:m], 4)
        print("分类结果{}, 真实的类别{}".format(res, labels[i]))
        if res != labels[i]:
            err_count += 1.0

    print("错误率{}%".format(err_count / float(nums_test) * 100))
    # show_data(mat, labels)


def knn_input():
    # get input
    fly_distance = int(input("fly distance:"))
    play_games = int(input("play games:"))
    ice_cream = int(input("ice cream:"))

    mat, labels = file2matrix()
    norm_mat, ranges, minvals = norm(mat)

    np_arr = np.array([fly_distance, play_games, ice_cream])
    np_arr_to1 = np_arr - minvals / ranges

    res = knn_classify(np_arr_to1, norm_mat, labels, 4)
    # print(res)
    labels_revert = {v: k
                     for k, v in label_map.items()}  # revert the label map
    print("you may {} the {}".format(
        labels_revert[res],
        dict(
            zip(("fly", "games", "cream"),
                [fly_distance, play_games, ice_cream]))))


def main():
    mat, labels = file2matrix()
    # norm(mat)
    show_data(mat, labels)
    # date_class_test()
    # knn_input()


if __name__ == "__main__":
    main()
