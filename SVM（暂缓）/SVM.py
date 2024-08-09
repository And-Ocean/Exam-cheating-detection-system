import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import model_selection as ms
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc
import pickle


def plot_point2(dataArr, labelArr, Support_vector_index):
    for i in range(np.shape(dataArr)[0]):
        if labelArr[i] == 0:
            plt.scatter(dataArr[i][0], dataArr[i][1], c='b', s=20)
        elif labelArr[i] == 1:
            plt.scatter(dataArr[i][0], dataArr[i][1], c='y', s=20)
        else:
            plt.scatter(dataArr[i][0], dataArr[i][1], c='g', s=20)

    for j in Support_vector_index:
        plt.scatter(dataArr[j][0], dataArr[j][1], s=100, c='', alpha=0.5, linewidth=1.5, edgecolor='red')
    plt.show()


if __name__ == '__main__':
    x = []
    y = []
    # # 分割数据集
    # piece = []
    # count = 0
    # for j in range(1,8):
    #     for i in range(1, 5401):
    #         try:
    #             with open(f"F:/大创/数据集/监考训练集/labels2/{j}_{i}.txt",'r') as f:
    #                 lines = f.readlines()
    #                 count += 1
    #                 piece += [float(a) for a in lines[0].split()]
    #                 if (count % 90 == 0 and count <= 90) or (count % 30 == 0 and count > 90):
    #                     x.append(piece)
    #                     l = int(len(piece) / 3)
    #                     piece = piece[l:]
    #         except:
    #             continue
    #
    # # 把数据整合写入文本
    # with open("data2.txt", 'w') as f:
    #     for data in x:
    #         b = ''
    #         for a in data:
    #             b += str(a) + ' '
    #         f.write(b + '\n')

    # # 从xlsx文件获取标签
    # df = pd.read_excel("F:\大创\数据集\监考训练集/target2.xlsx", index_col=None,header=None)
    # y = [int(value[0]) for value in df.iloc[:,[0]].values]

    # # 将标签写入文本
    # with open("target2.txt",'w') as f:
    #     b = ''
    #     for a in y:
    #         b += str(a) + ' '
    #     f.write(b)

    # 获取训练数据
    with open('data1.txt', 'r') as f:
        for line in f:
            x.append([float(value) for value in line.split()])
    with open('data2.txt', 'r') as f:
        for line in f:
            x.append([float(value) for value in line.split()])
    print(len(x))

    # 获取标签
    with open('target1.txt', 'r') as f:
        lines = f.readlines()
        y += [int(value) for value in lines[0].split()]
    with open('target2.txt', 'r') as f:
        lines = f.readlines()
        y += [int(value) for value in lines[0].split()]
    print(len(y))

    x = np.array(x)
    y = np.array(y)

    #  分割数据集，20%为测试集
    x_train, x_test, y_train, y_test = ms.train_test_split(x, y, test_size=0.2, random_state=42)

    #  训练svm分类器

    # 调参选取最优参数
    svm = GridSearchCV(SVC(), param_grid={"kernel": ['rbf', 'linear', 'poly', 'sigmoid'],
                                          "C": [0.1, 1, 10], "gamma": [1, 0.1, 0.01]}, cv=3)
    svm.fit(x_train, y_train)
    print("The best parameters are %s with a score of %0.2f" % (svm.best_params_, svm.best_score_))
    print('Train accuracy:', svm.score(x_train, y_train))
    print("Test accuracy: ", svm.score(x_test, y_test))

    # 保存模型
    with open('model.pkl', 'wb') as f:
        pickle.dump(svm, f)

    # # 调用模型
    # with open('model.pkl', 'rb') as f:
    #     svm = pickle.load(f)

    # 评估指标
    y_score = svm.score(x_test, y_test)
    print("Test accuracy: ", y_score)
    y_pred = svm.predict(x_test)
    # recall
    target_names = ['class0', 'class1']
    print(classification_report(y_test, y_pred, target_names=target_names))
    # PR曲线
    precision, recall, t = precision_recall_curve(y_test, y_pred)  # y_score是预测的概率值，y_test是真实值标签值
    print(t)  # t是阈值
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.plot(recall, precision)
    plt.title("Precision-Recall")
    plt.show()
    # ROC AUC
    fpr, tpr, threshold = roc_curve(y_test, y_pred)  # y_score是预测概率，y_test是真实值类别
    roc_auc = auc(fpr, tpr)  # 计算auc的值
    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr,
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  # 假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="best")
    plt.show()

    n_Support_vector = svm.n_support_
    print("vector num is : ", n_Support_vector)
    Support_vector_index = svm.support_

    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    h = 0.02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
    plot_point2(x, y, Support_vector_index)
