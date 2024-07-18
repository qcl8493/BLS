# BLS增量学习
from ucimlrepo import fetch_ucirepo
import time
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('9types_feature_extract.csv')

X = data.iloc[:4750, 1:]   #第0行到4749行
Y = data.iloc[:4750, 0]    #data.iloc只抽取数据，行名和列名不算在内，并且序号都从零开始
new_x = data.iloc[4750:4751, 1:]
new_y = data.iloc[4750:4751, 0]

output_data = np.zeros((4750, 9))
Y = Y.values
for i in range(output_data.shape[0]):
    if Y[i] == 0:
        output_data[i] = [1, 0, 0, 0, 0, 0, 0, 0, 0]
    elif Y[i] == 1:
        output_data[i] = [0, 1, 0, 0, 0, 0, 0, 0, 0]
    elif Y[i] == 2:
        output_data[i] = [0, 0, 1, 0, 0, 0, 0, 0, 0]
    elif Y[i] == 3:
        output_data[i] = [0, 0, 0, 1, 0, 0, 0, 0, 0]
    elif Y[i] == 4:
        output_data[i] = [0, 0, 0, 0, 1, 0, 0, 0, 0]
    elif Y[i] == 5:
        output_data[i] = [0, 0, 0, 0, 0, 1, 0, 0, 0]
    elif Y[i] == 6:
        output_data[i] = [0, 0, 0, 0, 0, 0, 1, 0, 0]
    elif Y[i] == 7:
        output_data[i] = [0, 0, 0, 0, 0, 0, 0, 1, 0]

X_array = X.values
new_x = new_x.values

# 转换数组形状
X_reshaped = X_array.reshape(4750, 1, 73)
new_X = new_x.reshape(1, 1, 73)                                                            #new_X为new_x变换成可用格式的矩阵
array = np.zeros((1, 8))
new_Y = np.c_[array, np.ones(1)]

X_train, X_test, y_train, y_test = train_test_split(X_reshaped, output_data, test_size=0.33, random_state=42)

def sigmoid(x):
    sig = 1.0 / (1 + np.exp(-x))
    return sig

def tanh(x):
    tanh = (np.exp(x) - np.exp(-1.0 * x)) / (np.exp(x) + np.exp(-1.0 * x))
    return tanh

def feature_map(input_data):
    print('input_data.shape:',input_data.shape)
    i, j, k = input_data.shape
    W_z = np.random.rand(k, k)      # 矩阵大小待修改
    BETA_z = np.random.rand(1)
    Z = np.zeros((j, i, k))
    for l in range(j):
        Z[l] = input_data[:, l, :].dot(W_z) + BETA_z                                                 #与输入input_data相关
        # 激活函数
        Z[l] = sigmoid(Z[l])
        if l == 0:
            Z_n = Z[l]
        else:
            Z_n = np.concatenate((Z_n, Z[l]), axis=1)
    return Z, Z_n, W_z, BETA_z                                               #z与z_n与input相关，W_z 、BETA_z与input_data无关


# enhancement feature nodes
def enhancement(feature):
    i, j, k = feature.shape
    W_h = np.random.rand(k, k)
    BETA_h = np.random.rand(1)
    H = np.zeros((i, j, k))
    for l in range(i):
        H[l] = feature[l].dot(W_h) + BETA_h                                                             #与输入feature相关
        # 激活函数
        H[l] = sigmoid(H[l])
        H_m = H[l]
        if l == 0:
            H_m = H[l]
        else:
            H_m = np.concatenate((H_m, H[l]), axis=1)
    return H, H_m, W_h, BETA_h                                                                    #H, H_m与输入feature相关


def weight(A, Y):
    W = np.linalg.pinv(A.T.dot(A)).dot(A.T).dot(Y)
    return W


def classfication(input, W_z, BETA_z, W_h, BETA_h, W):
    i, j, k = input.shape
    Z = np.zeros((j, i, k))
    for l in range(j):
        Z[l] = input[:, l, :].dot(W_z) + BETA_z                                                              #与input有关
        # 激活函数
        Z[l] = sigmoid(Z[l])
        if l == 0:
            Z_n = Z[l]
        else:
            Z_n = np.concatenate((Z_n, Z[l]), axis=1)
    H = np.zeros((j, i, k))
    for l in range(j):
        H[l] = Z[l].dot(W_h) + BETA_h
        # 激活函数
        H[l] = sigmoid(H[l])
        H_m = H[l]
        if l == 0:
            H_m = H[l]
        else:
            H_m = np.concatenate((H_m, H[l]), axis=1)
    A = np.concatenate((Z_n, H_m), axis=1)
    Y_hat = A.dot(W)
    return Y_hat


def train_model(train_x, train_y):
    Z, Z_n, W_z, BETA_z = feature_map(train_x)
    H, H_m, W_h, BETA_h = enhancement(Z)
    A = np.concatenate((Z_n, H_m), axis=1)

    W = weight(A, train_y)
    parameters = {"W_z": W_z, "BETA_z": BETA_z, "H": H, "W_h": W_h, "BETA_h": BETA_h, "A": A, "W": W}
    return parameters


def increment(parameters, new_x, new_y):
    W_z = parameters["W_z"]
    BETA_z = parameters["BETA_z"]
    W_h = parameters["W_h"]
    # U = parameters["U"]
    BETA_h = parameters["BETA_h"]
    A = parameters["A"]
    W = parameters["W"]
    i, j, k = new_x.shape
    # new_x = np.reshape(new_x, (1, i, j))                                                                  #
    # feature_map(new_X)                                                                                              #
    Z_a = np.zeros((j, i, k))
    for l in range(j):
        Z_a[l] = new_x[:, l, :].dot(W_z) + BETA_z
        Z_a[l] = sigmoid(Z_a[l])
        if l == 0:
            Z_an = Z_a[l]
        else:
            Z_an = np.concatenate((Z_an, Z_a[l]), axis=1)
    # enhancement node
    H0 = np.zeros((i, k))
    H_a = np.zeros((j, i, k))
    for l in range(j):
        if l == 0:
            H_a[l] = Z_a[l].dot(W_h)+ BETA_h
            # 激活函数
            H_a[l] = sigmoid(H_a[l])
            H_am = H_a[l]
        else:
            H_a[l] = Z_a[l].dot(W_h)  + BETA_h
            # 激活函数
            H_a[l] = sigmoid(H_a[l])
            H_am = np.concatenate((H_am, H_a[l]), axis=1)
    # for l in range(j):                                                                                                #
    #  if l == 0:
    #     H_a[l] = Z_a[l].dot(W_h) + H0.dot(U) + BETA_h
    #     # 激活函数
    #     H_a[l] = sigmoid(H_a[l])
    #     H_am = H_a[l]
    #  else:
    #     H_a[l] = Z_a[l].dot(W_h) + H_a[l - 1].dot(U) + BETA_h
    #     # 激活函数
    #     H_a[l] = sigmoid(H_a[l])
        # H_am = np.concatenate((H_am, H_a[l]), axis=1)                                                           #
    A_a = np.concatenate((Z_an, H_am), axis=1)
    A_mna = np.concatenate((A, A_a))
    D = np.linalg.pinv(A.T).dot(A_a.T)
    C = A_a.T - A.T.dot(D)
    # print(C)
    # if C.all == 0:                                                                                                   #
    B = np.linalg.pinv(1+D.T.dot(D)).dot(D.T).dot(np.linalg.pinv(A.T))
    # else:                                                                                                            #
    #     B = np.linalg.pinv(C)                                                                                        #
    B = B.T
    # print(B)
    W_mna = W + B.dot(new_y-A_a.dot(W))
    # W_mna = W_mna.astype(np.float)                                                                                    #
    # if np.isnan(W_mna).any() == True or np.isinf(W_mna).any() == True:
    #     W_mna = W
    #     print(W_mna)                                                                                                  #
    parameters["A"] = A_mna
    parameters["W"] = W_mna
    return parameters


tic = time.time()
# RBLS-ELM
parameters = train_model(X_train, y_train)
y_hat_0 = classfication(X_test, parameters["W_z"], parameters["BETA_z"], parameters["W_h"],
                        parameters["BETA_h"], parameters["W"])
'''从这开始写'''

parameters = increment(parameters, new_X, new_Y)
y_hat_1 = classfication(X_test, parameters["W_z"], parameters["BETA_z"], parameters["W_h"],
                        parameters["BETA_h"], parameters["W"])



toc = time.time()

idx = y_hat_0.argmax(axis=1)
out = (idx[:, None] == np.arange(y_hat_0.shape[1])).astype(float)
print('训练数据集输出矩阵形式：',out.shape)

# print(out)
# 准确率
correct = np.sum((out == y_test).all(1))/y_test.shape[0]
print('The test accuracy is: %.4f' % correct)
print('Test time: %.4f' % (toc-tic))


idx_1 = y_hat_1.argmax(axis=1)
out_1 = (idx_1[:, None] == np.arange(y_hat_1.shape[1])).astype(float)  # 将预测类别转换为独热编码
print('增量数据集输出矩阵形式',out_1.shape)

# 计算准确率
correct_1 = np.sum((out_1 == y_test).all(1)) / y_test.shape[0]
print('The test accuracy for y_hat_1 is: %.4f' % correct_1)

