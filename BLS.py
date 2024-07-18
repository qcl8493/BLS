# BLS增量学习
from ucimlrepo import fetch_ucirepo
import time
import numpy as np
from sklearn.preprocessing import MinMaxScaler
# fetch dataset
iris = fetch_ucirepo(id=53)

# data (as pandas dataframes)
X = iris.data.features
y = iris.data.targets

input_data = X.valueshu
output_data = np.zeros((150, 3))
Y = y.values
for i in range(Y.shape[0]):
    if Y[i] == 'Iris-setosa':
        output_data[i] = [1, 0, 0]
    elif Y[i] == 'Iris-versicolor':
        output_data[i] = [0, 1, 0]
    else:
        output_data[i] = [0, 0, 1]


normalization = input_data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(normalization)

X1 = scaled[0:50]
X2 = scaled[50:100]
X3 = scaled[100:150]

train_x = np.concatenate((X1[:30], X2[:30], X3[:30]))
test_x = np.concatenate((X1[30:], X2[30:], X3[30:]))
train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
test_x = np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))


Y1 = output_data[0:50]
Y2 = output_data[50:100]
Y3 = output_data[100:150]

train_y = np.concatenate((Y1[:30], Y2[:30], Y3[:30]))
test_y = np.concatenate((Y1[30:], Y2[30:], Y3[30:]))


def sigmoid(x):
    sig = 1.0 / (1 + np.exp(-x))
    return sig


def tanh(x):
    tanh = (np.exp(x) - np.exp(-1.0 * x)) / (np.exp(x) + np.exp(-1.0 * x))
    return tanh


def feature_map(input_data):
    i, j, k = input_data.shape
    W_z = np.random.rand(k, k)      # 矩阵大小待修改
    BETA_z = np.random.rand(1)
    Z = np.zeros((j, i, k))
    for l in range(j):
        Z[l] = input_data[:, l, :].dot(W_z) + BETA_z
        # 激活函数
        Z[l] = sigmoid(Z[l])
        if l == 0:
            Z_n = Z[l]
        else:
            Z_n = np.concatenate((Z_n, Z[l]), axis=1)
    return Z, Z_n, W_z, BETA_z


# enhancement feature nodes
def enhancement(feature):
    i, j, k = feature.shape
    W_h = np.random.rand(k, k)
    BETA_h = np.random.rand(1)
    H = np.zeros((i, j, k))
    for l in range(i):
        H[l] = feature[l].dot(W_h) + BETA_h
        # 激活函数
        H[l] = sigmoid(H[l])
        H_m = H[l]
        if l == 0:
            H_m = H[l]
        else:
            H_m = np.concatenate((H_m, H[l]), axis=1)
    return H, H_m, W_h, BETA_h


def weight(A, Y):
    W = np.linalg.pinv(A.T.dot(A)).dot(A.T).dot(Y)
    return W


def classfication(input, W_z, BETA_z, W_h, BETA_h, W):
    i, j, k = input.shape
    Z = np.zeros((j, i, k))
    for l in range(j):
        Z[l] = input[:, l, :].dot(W_z) + BETA_z
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
    U = parameters["U"]
    BETA_h = parameters["BETA_h"]
    A = parameters["A"]
    W = parameters["W"]
    i, j, k = new_x.shape
    # new_x = np.reshape(new_x, (1, i, j))
    # feature map
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
            H_a[l] = Z_a[l].dot(W_h) + H0.dot(U) + BETA_h
            # 激活函数
            H_a[l] = sigmoid(H_a[l])
            H_am = H_a[l]
        else:
            H_a[l] = Z_a[l].dot(W_h) + H_a[l - 1].dot(U) + BETA_h
            # 激活函数
            H_a[l] = sigmoid(H_a[l])
            H_am = np.concatenate((H_am, H_a[l]), axis=1)
    A_a = np.concatenate((Z_an, H_am), axis=1)
    A_mna = np.concatenate((A, A_a))
    D = np.linalg.pinv(A.T).dot(A_a.T)
    C = A_a.T - A.T.dot(D)
    # print(C)
    # if C.all == 0:
    B = np.linalg.pinv(1+D.T.dot(D)).dot(D.T).dot(np.linalg.pinv(A.T))
    # else:
    #     B = np.linalg.pinv(C)
    B = B.T
    # print(B)
    W_mna = W + B.dot(new_y-A_a.dot(W))
    # W_mna = W_mna.astype(np.float)
    # if np.isnan(W_mna).any() == True or np.isinf(W_mna).any() == True:
    #     W_mna = W
    #     print(W_mna)
    parameters["A"] = A_mna
    parameters["W"] = W_mna
    return parameters


tic = time.time()
# RBLS-ELM
parameters = train_model(train_x, train_y)
y_hat_0 = classfication(test_x, parameters["W_z"], parameters["BETA_z"], parameters["W_h"],
                        parameters["BETA_h"], parameters["W"])

# qcl
# parameters = increment(parameters, new_x, new_y)
# y_hat_1 = classfication(test_x, parameters["W_z"], parameters["BETA_z"], parameters["W_h"],
#                         parameters["BETA_h"], parameters["W"])
#
# #
# y_hat_0 ?= y_hat_0

toc = time.time()

idx = y_hat_0.argmax(axis=1)
out = (idx[:, None] == np.arange(y_hat_0.shape[1])).astype(float)
print(out.shape)

# print(out)
# 准确率
correct = np.sum((out == test_y).all(1))/test_y.shape[0]
print('The test accuracy is: %.4f' % correct)
print('Test time: %.4f' % (toc-tic))

