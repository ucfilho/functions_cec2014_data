# start working util funciton 27-step 1
import numpy as np
sin = np.sin
cos = np.cos
sqrt = np.sqrt
pi = np.pi
exp = np.exp
e = np.e
dot = np.dot
array = np.array
sum = np.sum
matmul = np.matmul
where = np.where
sign = np.sign
min = np.min
round = np.round
ceil = np.ceil
ones = np.ones
concatenate = np.concatenate


def F1(solution , shift_data , matrix):
       
    #   "F1":"Rotated High Conditioned Elliptic Function"
    problem_size = len(solution)
    shift_data = shift_data[:problem_size]
    z = dot(solution - shift_data, matrix)
    return z


def F2(solution , shift_data , matrix):
    #   "F2":"Rotated Bent Cigar Function"
    problem_size = len(solution)
    shift_data = shift_data[:problem_size]
    z = dot(solution - shift_data, matrix)
    return z


def F3(solution , shift_data , matrix):
    #   "F3":"Rotated Discus Function"
    problem_size = len(solution)
    shift_data = shift_data[:problem_size]
    z = dot(solution - shift_data, matrix)
    return z


def F4(solution , shift_data , matrix):
    #   "F4":"Shifted and Rotated Rosenbrock’s Function"
    problem_size = len(solution)
    shift_data = shift_data[:problem_size]
    z = 2.048 * (solution - shift_data) / 100
    z = dot(z, matrix) + 1
    return z


def F5(solution , shift_data , matrix):
    #   "F5":"Shifted and Rotated Ackley’s Function"
    problem_size = len(solution)
    shift_data = shift_data[:problem_size]
    z = dot(solution - shift_data, matrix)
    return z


def F6(solution , shift_data , matrix):
    #   "F6":"Shifted and Rotated Weierstrass Function"
    problem_size = len(solution)
    shift_data = shift_data[:problem_size]
    z = 0.5 * (solution - shift_data) / 100
    z = dot(z, matrix)
    return z


def F7(solution , shift_data , matrix):
    #   "F7":"Shifted and Rotated Griewank’s Function"
    problem_size = len(solution)
    shift_data = shift_data[:problem_size]
    z = 600 * (solution - shift_data) / 100
    z = dot(z, matrix)
    return z


def F8(solution , shift_data , matrix):
    #   "F8":"Shifted Rastrigin’s Function"
    problem_size = len(solution)
    shift_data = shift_data[:problem_size]
    z = 5.12 * (solution - shift_data) / 100
    z = dot(z, matrix)
    return z


def F9(solution , shift_data , matrix):
    #   "F9":"Shifted and Rotated Rastrigin’s Function"
    problem_size = len(solution)
    shift_data = shift_data[:problem_size]
    z = 5.12 * (solution - shift_data) / 100
    z = dot(z, matrix)
    return z


def F10(solution , shift_data , matrix):
    #   "F10":"Shifted Schwefel’s Function"
    problem_size = len(solution)
    shift_data = shift_data[:problem_size]
    z = 1000 * (solution - shift_data) / 100
    z = dot(z, matrix)
    return f9_modified_schwefel__(z) + bias


def F11(solution , shift_data , matrix):
    #   "F11":"Shifted and Rotated Schwefel’s Function"
    problem_size = len(solution)
    shift_data = shift_data[:problem_size]
    z = 1000 * (solution - shift_data) / 100
    z = dot(z, matrix)
    return z


def F12(solution , shift_data , matrix):
    #   "F12":"Shifted and Rotated Katsuura Function"
    problem_size = len(solution)
    shift_data = shift_data[:problem_size]
    z = 5 * (solution - shift_data) / 100
    z = dot(z, matrix)
    return z


def F13(solution , shift_data , matrix):
    #   "F13":"Shifted and Rotated HappyCat Function"
    problem_size = len(solution)
    shift_data = shift_data[:problem_size]
    z = 5 * (solution - shift_data) / 100
    z = dot(z, matrix)
    return z


def F14(solution , shift_data , matrix):
    #   "F14":"Shifted and Rotated HGBat Function"
    problem_size = len(solution)
    shift_data = shift_data[:problem_size]
    z = 5 * (solution - shift_data) / 100
    z = dot(z, matrix)
    return z


def F15(solution , shift_data , matrix):
    #   "F15":"Shifted and Rotated Expanded Griewank’s plus Rosenbrock’s Function"
    problem_size = len(solution)
    shift_data = shift_data[:problem_size]
    z = 5 * (solution - shift_data) / 100
    z = dot(z, matrix) + 1
    return z


def F16(solution , shift_data , matrix):
    #   "F16":"Shifted and Rotated Expanded Scaffer’s F6 Function"
    problem_size = len(solution)
    shift_data = shift_data[:problem_size]
    z = dot(solution - shift_data, matrix) + 1
    return z


### ================== Hybrid function ========================

def F17(solution=None, shift_data=None, matrix=None,f_bias=None):
    #   "F17":""Hybrid Function 1"
    problem_size = len(solution)
    p = array([0.3, 0.3, 0.4])
    n1 = int(ceil(p[0] * problem_size))
    n2 = int(ceil(p[1] * problem_size))
    shift_data = shift_data[:problem_size]
    shuffle = (shift_data - ones(problem_size)).astype(int)
    idx1 = shuffle[:n1]
    idx2 = shuffle[n1:(n1+n2)]
    idx3 = shuffle[(n1+n2):]
    mz = dot(solution - shift_data, matrix)
    return f9_modified_schwefel__(mz[idx1]) + f8_rastrigin__(mz[idx2]) + f1_elliptic__(mz[idx3]) + bias


def F18(solution=None, shift_data=None, matrix=None,f_bias=None):
    #   "F18":""Hybrid Function 2"
    problem_size = len(solution)
    shift_data = shift_data[:problem_size]
    p = array([0.3, 0.3, 0.4])
    n1 = int(ceil(p[0] * problem_size))
    n2 = int(ceil(p[1] * problem_size))

    shuffle = ( shift_data - ones(problem_size)).astype(int)
    idx1 = shuffle[:n1]
    idx2 = shuffle[n1:(n1 + n2)]
    idx3 = shuffle[(n1 + n2):]
    mz = dot(solution - shift_data, matrix)
    return f2_bent_cigar__(mz[idx1]) + f12_hgbat__(mz[idx2]) + f8_rastrigin__(mz[idx3]) + bias


def F19(solution=None, shift_data=None, matrix=None,f_bias=None):
    #   "F19":""Hybrid Function 3"
    problem_size = len(solution)
    shift_data = shift_data[:problem_size]
    p = array([0.2, 0.2, 0.3, 0.3])
    n1 = int(ceil(p[0] * problem_size))
    n2 = int(ceil(p[1] * problem_size))
    n3 = int(ceil(p[2] * problem_size))

    shuffle = (shift_data  - ones(problem_size)).astype(int)
    idx1 = shuffle[:n1]
    idx2 = shuffle[n1:(n1 + n2)]
    idx3 = shuffle[(n1 + n2):(n1+n2+n3)]
    idx4 = shuffle[n1+n2+n3:]
    mz = dot(solution - shift_data, matrix)
    return f7_griewank__(mz[idx1]) + f6_weierstrass__(mz[idx2]) + f4_rosenbrock__(mz[idx3]) + f14_expanded_scaffer__(mz[idx4])+ bias


def F20(solution=None, shift_data=None, matrix=None,f_bias=None):
    #   "F20":""Hybrid Function 4"
    problem_size = len(solution)
    shift_data = shift_data[:problem_size]
    p = array([0.2, 0.2, 0.3, 0.3])
    n1 = int(ceil(p[0] * problem_size))
    n2 = int(ceil(p[1] * problem_size))
    n3 = int(ceil(p[2] * problem_size))


    shuffle = ( shift_data - ones(problem_size)).astype(int)
    idx1 = shuffle[:n1]
    idx2 = shuffle[n1:(n1 + n2)]
    idx3 = shuffle[(n1 + n2):(n1 + n2 + n3)]
    idx4 = shuffle[n1 + n2 + n3:]
    mz = dot(solution - shift_data, matrix)
    return f12_hgbat__(mz[idx1]) + f3_discus__(mz[idx2]) + f13_expanded_griewank__(mz[idx3]) + f8_rastrigin__(mz[idx4]) + bias


def F21(solution=None, shift_data=None, matrix=None,bias=None):
    #   "F21":""Hybrid Function 5"
    problem_size = len(solution)
    shift_data = shift_data[:problem_size]
    p = array([0.1, 0.2, 0.2, 0.2, 0.3])
    n1 = int(ceil(p[0] * problem_size))
    n2 = int(ceil(p[1] * problem_size))
    n3 = int(ceil(p[2] * problem_size))
    n4 = int(ceil(p[3] * problem_size))


    shuffle = ( shift_data - ones(problem_size)).astype(int)
    idx1 = shuffle[:n1]
    idx2 = shuffle[n1:(n1 + n2)]
    idx3 = shuffle[(n1 + n2):(n1 + n2 + n3)]
    idx4 = shuffle[(n1+n2+n3):(n1+n2+n3+n4)]
    idx5 = shuffle[n1+n2+n3+n4:]
    mz = dot(solution - shift_data, matrix)
    return f14_expanded_scaffer__(mz[idx1]) + f12_hgbat__(mz[idx2]) + f4_rosenbrock__(mz[idx3]) + \
           f9_modified_schwefel__(mz[idx4]) + f1_elliptic__(mz[idx5]) + bias


def F22(solution=None, shift_data=None, matrix=None,bias=None):
    #   "F22":""Hybrid Function 6"
    problem_size = len(solution)
    shift_data = shift_data[:problem_size]
    p = array([0.1, 0.2, 0.2, 0.2, 0.3])
    n1 = int(ceil(p[0] * problem_size))
    n2 = int(ceil(p[1] * problem_size))
    n3 = int(ceil(p[2] * problem_size))
    n4 = int(ceil(p[3] * problem_size))


    shuffle = (shift_data - ones(problem_size)).astype(int)
    idx1 = shuffle[:n1]
    idx2 = shuffle[n1:(n1 + n2)]
    idx3 = shuffle[(n1 + n2):(n1 + n2 + n3)]
    idx4 = shuffle[(n1 + n2 + n3):(n1 + n2 + n3 + n4)]
    idx5 = shuffle[n1 + n2 + n3 + n4:]
    mz = dot(solution - shift_data, matrix)
    return f10_katsuura__(mz[idx1]) + f11_happy_cat__(mz[idx2]) + f13_expanded_griewank__(mz[idx3]) + \
           f9_modified_schwefel__(mz[idx4]) + f5_ackley__(mz[idx5]) + bias

### ================== Composition function ========================

def F23(solution=None, shift_data=None, matrix=None,f_bias=None):
    #   "F23":"Composition Function 1"
    problem_size = len(solution)
    shift_data = shift_data[:problem_size]
    shift_data = shift_data[:, :problem_size]
    xichma = array([10, 20, 30, 40, 50])
    lamda = array([1, 1e-6, 1e-26, 1e-6, 1e-6])
    bias = array([0, 100, 200, 300, 400])

    # 1. Rotated Rosenbrock’s Function F4’
    t1 = solution - shift_data[0]
    g1 = lamda[0] * f4_rosenbrock__(dot(t1, matrix[:problem_size, :])) + bias[0]
    w1 = (1.0 / sqrt(sum(t1 ** 2))) * exp(-sum(t1 ** 2) / (2 * problem_size * xichma[0] ** 2))

    # 2. High Conditioned Elliptic Function F1’
    t2 = solution - shift_data[1]
    g2 = lamda[1] * f1_elliptic__(solution) + bias[1]
    w2 = (1.0 / sqrt(sum(t2 ** 2))) * exp(-sum(t2 ** 2) / (2 * problem_size * xichma[1] ** 2))

    # 3. Rotated Bent Cigar Function F2’
    t3 = solution - shift_data[2]
    g3 = lamda[2] * f2_bent_cigar__(dot(matrix[2 * problem_size: 3 * problem_size, :], t3)) + bias[2]
    w3 = (1.0 / sqrt(sum(t3 ** 2))) * exp(-sum(t3 ** 2) / (2 * problem_size * xichma[2] ** 2))

    # 4. Rotated Discus Function F3’
    t4 = solution - shift_data[3]
    g4 = lamda[3] * f3_discus__(dot(matrix[3 * problem_size: 4 * problem_size, :], t4)) + bias[3]
    w4 = (1.0 / sqrt(sum(t4 ** 2))) * exp(-sum(t4 ** 2) / (2 * problem_size * xichma[3] ** 2))

    # 4. High Conditioned Elliptic Function F1’
    t5 = solution - shift_data[4]
    g5 = lamda[4] * f1_elliptic__(solution) + bias[4]
    w5 = (1.0 / sqrt(sum(t5 ** 2))) * exp(-sum(t5 ** 2) / (2 * problem_size * xichma[4] ** 2))

    sw = sum([w1, w2, w3, w4, w5])
    result = (w1 * g1 + w2 * g2 + w3 * g3 + w4 * g4 + w5 * g5) / sw
    return result + f_bias


def F24(solution=None, shift_data=None, matrix=None,f_bias=None):
    #   "F24":"Composition Function 2"
    problem_size = len(solution)
    shift_data = shift_data[:problem_size]
    shift_data = shift_data[:, :problem_size]
    xichma = array([20, 20, 20])
    lamda = array([1, 1, 1])
    bias = array([0, 100, 200])

    # 1. Rotated Rosenbrock’s Function F4’
    t1 = solution - shift_data[0]
    g1 = lamda[0] * f9_modified_schwefel__(solution) + bias[0]
    w1 = (1.0 / sqrt(sum(t1 ** 2))) * exp(-sum(t1 ** 2) / (2 * problem_size * xichma[0] ** 2))

    # 2. Rotated Rastrigin’s Function F9’
    t2 = solution - shift_data[1]
    g2 = lamda[1] * f8_rastrigin__(dot(matrix[problem_size: 2 * problem_size], t2)) + bias[1]
    w2 = (1.0 / sqrt(sum(t2 ** 2))) * exp(-sum(t2 ** 2) / (2 * problem_size * xichma[1] ** 2))

    # 3. Rotated HGBat Function F14’
    t3 = solution - shift_data[2]
    g3 = lamda[2] * f12_hgbat__(dot(matrix[2 * problem_size: 3 * problem_size, :], t3)) + bias[2]
    w3 = (1.0 / sqrt(sum(t3 ** 2))) * exp(-sum(t3 ** 2) / (2 * problem_size * xichma[2] ** 2))

    sw = sum([w1, w2, w3])
    result = (w1 * g1 + w2 * g2 + w3 * g3) / sw
    return result + f_bias


def F25(solution=None, shift_data=None, matrix=None,f_bias=None):
    #   "F25":"Composition Function 3"
    problem_size = len(solution)
    shift_data = shift_data[:problem_size]
    shift_data = shift_data[:, :problem_size]
    xichma = array([10, 30, 50])
    lamda = array([0.25, 1, 1e-7])
    bias = array([0, 100, 200])

    # 1. Rotated Schwefel's Function F11’
    t1 = solution - shift_data[0]
    g1 = lamda[0] * f9_modified_schwefel__(dot(matrix[:problem_size, :problem_size], t1)) + bias[0]
    w1 = (1.0 / sqrt(sum(t1 ** 2))) * exp(-sum(t1 ** 2) / (2 * problem_size * xichma[0] ** 2))

    # 2. Rotated Rastrigin’s Function F9’
    t2 = solution - shift_data[1]
    g2 = lamda[1] * f8_rastrigin__(dot(matrix[problem_size: 2 * problem_size], t2)) + bias[1]
    w2 = (1.0 / sqrt(sum(t2 ** 2))) * exp(-sum(t2 ** 2) / (2 * problem_size * xichma[1] ** 2))

    # 3. Rotated High Conditioned Elliptic Function F1'
    t3 = solution - shift_data[2]
    g3 = lamda[2] * f1_elliptic__(dot(matrix[2 * problem_size: 3 * problem_size, :], t3)) + bias[2]
    w3 = (1.0 / sqrt(sum(t3 ** 2))) * exp(-sum(t3 ** 2) / (2 * problem_size * xichma[2] ** 2))

    sw = sum([w1, w2, w3])
    result = (w1 * g1 + w2 * g2 + w3 * g3) / sw
    return result + f_bias


def F26(solution=None, shift_data=None, matrix=None,f_bias=None):
    #   "F26":"Composition Function 4"
    problem_size = len(solution)
    shift_data = shift_data[:problem_size]
    shift_data = shift_data[:, :problem_size]
    xichma = array([10, 10, 10, 10, 10])
    lamda = array([0.25, 1, 1e-7, 2.5, 10])
    bias = array([0, 100, 200, 300, 400])

    # 1. Rotated Schwefel's Function F11’
    t1 = solution - shift_data[0]
    g1 = lamda[0] * f9_modified_schwefel__(dot(matrix[:problem_size, :], t1)) + bias[0]
    w1 = (1.0 / sqrt(sum(t1 ** 2))) * exp(-sum(t1 ** 2) / (2 * problem_size * xichma[0] ** 2))

    # 2. Rotated HappyCat Function F13’
    t2 = solution - shift_data[1]
    g2 = lamda[1] * f11_happy_cat__(dot(matrix[problem_size:2 * problem_size, :], t2)) + bias[1]
    w2 = (1.0 / sqrt(sum(t2 ** 2))) * exp(-sum(t2 ** 2) / (2 * problem_size * xichma[1] ** 2))

    # 3. Rotated High Conditioned Elliptic Function F1’
    t3 = solution - shift_data[2]
    g3 = lamda[2] * f1_elliptic__(dot(matrix[2 * problem_size: 3 * problem_size, :], t3)) + bias[2]
    w3 = (1.0 / sqrt(sum(t3 ** 2))) * exp(-sum(t3 ** 2) / (2 * problem_size * xichma[2] ** 2))

    # 4. Rotated Weierstrass Function F6’
    t4 = solution - shift_data[3]
    g4 = lamda[3] * f6_weierstrass__(dot(matrix[3 * problem_size: 4 * problem_size, :], t4)) + bias[3]
    w4 = (1.0 / sqrt(sum(t4 ** 2))) * exp(-sum(t4 ** 2) / (2 * problem_size * xichma[3] ** 2))

    # 5. Rotated Griewank’s Function F7’
    t5 = solution - shift_data[4]
    g5 = lamda[4] * f7_griewank__(dot(matrix[4*problem_size:, :], t5)) + bias[4]
    w5 = (1.0 / sqrt(sum(t5 ** 2))) * exp(-sum(t5 ** 2) / (2 * problem_size * xichma[4] ** 2))

    sw = sum([w1, w2, w3, w4, w5])
    result = (w1 * g1 + w2 * g2 + w3 * g3 + w4 * g4 + w5 * g5) / sw
    return result + f_bias


def F27(solution=None, shift_data=None, matrix=None,f_bias=None):
    #   "F27":"Composition Function 5"
    problem_size = len(solution)
    shift_data = shift_data[:problem_size]
    shift_data = shift_data[:, :problem_size]
    xichma = array([10, 10, 10, 20, 20])
    lamda = array([10, 10, 2.5, 25, 1e-6])
    bias = array([0, 100, 200, 300, 400])

    # 1. Rotated HGBat Function F14'
    t1 = solution - shift_data[0]
    g1 = lamda[0] * f12_hgbat__(dot(matrix[:problem_size, :], t1)) + bias[0]
    w1 = (1.0 / sqrt(sum(t1 ** 2))) * exp(-sum(t1 ** 2) / (2 * problem_size * xichma[0] ** 2))

    # 2. Rotated Rastrigin’s Function F9’
    t2 = solution - shift_data[1]
    g2 = lamda[1] * f8_rastrigin__(dot(matrix[problem_size:2 * problem_size, :], t2)) + bias[1]
    w2 = (1.0 / sqrt(sum(t2 ** 2))) * exp(-sum(t2 ** 2) / (2 * problem_size * xichma[1] ** 2))

    # 3. Rotated Schwefel's Function F11’
    t3 = solution - shift_data[2]
    g3 = lamda[2] * f9_modified_schwefel__(dot(matrix[2 * problem_size: 3 * problem_size, :], t3)) + bias[2]
    w3 = (1.0 / sqrt(sum(t3 ** 2))) * exp(-sum(t3 ** 2) / (2 * problem_size * xichma[2] ** 2))

    # 4. Rotated Weierstrass Function F6’
    t4 = solution - shift_data[3]
    g4 = lamda[3] * f6_weierstrass__(dot(matrix[3 * problem_size: 4 * problem_size, :], t4)) + bias[3]
    w4 = (1.0 / sqrt(sum(t4 ** 2))) * exp(-sum(t4 ** 2) / (2 * problem_size * xichma[3] ** 2))

    # 5. Rotated High Conditioned Elliptic Function F1’
    t5 = solution - shift_data[4]
    g5 = lamda[4] * f1_elliptic__(dot(matrix[4 * problem_size:, :], t5)) + bias[4]
    w5 = (1.0 / sqrt(sum(t5 ** 2))) * exp(-sum(t5 ** 2) / (2 * problem_size * xichma[4] ** 2))

    sw = sum([w1, w2, w3, w4, w5])
    result = (w1 * g1 + w2 * g2 + w3 * g3 + w4 * g4 + w5 * g5) / sw
    return result + f_bias

'''
def F28(solution=None, shift_data, matrix,f_bias):
    #   "F28":"Composition Function 6"
    problem_size = len(solution)
    shift_data = shift_data[:problem_size]
    shift_data = shift_data[:, :problem_size]
    xichma = array([10, 20, 30, 40, 50])
    lamda = array([2.5, 10, 2.5, 5e-4, 1e-6])
    bias = array([0, 100, 200, 300, 400])
    # 1. Rotated Expanded Griewank’s plus Rosenbrock’s Function F15’
    t1 = solution - shift_data[0]
    g1 = lamda[0] * F15(solution, bias=0) + bias[0]
    w1 = (1.0 / sqrt(sum(t1 ** 2))) * exp(-sum(t1 ** 2) / (2 * problem_size * xichma[0] ** 2))
    # 2. Rotated HappyCat Function F13’
    t2 = solution - shift_data[1]
    g2 = lamda[1] * f11_happy_cat__(dot(matrix[problem_size:2 * problem_size, :], t2)) + bias[1]
    w2 = (1.0 / sqrt(sum(t2 ** 2))) * exp(-sum(t2 ** 2) / (2 * problem_size * xichma[1] ** 2))
    # 3. Rotated Schwefel's Function F11’
    t3 = solution - shift_data[2]
    g3 = lamda[2] * f9_modified_schwefel__(dot(matrix[2 * problem_size: 3 * problem_size, :], t3)) + bias[2]
    w3 = (1.0 / sqrt(sum(t3 ** 2))) * exp(-sum(t3 ** 2) / (2 * problem_size * xichma[2] ** 2))
    # 4. Rotated Expanded Scaffer’s F6 Function F16’
    t4 = solution - shift_data[3]
    g4 = lamda[3] * f14_expanded_scaffer__(dot(matrix[3 * problem_size: 4 * problem_size, :], t4)) + bias[3]
    w4 = (1.0 / sqrt(sum(t4 ** 2))) * exp(-sum(t4 ** 2) / (2 * problem_size * xichma[3] ** 2))
    # 5. Rotated High Conditioned Elliptic Function F1’
    t5 = solution - shift_data[4]
    g5 = lamda[4] * f1_elliptic__(dot(matrix[4 * problem_size:, :], t5)) + bias[4]
    w5 = (1.0 / sqrt(sum(t5 ** 2))) * exp(-sum(t5 ** 2) / (2 * problem_size * xichma[4] ** 2))
    sw = sum([w1, w2, w3, w4, w5])
    result = (w1 * g1 + w2 * g2 + w3 * g3 + w4 * g4 + w5 * g5) / sw
    return result + f_bias
def F17_aux()#solution=None, name="Hybrid Function 1", shift_data_file="shift_data_17.txt", bias=1700, shuffle=None):
    problem_size = len(solution)
    p = array([0.3, 0.3, 0.4])
    n1 = int(ceil(p[0] * problem_size))
    n2 = int(ceil(p[1] * problem_size))
    if problem_size > 100:
        print("CEC 2014 not support for problem size > 100")
        return 1
    if problem_size in SUPPORT_DIMENSION_2:
        f_matrix = "M_17_D" + str(problem_size) + ".txt"
        if shuffle is None:
            f_shuffle = "shuffle_data_17_D" + str(problem_size) + ".txt"
        else:
            f_shuffle = "shuffle_data_" + str(shuffle) + "_D" + str(problem_size) + ".txt"
    else:
        print("CEC 2014 function only support problem size 10, 20, 30, 50, 100")
        return 1
    shift_data = load_shift_data__(shift_data_file)[:problem_size]
    matrix = load_matrix_data__(f_matrix)
    shuffle = (load_shift_data__(f_shuffle)[:problem_size] - ones(problem_size)).astype(int)
    idx1 = shuffle[:n1]
    idx2 = shuffle[n1:(n1+n2)]
    idx3 = shuffle[(n1+n2):]
    mz = dot(solution - shift_data, matrix)
    return f9_modified_schwefel__(mz[idx1]) + f8_rastrigin__(mz[idx2]) + f1_elliptic__(mz[idx3]) + bias
def F18_aux()#solution=None, name="Hybrid Function 2", shift_data_file="shift_data_18.txt", bias=1800, shuffle=None):
    problem_size = len(solution)
    p = array([0.3, 0.3, 0.4])
    n1 = int(ceil(p[0] * problem_size))
    n2 = int(ceil(p[1] * problem_size))
    if problem_size > 100:
        print("CEC 2014 not support for problem size > 100")
        return 1
    if problem_size in SUPPORT_DIMENSION_2:
        f_matrix = "M_18_D" + str(problem_size) + ".txt"
        if shuffle is None:
            f_shuffle = "shuffle_data_18_D" + str(problem_size) + ".txt"
        else:
            f_shuffle = "shuffle_data_" + str(shuffle) + "_D" + str(problem_size) + ".txt"
    else:
        print("CEC 2014 function only support problem size 10, 20, 30, 50, 100")
        return 1
    shift_data = load_shift_data__(shift_data_file)[:problem_size]
    matrix = load_matrix_data__(f_matrix)
    shuffle = (load_shift_data__(f_shuffle)[:problem_size] - ones(problem_size)).astype(int)
    idx1 = shuffle[:n1]
    idx2 = shuffle[n1:(n1 + n2)]
    idx3 = shuffle[(n1 + n2):]
    mz = dot(solution - shift_data, matrix)
    return f2_bent_cigar__(mz[idx1]) + f12_hgbat__(mz[idx2]) + f8_rastrigin__(mz[idx3]) + bias
def F19_aux()#solution=None, name="Hybrid Function 3", shift_data_file="shift_data_19.txt", bias=1900, shuffle=None):
    problem_size = len(solution)
    p = array([0.2, 0.2, 0.3, 0.3])
    n1 = int(ceil(p[0] * problem_size))
    n2 = int(ceil(p[1] * problem_size))
    n3 = int(ceil(p[2] * problem_size))
    if problem_size > 100:
        print("CEC 2014 not support for problem size > 100")
        return 1
    if problem_size in SUPPORT_DIMENSION_2:
        f_matrix = "M_19_D" + str(problem_size) + ".txt"
        if shuffle is None:
            f_shuffle = "shuffle_data_19_D" + str(problem_size) + ".txt"
        else:
            f_shuffle = "shuffle_data_" + str(shuffle) + "_D" + str(problem_size) + ".txt"
    else:
        print("CEC 2014 function only support problem size 10, 20, 30, 50, 100")
        return 1
    shift_data = load_shift_data__(shift_data_file)[:problem_size]
    matrix = load_matrix_data__(f_matrix)
    shuffle = (load_shift_data__(f_shuffle)[:problem_size] - ones(problem_size)).astype(int)
    idx1 = shuffle[:n1]
    idx2 = shuffle[n1:(n1 + n2)]
    idx3 = shuffle[(n1 + n2):(n1+n2+n3)]
    idx4 = shuffle[n1+n2+n3:]
    mz = dot(solution - shift_data, matrix)
    return f7_griewank__(mz[idx1]) + f6_weierstrass__(mz[idx2]) + f4_rosenbrock__(mz[idx3]) + f14_expanded_scaffer__(mz[idx4])+ bias
def F29(solution=None, shift_data, matrix,f_bias):
    num_funcs = 3
    #   "F29":"Composition Function 7"
    problem_size = len(solution)
    shift_data = shift_data[:problem_size]
    shift_data = shift_data[:, :problem_size]
    problem_size = len(solution)
    xichma = array([10, 30, 50])
    lamda = array([1, 1, 1])
    bias = array([0, 100, 200])
    
    def __fi__(solution=None, idx=None):
        if idx == 0:
            return F17_aux()#solution, bias=0, shuffle=29)
        elif idx == 1:
            return F18_aux()#solution, bias=0, shuffle=29)
        else:
            return F19_aux()#solution, bias=0, shuffle=29)
    weights = ones(num_funcs)
    fits = ones(num_funcs)
    for i in range(0, num_funcs):
        t1 = lamda[i] * __fi__(solution, i) + bias[i]
        t2 = 1.0 / sqrt(sum((solution - shift_data[i]) ** 2))
        w_i = t2 * exp(-sum((solution - shift_data[i]) ** 2) / (2 * problem_size * xichma[i] ** 2))
        weights[i] = w_i
        fits[i] = t1
    sw = sum(weights)
    result = 0.0
    for i in range(0, num_funcs):
        result += (weights[i] / sw) * fits[i]
    return result + f_bias
def F20_aux()#solution=None, name="Hybrid Function 4",  shift_data, matrix,f_bias, bias=2000, shuffle=None):
    problem_size = len(solution)
    p = array([0.2, 0.2, 0.3, 0.3])
    n1 = int(ceil(p[0] * problem_size))
    n2 = int(ceil(p[1] * problem_size))
    n3 = int(ceil(p[2] * problem_size))
    if problem_size > 100:
        print("CEC 2014 not support for problem size > 100")
        return 1
    if problem_size in SUPPORT_DIMENSION_2:
        f_matrix = "M_20_D" + str(problem_size) + ".txt"
        if shuffle is None:
            f_shuffle = "shuffle_data_20_D" + str(problem_size) + ".txt"
        else:
            f_shuffle = "shuffle_data_" + str(shuffle) + "_D" + str(problem_size) + ".txt"
    else:
        print("CEC 2014 function only support problem size 10, 20, 30, 50, 100")
        return 1
    shift_data = load_shift_data__(shift_data_file)[:problem_size]
    matrix = load_matrix_data__(f_matrix)
    shuffle = (load_shift_data__(f_shuffle)[:problem_size] - ones(problem_size)).astype(int)
    idx1 = shuffle[:n1]
    idx2 = shuffle[n1:(n1 + n2)]
    idx3 = shuffle[(n1 + n2):(n1 + n2 + n3)]
    idx4 = shuffle[n1 + n2 + n3:]
    mz = dot(solution - shift_data, matrix)
    return f12_hgbat__(mz[idx1]) + f3_discus__(mz[idx2]) + f13_expanded_griewank__(mz[idx3]) + f8_rastrigin__(mz[idx4]) + bias
def F21_aux()#solution=None, name="Hybrid Function 5", shift_data_file="shift_data_21.txt", bias=2100, shuffle=None):
    problem_size = len(solution)
    p = array([0.1, 0.2, 0.2, 0.2, 0.3])
    n1 = int(ceil(p[0] * problem_size))
    n2 = int(ceil(p[1] * problem_size))
    n3 = int(ceil(p[2] * problem_size))
    n4 = int(ceil(p[3] * problem_size))
    if problem_size > 100:
        print("CEC 2014 not support for problem size > 100")
        return 1
    if problem_size in SUPPORT_DIMENSION_2:
        f_matrix = "M_21_D" + str(problem_size) + ".txt"
        if shuffle is None:
            f_shuffle = "shuffle_data_21_D" + str(problem_size) + ".txt"
        else:
            f_shuffle = "shuffle_data_" + str(shuffle) + "_D" + str(problem_size) + ".txt"
    else:
        print("CEC 2014 function only support problem size 10, 20, 30, 50, 100")
        return 1
    shift_data = load_shift_data__(shift_data_file)[:problem_size]
    matrix = load_matrix_data__(f_matrix)
    shuffle = (load_shift_data__(f_shuffle)[:problem_size] - ones(problem_size)).astype(int)
    idx1 = shuffle[:n1]
    idx2 = shuffle[n1:(n1 + n2)]
    idx3 = shuffle[(n1 + n2):(n1 + n2 + n3)]
    idx4 = shuffle[(n1+n2+n3):(n1+n2+n3+n4)]
    idx5 = shuffle[n1+n2+n3+n4:]
    mz = dot(solution - shift_data, matrix)
    return f14_expanded_scaffer__(mz[idx1]) + f12_hgbat__(mz[idx2]) + f4_rosenbrock__(mz[idx3]) + \
           f9_modified_schwefel__(mz[idx4]) + f1_elliptic__(mz[idx5]) + bias
def F22_aux(solution=None, name="Hybrid Function 6", shift_data_file="shift_data_22.txt", bias=2200, shuffle=None):
    problem_size = len(solution)
    p = array([0.1, 0.2, 0.2, 0.2, 0.3])
    n1 = int(ceil(p[0] * problem_size))
    n2 = int(ceil(p[1] * problem_size))
    n3 = int(ceil(p[2] * problem_size))
    n4 = int(ceil(p[3] * problem_size))
    if problem_size > 100:
        print("CEC 2014 not support for problem size > 100")
        return 1
    if problem_size in SUPPORT_DIMENSION_2:
        f_matrix = "M_22_D" + str(problem_size) + ".txt"
        if shuffle is None:
            f_shuffle = "shuffle_data_21_D" + str(problem_size) + ".txt"
        else:
            f_shuffle = "shuffle_data_" + str(shuffle) + "_D" + str(problem_size) + ".txt"
    else:
        print("CEC 2014 function only support problem size 10, 20, 30, 50, 100")
        return 1
    shift_data = load_shift_data__(shift_data_file)[:problem_size]
    matrix = load_matrix_data__(f_matrix)
    shuffle = (load_shift_data__(f_shuffle)[:problem_size] - ones(problem_size)).astype(int)
    idx1 = shuffle[:n1]
    idx2 = shuffle[n1:(n1 + n2)]
    idx3 = shuffle[(n1 + n2):(n1 + n2 + n3)]
    idx4 = shuffle[(n1 + n2 + n3):(n1 + n2 + n3 + n4)]
    idx5 = shuffle[n1 + n2 + n3 + n4:]
    mz = dot(solution - shift_data, matrix)
    return f10_katsuura__(mz[idx1]) + f11_happy_cat__(mz[idx2]) + f13_expanded_griewank__(mz[idx3]) + \
           f9_modified_schwefel__(mz[idx4]) + f5_ackley__(mz[idx5]) + 
def F30(solution=None, shift_data, matrix,f_bias):
    num_funcs = 3
    #   "F30":"Composition Function 8"
    problem_size = len(solution)
    shift_data = shift_data[:problem_size]
    shift_data = shift_data[:, :problem_size]
    xichma = array([10, 30, 50])
    lamda = array([1, 1, 1])
    bias = array([0, 100, 200])
    def __fi__(solution=None, idx=None):
        if idx == 0:
            return F20_aux()#solution, bias=0, shuffle=30)
        elif idx == 1:
            return F21_aux()#solution, bias=0, shuffle=30)
        else:
            return F22_aux()#solution, bias=0, shuffle=30)
    weights = ones(num_funcs)
    fits = ones(num_funcs)
    for i in range(0, num_funcs):
        t1 = lamda[i] * __fi__(solution, i) + bias[i]
        t2 = 1.0 / sqrt(sum((solution - shift_data[i]) ** 2))
        w_i = t2 * exp(-sum((solution - shift_data[i]) ** 2) / (2 * problem_size * xichma[i] ** 2))
        weights[i] = w_i
        fits[i] = t1
    sw = sum(weights)
    result = 0.0
    for i in range(0, num_funcs):
        result += (weights[i] / sw) * fits[i]
    return result + f_bias
'''