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

def f1_elliptic__(solution=None):
    result = 0
    for i in range(len(solution)):
        result += (10**6)**(i/(len(solution)-1)) * solution[i]**2
    return result

def f2_bent_cigar__(solution=None):
    return solution[0]**2 + 10**6 * sum(solution[1:]**2)

def f3_discus__(solution=None):
    return 10**6 * solution[0]**2 + sum(solution[1:]**2)

def f4_rosenbrock__(solution=None):
    result = 0.0
    for i in range(len(solution)-1):
        result += 100*(solution[i]**2 - solution[i+1])**2 + (solution[i] - 1)**2
    return result

def f5_ackley__(solution=None):
    return -20*exp(-0.2*sqrt(sum(solution**2)/len(solution))) - exp(sum(cos(2*pi*solution))/len(solution)) + 20 + e

def f6_weierstrass__(solution=None, a=0.5, b=3, k_max=20):
    result = 0.0
    for i in range(0, len(solution)):
        t1 = sum([a**k * cos(2*pi*b**k*(solution[i] + 0.5)) for k in range(0, k_max)])
        result += t1
    t2 = len(solution) * sum([a**k * cos(2*pi*b**k * 0.5) for k in range(0, k_max)])
    return result - t2

def f7_griewank__(solution=None):
    result = sum(solution ** 2) / 4000
    temp = 1.0
    for i in range(len(solution)):
        temp *= cos(solution[i] / sqrt(i + 1))
    return result - temp + 1

def f8_rastrigin__(solution=None):
    return sum(solution ** 2 - 10 * cos(2 * pi * solution) + 10)

def f9_modified_schwefel__(solution=None):
    
    z =1.0*solution
    for i in range(len(solution)):
        z[i] = z[i] + 4.209687462275036e+002
    
    #z = solution + 4.209687462275036e+002
    result = 418.9829 * len(solution)
    for i in range(0, len(solution)):
        if z[i] > 500:
            result -= (500 - z[i]%500)*sin(sqrt(abs(500 - z[i]%500))) - (z[i] - 500)**2 / (10000*len(solution))
        elif z[i] < -500:
            result -= (z[i]%500 - 500)*sin(sqrt(abs(z[i]%500 - 500))) - (z[i] + 500)**2 / (10000*len(solution))
        else:
            result -= z[i]*sin(abs(z[i])**0.5)
    return result

def f10_katsuura__(solution=None):
    result = 1.0
    for i in range(0, len(solution)):
        t1 = sum([abs(2 ** j * solution[i] - round(2 ** j * solution[i])) / 2 ** j for j in range(1, 33) ])
        result *= (1 + (i+1)*t1)**(10.0 / len(solution)**1.2)
    return (result - 1) * 10 / len(solution)**2

def f11_happy_cat__(solution=None):
    return (abs(sum(solution**2) - len(solution)))**0.25 + (0.5 * sum(solution**2) + sum(solution))/len(solution) + 0.5

def f12_hgbat__(solution=None):
    return (abs(sum(solution**2)**2 - sum(solution)**2))**0.5 + (0.5*sum(solution**2)+sum(solution))/len(solution) + 0.5

def f13_expanded_griewank__(solution=None):
    def __f4__(x=None, y=None):
        return 100*(x**2 - y)**2 + (x - 1)**2
    def __f7__(z=None):
        return z**2/4000 - cos(z/sqrt(1)) + 1

    result = __f7__(__f4__(solution[-1], solution[0]))
    for i in range(0, len(solution) - 1):
        result += __f7__(__f4__(solution[i], solution[i + 1]))
    return result

def f14_expanded_scaffer__(solution=None):
    def __xy__(x, y):
        return 0.5 + (sin(sqrt(x ** 2 + y ** 2)) ** 2 - 0.5) / (1 + 0.001 * (x ** 2 + y ** 2)) ** 2

    result = __xy__(solution[-1], solution[0])
    for i in range(0, len(solution) - 1):
        result += __xy__(solution[i], solution[i + 1])
    return result


def F1(solution):# , shift_data , matrix ,f_bias ):
    #   "F1":"Rotated High Conditioned Elliptic Function"
    return f1_elliptic__(solution)# + bias


def F2(solution):
    #   "F2":"Rotated Bent Cigar Function"
    return f2_bent_cigar__(solution)


def F3(solution):
    #   "F3":"Rotated Discus Function"
    return f3_discus__(solution)


def F4(solution):
    #   "F4":"Shifted and Rotated Rosenbrock’s Function"
    return f4_rosenbrock__(solution)


def F5(solution):
    #   "F5":"Shifted and Rotated Ackley’s Function"
    return f5_ackley__(solution)


def F6(solution):
    #   "F6":"Shifted and Rotated Weierstrass Function"
    return f6_weierstrass__(solution)


def F7(solution):
    #   "F7":"Shifted and Rotated Griewank’s Function"
    return f7_griewank__(solution)


def F8(solution):
    #   "F8":"Shifted Rastrigin’s Function"
    return f8_rastrigin__(solution)


def F9(solution):
    #   "F9":"Shifted and Rotated Rastrigin’s Function"
    return f9_modified_schwefel__(solution)


def F10(solution):
    #   "F10":"Shifted Schwefel’s Function"
    return f9_modified_schwefel__(solution)


def F11(solution):
    #   "F11":"Shifted and Rotated Schwefel’s Function"
    return f9_modified_schwefel__(solution)


def F12(solution):
    #   "F12":"Shifted and Rotated Katsuura Function"
    return f10_katsuura__(solution)


def F13(solution):
    #   "F13":"Shifted and Rotated HappyCat Function"
    return f11_happy_cat__(solution)


def F14(solution):
    #   "F14":"Shifted and Rotated HGBat Function"
    return f12_hgbat__(solution)


def F15(solution):
    #   "F15":"Shifted and Rotated Expanded Griewank’s plus Rosenbrock’s Function"
    return f13_expanded_griewank__(solution)


def F16(solution):
    #   "F16":"Shifted and Rotated Expanded Scaffer’s F6 Function"
    return f14_expanded_scaffer__(solution)


### ================== Hybrid function ========================

def F17(solution):
    #   "F17":""Hybrid Function 1"
    return solution

def F18(solution):
    #   "F18":""Hybrid Function 2"
    return solution


def F19(solution):
    #   "F19":""Hybrid Function 3"
    return solution


def F20(solution):
    #   "F20":""Hybrid Function 4"
    return solution


def F21(solution):
    #   "F21":""Hybrid Function 5"
    return solution
           


def F22(solution):
    #   "F22":""Hybrid Function 6"
    return solution

### ================== Composition function ========================

def F23(solution):
    #   "F23":"Composition Function 1"
    return solution


def F24(solution):
    #   "F24":"Composition Function 2"
    return solution


def F25(solution):
    #   "F25":"Composition Function 3"
    return solution


def F26(solution):
    return solution


def F27(solution):
    #   "F27":"Composition Function 5"
    return solution

def F15_hybrid(solution , shift_data , matrix,shuffle):
    #   "F15":"Shifted and Rotated Expanded Griewank’s plus Rosenbrock’s Function"
    z = 5 * (solution - shift_data) / 100
    z = dot(z, matrix) + 1
    return f13_expanded_griewank__(z) 

def F28(solution , shift_data , matrix,shuffle):
    problem_size = len(solution)
    xichma = array([10, 20, 30, 40, 50])
    lamda = array([2.5, 10, 2.5, 5e-4, 1e-6])
    bias = array([0, 100, 200, 300, 400])

    # 1. Rotated Expanded Griewank’s plus Rosenbrock’s Function F15’
    t1 = solution - shift_data[0]
    g1 = lamda[0] * F15_hybrid(solution , shift_data , matrix,shuffle) + bias[0]
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
    return result #+ f_bias


def F29(solution=None, name="Composition Function 7", shift_data_file="shift_data_29.txt", f_bias=2900):
    num_funcs = 3
    problem_size = len(solution)
    xichma = array([10, 30, 50])
    lamda = array([1, 1, 1])
    bias = array([0, 100, 200])

    if problem_size > 100:
        print("CEC 2014 not support for problem size > 100")
        return 1
    shift_data = load_matrix_data__(shift_data_file)[:problem_size]
    shift_data = shift_data[:, :problem_size]

    def __fi__(solution=None, idx=None):
        if idx == 0:
            return F17(solution, bias=0, shuffle=29)
        elif idx == 1:
            return F18(solution, bias=0, shuffle=29)
        else:
            return F19(solution, bias=0, shuffle=29)

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
def F28():
# def F28(solution=None, shift_data=None, matrix=None,f_bias=None):
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

def F29():
# def F29(solution=None, shift_data=None, matrix=None,f_bias=None):
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

def F30(solution=None, shift_data=None, matrix=None,f_bias=None):
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
