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
    z = solution + 4.209687462275036e+002
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


### ================== Hybrid function ========================

def F17(solution, shift_data, matrix):
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
    result = f9_modified_schwefel__(mz[idx1]) + f8_rastrigin__(mz[idx2]) + f1_elliptic__(mz[idx3])
    
    return result


def F18(solution, shift_data, matrix):
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
    result = f2_bent_cigar__(mz[idx1]) + f12_hgbat__(mz[idx2]) + f8_rastrigin__(mz[idx3])
    return result 


def F19(solution, shift_data, matrix):
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
    result = f7_griewank__(mz[idx1]) + f6_weierstrass__(mz[idx2]) + f4_rosenbrock__(mz[idx3]) + f14_expanded_scaffer__(mz[idx4])
    return result


def F20(solution, shift_data, matrix):
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
    resultf = f12_hgbat__(mz[idx1]) + f3_discus__(mz[idx2]) + f13_expanded_griewank__(mz[idx3]) + f8_rastrigin__(mz[idx4])
    return  result


def F21(solution, shift_data, matrix):
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
    result = f14_expanded_scaffer__(mz[idx1]) + f12_hgbat__(mz[idx2]) + f4_rosenbrock__(mz[idx3]) 
    result = result + f9_modified_schwefel__(mz[idx4]) + f1_elliptic__(mz[idx5]) 
    return  result


def F22(solution, shift_data, matrix):
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
    result = f10_katsuura__(mz[idx1]) + f11_happy_cat__(mz[idx2]) + f13_expanded_griewank__(mz[idx3])
    result = result + f9_modified_schwefel__(mz[idx4]) + f5_ackley__(mz[idx5])
    return  result

