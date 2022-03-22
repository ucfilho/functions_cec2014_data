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

#def F28(solution=None, shift_data=None, matrix=None,f_bias=None):
def F28(solution , shift_data , matrix,sda_01, ma_01, Fba_01):
    problem_size = len(solution)
    xichma = array([10, 20, 30, 40, 50])
    lamda = array([2.5, 10, 2.5, 5e-4, 1e-6])
    bias = array([0, 100, 200, 300, 400])
    
    '''
    if problem_size in SUPPORT_DIMENSION_2:
        f_matrix = f_matrix + str(problem_size) + ".txt"
    else:
        print("CEC 2014 function only support problem size 10, 20, 30, 50, 100")
        return 1
    shift_data = load_matrix_data__(f_shift_file)[:problem_size]
    shift_data = shift_data[:, :problem_size]
    matrix = load_matrix_data__(f_matrix)
    '''
    ##################################################
    ###########                            ###########
    ##################################################
    
    def F15(solution, sda_01, ma_01):
        problem_size = len(solution)
        bias = 0 # bias in composed is zero not 1500

        shift_data = sda_01[:problem_size]
        matrix = ma_01
        z = 5 * (solution - shift_data) / 100
        z = dot(z, matrix) + 1
        result = f13_expanded_griewank__(z) + bias
        return result

    # 1. Rotated Expanded Griewank’s plus Rosenbrock’s Function F15’
    t1 = solution - shift_data[0]
    g1 = lamda[0] * F15(solution, sda_01, ma_01) + bias[0]
    ##################################################
    ###########                            ###########
    ##################################################
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
    
    return result 
