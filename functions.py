from math import ceil, sqrt
import random
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial
from scipy.interpolate import make_interp_spline

#Georgios Mpalanos

function_calls = 0
gradient_calls = 0

seed_value = 339
np.random.seed(seed_value)

def randomPointGen(l,r):
    return [l+np.random.rand()*2*r for i in range(5)] #Generate numbers from l to r : l+2*r*rand() , rand() ~ (0,1)


###########################################

def pol(t,a):
    returnVal = 0
    for i in range(len(a)):
        returnVal += a[i]*pow(t,i)
    return returnVal

############################################

def f(a,y_t):
    global function_calls
    function_calls += 1
    returnVal = 0
    for i in range(len(y_t)):
        returnVal += pow(y_t[i] - pol(i+1,a),2)
    return returnVal/len(y_t)


############################################


def fpd(a,j,y_t):
    returnVal = 0
    for i in range(len(y_t)):
        returnVal += -2 * (y_t[i] - pol(i+1,a)) * pow(i+1,j)
    return returnVal/len(y_t)

def gradient(a,y_t):
    global gradient_calls
    gradient_calls += 1
    returnVal = [fpd(a,j,y_t) for j in range(len(a))]
    return returnVal

        
############################################


def line_search(xk,pk,y_t,c1=1e-4,c2=0.9,max_iterations = 200, min_step_size=1e-1):
    alpha = 0.5
    a = 0
    b = 1
    i = 0

    while(max_iterations > i and min_step_size < alpha):
        x_new = xk + alpha * pk
        fx_k = f(xk,y_t)
        gradientx_k = gradient(xk,y_t)

        fx_new = f(x_new, y_t)
        gradientx_new = gradient(x_new,y_t)

        if(fx_new > fx_k + c1 * alpha * np.dot(gradientx_k,pk)):
            b = alpha
            alpha = (a+b) * 0.5

        elif(abs(np.dot(gradientx_new,pk)) > c2 * abs(np.dot(gradientx_k,pk))):
            a = alpha
            if(b == 1):
                alpha = 2*a
            else:
                alpha = (a+b)*0.5
        else:
            break
        
        i+=1

    return max(min_step_size,alpha)


def BFGS(x0,epsilon,y_t,max_iterations=1000):
    k = 0
    xk = x0
    h = np.identity(len(xk))


    while(np.linalg.norm(gradient(xk,y_t),ord=2) > epsilon and max_iterations > k):
        pk = (-1) * h @ np.array(gradient(xk,y_t))

        ak = line_search(xk,pk,y_t)

        if(np.dot(pk,gradient(xk,y_t)) >= 0):
            print("Not negative:",np.dot(pk,gradient(xk,y_t)))

        xnew = xk + ak * np.array(pk)

        sk = xnew - xk    
        yk = np.subtract(gradient(xnew,y_t),gradient(xk,y_t))
        rk = 1/np.dot(yk,sk)

        A = np.identity(len(sk)) - rk*np.outer(sk,yk)
        B = np.identity(len(sk)) - rk*np.outer(yk,sk)
        h = A @ h @ B + rk*np.outer(sk,sk)
        xk = xnew

        k += 1

    print("BFGS iterations: ", k)

    global gradient_calls
    global function_calls

    print("BFGS Function Calls: ", function_calls) 
    print("BFGS Gradient Calls: ", gradient_calls)

    gradient_calls = 0
    function_calls = 0
    return xk

def solve_quadratic_equation(a,b,c): 
    d = b**2 - 4*a*c

    if d < 0:
        raise ValueError("No real solutions")

    sqrt_d = np.sqrt(d)
    x1 = (-b + sqrt_d) / (2*a)
    x2 = (-b - sqrt_d) / (2*a)

    return max(x1, x2)


def dogleg(gk, Bk, Hk, trust_region_radius):
    p_B = (-1)*np.array(Hk) @ np.array(gk)

    if(np.linalg.norm(p_B,ord=2) <= trust_region_radius):
        p_star = p_B
    
    else:
        p_U = -np.array(gk) * (np.dot(gk, gk) / np.dot(gk, Bk @ gk))

        if(np.linalg.norm(p_U) >= trust_region_radius):
            p_star = -(trust_region_radius/np.linalg.norm(gk,ord = 2)) * np.array(gk)

        else:
            a = np.dot(p_B,p_B) - 2*np.dot(p_B,p_U) + np.dot(p_U,p_U)
            b = 2*np.dot(p_B,p_U) - 2*np.dot(p_U,p_U)
            c = np.dot(p_U,p_U) - trust_region_radius**2

            t_star = solve_quadratic_equation(a,b,c)
            
            p_star = p_U + (t_star) * (p_B - p_U)

    return p_star


def update_step_size(r_k, delta_k, delta_max, pk):
    if r_k < 1/4:
        # Case: ρk -> 0
        delta_k1 = 1/4 * delta_k
    elif r_k > 3/4 and np.linalg.norm(pk,ord=2) == delta_k:
        # Case: ρk -> 1
        delta_k1 = min(2 * delta_k, delta_max)
    else:
        # Case: 0 << ρk < 1
        delta_k1 = delta_k
        

    return delta_k1

def quadratic_model(p, xk , y_t , gradient_xk, Bk,delta):
    if(np.linalg.norm(p) > delta):
        #print(np.linalg.norm(p))
        p = delta * (p / np.linalg.norm(p))

    f_xk = f(xk, y_t)  
    gp = np.dot(gradient_xk, p)
    Bp = np.dot(Bk, p)
    quadratic_term = 0.5 * np.dot(p, Bp)
    
    return f_xk + gp + quadratic_term

def dogleg_BFGS(x0,epsilon,y_t,max_iterations=1000):
    k = 0
    xk = x0
    h = np.identity(len(xk))
    max_radius = 100
    trust_region_radius = 1

    while(np.linalg.norm(gradient(xk,y_t),ord=2) > epsilon and max_iterations > k):
        pk = dogleg(gradient(xk,y_t),np.linalg.inv(h),h,trust_region_radius)

        if(np.dot(pk,gradient(xk,y_t)) >= 0):
            print("Not negative:",np.dot(pk,gradient(xk,y_t)))

        xnew = xk + np.array(pk)

        sk = xnew - xk    
        yk = np.subtract(gradient(xnew,y_t),gradient(xk,y_t))
        rk = 1/np.dot(yk,sk)

        A = np.identity(len(sk)) - rk*np.outer(sk,yk)
        B = np.identity(len(sk)) - rk*np.outer(yk,sk)
        h = A @ h @ B + rk*np.outer(sk,sk)

        fxk = f(xk, y_t)
        fxnew = f(xnew, y_t)

        r_k = (fxk - fxnew) / (fxk - quadratic_model(pk,xnew,y_t,gradient(xnew,y_t),np.linalg.inv(h),trust_region_radius))

        trust_region_radius = update_step_size(r_k,trust_region_radius,max_radius,pk)

        if(r_k > 0.25):
            xk = xnew

        k += 1

    print("Dogleg-BFGS iterations: ", k)

    global gradient_calls
    global function_calls

    print("Dogleg-BFGS Function Calls: ", function_calls) 
    print("Dogleg-BFGS Gradient Calls: ", gradient_calls)

    gradient_calls = 0
    function_calls = 0
    return xk

def readFile(filename):
    y_t = [0 for i in range(25)]

    file = open(filename)
    total = 0

    index = 0

    for i in range(5,30):
        file.seek(i*16 + 10)
        s = file.readline().strip()
        y_t[index] = float(s)
        total += y_t[index]
        index += 1


    y_t = [y_t[i]/(total*1/len(y_t)) for i in range(25)]
    file.close()

    y_t.reverse()
    return y_t

def genStartingPoints(seed):
    seed_value = seed
    np.random.seed(seed_value)

    starting_points = [[0 for i in range(5)] for j in range(10)] #Initializing points to 0
    for i in range(10):
        starting_points[i] = randomPointGen(-2,2)
    return starting_points


def plotGraph(y_t,coefficients,algo,starting_point):
    x_values = np.arange(1,26,1)
    y_values = y_t

    poly = Polynomial(coefficients)
    y = poly(x_values)

    X_Y_Spline = make_interp_spline(x_values, y)

    X_ = np.linspace(x_values.min(), x_values.max(), 500)
    Y_ = X_Y_Spline(X_)

    plt.plot(x_values, y_values, 'o' ,color = "blue",markersize = 2 ,label='Real Values')
    plt.plot(X_, Y_, color = "red" , lw = 1.5 ,label='Polynomial-'+algo)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(algo)

    plt.legend()

    plt.grid(True)
    plt.show()
