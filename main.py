from math import sqrt
from functions import *

#Georgios Mpalanos

real_values = readFile('Government_Securities_Yield_GR.txt')
epsilon = 1e-6
seed = 4561620

starting_points = genStartingPoints(seed)

bfgs_mean = 0
bfgs_results = [0 for i in range(10)]
index = 0

for i in starting_points:
    bfgs = BFGS(i,epsilon,real_values,max_iterations=500)
    f_val = f(bfgs,real_values)

    bfgs_mean += 0.1*f_val
    bfgs_results[index] = f_val
    index += 1

    print("Starting-point: ",i)
    print("Minimizer: ",bfgs)
    print("f(x): ",f_val)
    plotGraph(real_values,bfgs,"BFGS",i)
    print()

    
    
    futureValues = [3.01,2.98,3.00,2.96,2.97]
    x_values = np.arange(1,6,1)
    y_values = futureValues

    poly = Polynomial(bfgs)
    y = poly(x_values)

    X_Y_Spline = make_interp_spline(x_values, y)
    X_ = np.linspace(x_values.min(), x_values.max(), 500)
    Y_ = X_Y_Spline(X_)

    result = f(bfgs,futureValues)

    plt.plot(x_values, y_values, 'o' ,color = "blue",markersize = 2 ,label='Real Values')
    plt.plot(X_, Y_, color = "red" , lw = 1.5 ,label='Polynomial-BFGS')

    plt.xlabel('x')
    plt.ylabel('y')
    title = f'BFGS\nf(x) = {result}'
    plt.title(title)

    plt.legend()

    plt.grid(True)
    plt.show()
    print("BFGS: Future values f(x) = ",result)
    print()


temp = [(i - bfgs_mean)**2 for i in bfgs_results]
variance = 0
for i in temp:
    variance += 0.1*i

print("BFGS(with wolfe conditions for line search) statistics:")
print("Mean f value: ",bfgs_mean)
print("Standard deviation: ",sqrt(variance))
print("Minimun f value: ",min(bfgs_results))
print("Maximum f value: ",max(bfgs_results))
print()


dogleg_mean = 0
dogleg_results = [0 for i in range(10)]
index = 0

result = 0

for i in starting_points:
    doglegBFGS = dogleg_BFGS(i,epsilon,real_values,max_iterations=500)
    f_val = f(doglegBFGS,real_values)   

    dogleg_mean += 0.1*f_val
    dogleg_results[index] = f_val
    index += 1

    print("Starting-point: ",i)
    print("Minimizer: ",doglegBFGS)
    print("f(x): ",f(doglegBFGS,real_values))
    plotGraph(real_values,doglegBFGS,"Dogleg BFGS",i)
    print()

    
    futureValues = [3.01,2.98,3.00,2.96,2.97]
    x_values = np.arange(1,6,1)
    y_values = futureValues

    poly = Polynomial(doglegBFGS)
    y = poly(x_values)

    X_Y_Spline = make_interp_spline(x_values, y)
    X_ = np.linspace(x_values.min(), x_values.max(), 500)
    Y_ = X_Y_Spline(X_)

    result = f(doglegBFGS,futureValues)

    plt.plot(x_values, y_values, 'o' ,color = "blue",markersize = 2 ,label='Real Values')
    plt.plot(X_, Y_, color = "red" , lw = 1.5 ,label='Polynomial-Dogleg BFGS')

    plt.xlabel('x')
    plt.ylabel('y')
    title = f'Dogleg BFGS\nf(x) = {result}'
    plt.title(title)

    plt.legend()

    plt.grid(True)
    plt.show()

    print("Dogleg BFGS: Future f(x) = ",result)
    print()


temp = [(i - dogleg_mean)**2 for i in dogleg_results]
variance = 0
for i in temp:
    variance += 0.1*i

print("Dogleg BFGS statistics:")
print("Mean f value: ",dogleg_mean)
print("Standard deviation: ",sqrt(variance))
print("Minimun f value: ",min(dogleg_results))
print("Maximum f value: ",max(dogleg_results))
print()