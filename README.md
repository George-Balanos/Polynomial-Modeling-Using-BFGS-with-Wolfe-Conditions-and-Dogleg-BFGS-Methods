# Polynomial-Modeling-Using-BFGS-with-Wolfe-Conditions-and-Dogleg-BFGS-Methods

### Introduction
The provided file, Government Securities Yield GR.txt, contains the yields (%) of 3-year Greek government bonds over a period of 30 business days, spanning from October 14, 2022, to November 25, 2022. This project aims to model this time series data using a polynomial approach. While polynomial approximations may not perfectly predict future yields, they offer simple and effective analytical models that can significantly aid investors in the analysis and forecasting of financial time series.

The precision of the polynomial model’s approximation directly depends on the choice of its parameters. To achieve an optimal fit, it is essential to carefully optimize these parameters based on the historical yield data. Polynomial models are particularly advantageous for optimization tasks, especially when using derivative-based methods, as they produce continuously differentiable objective functions. This quality allows for the application of sophisticated optimization algorithms, such as BFGS with Wolfe conditions and Dogleg BFGS, to enhance model performance and accuracy.

This project implements optimization algorithms to fit a polynomial model to government securities yield data. The main objective is to minimize the error between predicted and actual yield values using two methods: BFGS and Dogleg BFGS.

As this is my first optimization project, I recognize that there may be mistakes or areas for improvement in my approach. There is always room to refine the optimization techniques and explore alternative methods to achieve better results. Future iterations could focus on addressing these potential shortcomings to enhance the overall robustness and effectiveness of the model.

Requirements
To run this project, you need the following Python packages:

* numpy
* matplotlib
* scipy

You can install the required packages using pip:

```cpp
pip install numpy matplotlib scipy
```

### Usage
1. Data Preparation: Ensure that Government_Securities_Yield_GR.txt is in the same directory as your scripts. This file should contain the yield data formatted appropriately.
2. Run the Main Script: Execute the main script to perform the optimization.

### Main Components

Optimization Algorithms
* BFGS (Broyden-Fletcher-Goldfarb-Shanno):
  * Utilizes an iterative approach to minimize the objective function by estimating the Hessian matrix.
* Dogleg BFGS:
  * A trust-region method that combines the BFGS method with a trust-region approach for improved convergence.
 
### Functions

* readFile(filename): Reads yield data from a specified file and normalizes it.
* genStartingPoints(seed): Generates random starting points for optimization using a given seed.
* f(a, y_t): Computes the mean squared error between the polynomial predictions and actual yield values.
* plotGraph(y_t, coefficients, algo, starting_point): Plots the actual yield values against the predicted polynomial values.

### Outputs
The script will output the following for each starting point:

* The starting point used for optimization.
* The resulting minimizer coefficients.
* The calculated function value 𝑓(𝑥)
* A plot comparing actual yield values against predicted values from the polynomial.

### Example Output
After running the script, you will see printed outputs similar to the following:

```cpp
Starting-point: [0.5, -1.3, ...]
Minimizer: [1.2, -0.5, ...]
f(x): 0.0032
BFGS: Future values f(x) = 0.0035
...
```

![image](https://github.com/user-attachments/assets/4b3555a9-491f-4dca-bb24-b3ee6a313d01)
![image](https://github.com/user-attachments/assets/d6dd7139-8865-47da-96ae-06c62eb83091)


### Execution Instructions

* To run the code, execute the main.py file without any additional arguments.
* Make sure you have the file Government_Securities_Yield_GR.txt in the same folder as main.py and functions.py.
