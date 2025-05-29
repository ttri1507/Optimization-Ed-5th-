import cvxpy as cp
import numpy as np

# Define the variables
x1 = cp.Variable()
x2 = cp.Variable()

# Define the objective function (maximize 2x1 + x2, so minimize -2x1 - x2)
objective = cp.Minimize(-2 * x1 - x2)

# Define the constraints
constraints = [
    x1 >= 0,          # x1 >= 0
    x1 <= 2,          # x1 <= 2 0 <= x1 <= 2
    x1 + x2 <= 3,     # x1 + x2 <= 3
    x1 + 2*x2 <= 5,   # x1 + 2x2 <= 5
    x2 >= 0           # x2 >= 0
]

# Define the problem
problem = cp.Problem(objective, constraints)

# Solve the problem
problem.solve()

# Print the results
print("Optimal value:", -problem.value)  # Negate to get the maximized value
print("x1:", x1.value)
print("x2:", x2.value)