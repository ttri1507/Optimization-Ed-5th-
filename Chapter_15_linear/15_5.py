import cvxpy as cp
import numpy as np

# Define the variables
x_AC = cp.Variable()
x_AD = cp.Variable()
x_BC = cp.Variable()
x_BD = cp.Variable()

# Define the objective function (minimize total cost)
objective = cp.Minimize(x_AC + 2*x_AD + 3*x_BC + 4*x_BD)

# Define the constraints
constraints = [
    x_AC + x_AD <= 70,    # Supply constraint for A
    x_BC + x_BD <= 80,    # Supply constraint for B
    x_AC + x_BC == 50,    # Demand constraint for C
    x_AD + x_BD == 60,    # Demand constraint for D
    x_AC >= 0, x_AD >= 0, x_BC >= 0, x_BD >= 0  # Non-negativity
]

# Define the problem
problem = cp.Problem(objective, constraints)

# Solve the problem
problem.solve()

# Print the results
print("Minimum shipping cost:", problem.value)
print("Units from A to C:", x_AC.value)
print("Units from A to D:", x_AD.value)
print("Units from B to C:", x_BC.value)
print("Units from B to D:", x_BD.value)