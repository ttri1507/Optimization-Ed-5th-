import numpy as np

# Define the data matrix X
X = np.array([[1, 3.1, 5.1, 7.9, 10, 12],
              [45, 42, 50, 24, 39, 28]])

# Part a: Compute the condition number of the Hessian of the quadratic loss
# The Hessian is given by (X @ X.T) / 3, scaled by 1/6
# Compute X @ X.T
XXT = X @ X.T
Hessian = (1/3) * XXT / 6  # (1/3) * (X @ X.T) / 6

# Compute eigenvalues of the Hessian
eigenvalues = np.linalg.eigvals(Hessian)
condition_number_a = np.max(np.abs(eigenvalues)) / np.min(np.abs(eigenvalues))
print(f"Part a: Condition number of the Hessian = {condition_number_a:.4f}")

# Part b: Apply standard data normalization to X
# Standard normalization: (x - mean) / std
X_normalized = (X - np.mean(X, axis=1, keepdims=True)) / np.std(X, axis=1, keepdims=True)
print("\nPart b: Normalized data matrix X̃:")
print(X_normalized)

# Part c: Compute the condition number of the Hessian with normalized data
XXT_normalized = X_normalized @ X_normalized.T
Hessian_normalized = (1/3) * XXT_normalized / 6

# Compute eigenvalues of the normalized Hessian
eigenvalues_normalized = np.linalg.eigvals(Hessian_normalized)
condition_number_c = np.max(np.abs(eigenvalues_normalized)) / np.min(np.abs(eigenvalues_normalized))
print(f"\nPart c: Condition number of the Hessian with normalized data = {condition_number_c:.4f}")

# Comparison
print("\nComparison:")
print(f"Condition number before normalization: {condition_number_a:.4f}")
print(f"Condition number after normalization: {condition_number_c:.4f}")
print("The condition number decreases significantly after normalization, indicating that the Hessian of the quadratic loss becomes better conditioned, which can lead to more stable optimization.")