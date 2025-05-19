#1.
# Create two vectors, u = [1, 2, 3] and v = [4, 0, -1].
# Calculate u + v.
# Calculate 3 * u.
# Calculate the dot product of u and v.
# Calculate the L2 norm of u and v.

# 2.
# Create two matrices: $M1 = \\begin{pmatrix} 1 & 0 \\ 2 & -1 \\end{pmatrix}$ $M2 = \\begin{pmatrix} 3 & 4 \\ 1 & 2 \\end{pmatrix}$
# Calculate M1 + M2.
# Calculate M1 - M2.
# Calculate the matrix product M1 @ M2.
# Calculate the transpose of M1.

# 3.
# What is an identity matrix? Create a 3x3 identity matrix using NumPy. Multiply any 3x3 matrix you create by this identity matrix and observe the result.
# In your own words, explain what a derivative represents. Why is this concept important for training machine learning models?
# In your own words, explain what a gradient represents for a function with multiple variables. How is it used in an algorithm like gradient descent?
import numpy as np

u = np.array([1, 2, 3])
v = np.array([4, 0, -1])

print(u + v)
print(3 * u)
print(np.dot(u, v))
print(np.linalg.norm(u))
print(np.linalg.norm(v))
print("---"*30)

M1 = np.array([[1, 0], [2, -1]])
M2 = np.array([[3, 4], [1, 2]])

print(M1 + M2)
print(M1 - M2)
print(M1 @ M2)
print(M1.T)
print("---"*30)

I = np.eye(3)
print(I)
print("---"*30)

M = np.random.rand(3, 3)
print(M)
print(M @ I)

