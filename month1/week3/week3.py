#1. Scalars, Vectors, Matrices with NumPy:
import numpy as np

#Scalar
scalar_a = 10
print(f"Scalar a: {scalar_a}")
print("---" *30)

#Vector 1d array
vector_v = np.array([1, 2, 3])
vector_w = np.array([4, 5, 6])
print(f"Vector v: {vector_v}")
print(f"Shape of vector v: {vector_v.shape}")
print("---"*30)

#Matrix 2d array
matrix_a = np.array([[1, 2], [3, 4], [5, 6]])
matrix_b = np.array([[7, 8], [9, 10], [11, 12]])
print(f"Matrix A: \n{matrix_a}")
print(f"Shape of matrix a: \n{matrix_a.shape}")
print("---"*30)

# Tensor (NumPy arrays can be N-dimensional)
tensor_t = np.array([
    [[1,2],[3,4]],
    [[5, 6], [7,8]],
    [[9,10],[11,12]]
])#3x2x2 tensor
print(f"Tensor t (shape: {tensor_t.shape}): \n{tensor_t}")
print("---"*30)



#2. Vector Operations with NumPy:
v = np.array([1, 2, 3])
w = np.array([4, 5, 6])
s = 2 #scalar

#Addition
vector_sum = v + w
print(f"v + w = {vector_sum}")
print("---"*30)

#Subtraction
vector_diff = v - w
print(f"v - w = {vector_diff}")
print("---"*30)

#Multiplication
scalar_mult = s * v
print(f"s * v = {scalar_mult}")
print("---"*30)

#Dot product
#Method 1: np.dot()
dot_product1 = np.dot(v, w)
print(f"Dot product (np.dot(v, w)): {dot_product1}")
#Method 2: @ operator
dot_product2 = v @ w
print(f"Dot product (v @ w): {dot_product2}")
#Method 3
dot_product3 = np.sum(v * w)
print(f"Dot product (sum(v * w)): {dot_product3}")
print("---"*30)

#Vector Norm
#L2 norm 
norm_v_l2 = np.linalg.norm(v)
#sqrt(1^2 + 2^2 + 3^2) = sqrt(1 + 4 + 9) = sqrt(14)
print(f"L2 norm of v: {norm_v_l2}")
print("---"*30)

#L1 norm
norm_v_l1 = np.linalg.norm(v)
# |1| + |2| + |3| = 1 + 2 + 3 = 6
print(f"L1 norm of v: {norm_v_l1}")
print("---"*30)

#3. Matrix Operations with NumPy:

A = np.array([[1,2], [3,4]])
B = np.array([[5,6], [7,8]])
print(f"Matrix A: \n{A}")
print(f"Matrix B: \n{B}")
print("---"*30)

#Addition
matrix_sum = A + B
print(f"Sum of matrix A + B = {matrix_sum}.")
print("---"*30)

#Subtraction
matrix_diff = A - B
print(f"Difference of A - B = {matrix_diff}.")
print("---"*30)

#Scalar multiplication
scalar_mult_matrix = s * A
print(f"Scalar multiplication s * A = {scalar_mult_matrix}.")
print("---"*30)

#Element-wise multiplication
element_wise_mult = A * B
print(f"Element wise multiplication (not matrix) A * B = {element_wise_mult}.")
print("---"*30)

#Martix Multiplication (Dot product)
#Method #1: np.dot()
matrix_mult1 = np.dot(A, B)
# [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
print(f"Matrix Multiplication (np.dot(A, B)): \n{matrix_mult1}")
print("---"*30)
#Method 2: @ operator
matrix_mult2 = A @ B
print(f"Matrix multiplication (A @ B): \n{matrix_mult2}")
print("---"*30)

#Transposition
A_transpose = A.T
#or  np.transpose(A)
print(f"Transpose of A: \n{A_transpose}")
print("---"*30)

#Identity Matrix
# A square matrix with ones on the main diagonal and zeros elsewhere.
# A @ I = A
identity_2x2 = np.eye(2)
print(f"Identity matrix (2x2): \n{identity_2x2}")
print(f"A @ I: \n{A @ identity_2x2}")
print("---"*30)

# Inverse Matrix (Conceptual)
# For a square matrix A, its inverse A_inv is such that A @ A_inv = I (Identity)
# Not all matrices have an inverse.
# NumPy can calculate it:
try:
    A_inv = np.linalg.inv(A)
    print(f"Inverse of A: \n{A_inv}")
    print(f"A @ A_inv (should be close to identity: \m{A @ A_inv})")
except np.linalg.LinAlgError:
    print("Matrix A is a singular and does not have an inverse.")
print("---"*30)





# Illustrative: A simple function and its "slope" idea
# We won't calculate derivatives here, just visualize the concept.
import matplotlib.pyplot as plt # We'll cover this more next week

def f(x):
    return x**2

x_vals = np.linspace(-10, 10, 100)
y_vals = f(x_vals)

plt.plot(x_vals, y_vals, label='$f(x) = x^2$')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Concept of a function and its slope')

# At x=2, f(x)=4. The slope (derivative) is 2*2=4
plt.plot(2, f(2), 'ro') #Martgk this point
#Plot tangent line
x_tan = np.array([0, 4])
y_tan = 4 * (x_tan -2) + 4
plt.plot(x_tan, y_tan, 'r--', label='Approximate tangent (slope=4) at x=2')
plt.legend()
plt.grid(True)
plt.show()
print("\nA plot illustrating a function and its tangent (slope) would be shown if Matplotlib is displayed.")
print("The key idea is that the derivative gives the slope at a point.")
