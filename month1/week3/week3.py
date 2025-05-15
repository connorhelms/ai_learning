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

