# Calculate the mean, median (use np.median()), and sum of all elements in the 5x5 NumPy array.
import numpy as np
arr_5d = np.array([[1, 2, 3, 4, 5],
                 [4, 5, 6, 7, 8],
                 [7, 8, 9, 10, 11], 
                 [10, 11, 12, 13, 14], 
                 [13, 14, 15, 16, 17]])
print(f"Original array: \n{arr_5d}")
print('---'*30)
print(f"Elements at 3rd row and 4th column: \n{arr_5d[3:4]}")

print(f"Mean of array: \n{np.mean(arr_5d)}")
print(f"Median of array: \n{np.median(arr_5d)}")
print(f"Sum of array: \n{np.sum(arr_5d)}")
