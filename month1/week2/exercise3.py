# Create a 5x5 NumPy array with random integers.
    # Select and print the element in the 3rd row and 4th column.
    # Select and print all elements in the 2nd row.
    # Select and print all elements in the 3rd column.
    # Select all elements greater than a certain value (e.g., 50 if your random numbers are between 1 and 100).
import numpy as np
arr_5d = np.array([[1, 2, 3, 4, 5],
                 [4, 5, 6, 7, 8],
                 [7, 8, 9, 10, 11], 
                 [10, 11, 12, 13, 14], 
                 [13, 14, 15, 16, 17]])
print(f"Original array: \n{arr_5d}")
print('---'*30)
print(f"Elements at 3rd row and 4th column: \n{arr_5d[3:4]}")