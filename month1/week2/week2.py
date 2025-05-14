# 1. Advanced Python for Data Handling:
# List Comprehensions
squares = [x**2 for x in range(10)]
print(squares)
print("---"*30)

even_nums = [x for x in range(20) if x % 2 == 0]
print(even_nums)
print("---"*30)

#Lambda Functions
add = lambda x, y: x + y
print(f"Sum using lambda: {add(5, 3)}")
print("---"*30)

#map() with lambda
numbers = [1, 2, 3, 4, 5]
doubled = list(map(lambda x: x * 2, numbers))
print(f"Doubled numbers: {doubled}")
print("---"*30)

#filter() with lambda
ages = [12, 17, 22, 23, 18]
adults = list(filter(lambda age: age >= 18, ages))
print(f"Adults ages with filter: {adults}")
print("---"*30)

 #Working with files
 #Writing to file
with open("ex_data.txt", "w") as f:
    f.write("Hello, python for ML!\n")
    f.write("This is week 2.\n")
    print("---"*30)
#Reading
# with open("month1/week2/week2.txt", "r") as f:
#     content_file1 = f.read()
#     print(f"Content from file:\n{content_file1}")
#     print("---"*30)

#2. NumPy - Deeper Dive:
import numpy as np

#indexing and slicing
arr_1d = np.arange(10)
print(f"OG 1d array: {arr_1d}")
print(f"Element at index 3: {arr_1d[3]}")
print(f"Elements from index 1-3: {arr_1d[1:3]}")
print(f"Elements from index 6 to the end: {arr_1d[6:]}")
print(f"Elements from index beginning to index 6: {arr_1d[:6]}")
arr_1d[0:2] = 99
print(f"Modified 1d array: {arr_1d}")
print("---"*30)

arr_2d = np.array([[1, 2, 3], 
                   [4, 5, 6], 
                   [7, 8, 9]])
print(f"Original 2d array: \n{arr_2d}")
print(f"Element at row 1 column 2: {arr_2d[1, 2]}") #row, column
print(f"Entire first row: {arr_2d[0, :]}") #index 0 row, : no column
print(f"Second column: {arr_2d[:, 1]}") #: no row, index 1 column
print(f"Sub array (first 2 rows, last 2 columns: \n{arr_2d[0:2, 1:3]})")
print("---"*30)

#Boolean Indexing
ages_arr = np.array([17, 22, 32, 27, 12, 19, 31, 26, 16])
adult_ages_arr = ages_arr[ages_arr >= 18]
print(f"Ages array: {ages_arr}")
print(f"Adults over 18 (using boolean indexing): {adult_ages_arr}")
print(f"Ages greater than 30: {ages_arr[ages_arr > 30]}")
print("---"*30)

#Mathamatical and statistical functions
data = np.array([10, 15, 12, 18, 11, 20])
print(f"Data array: {data}")
print(f"Min: {np.min(data)}, Max: {np.max(data)}")
print(f"Sum: {np.sum(data)}")
print(f"Mean: {np.mean(data)}")
print(f"Standard deviation: {np.std(data)}")
print("---"*30)

vec_a = np.array([1, 2, 3])
vec_b = np.array([4, 5, 6])
dot_product = np.dot(vec_a, vec_b)
print(f"Dot product of a and b: {dot_product}")
print("---"*30)

#Reshaping arrays
arr_to_reshape = np.arange(1, 10) #1-9
reshaped_arr = arr_to_reshape.reshape((3, 3))
print(f"OG array for reshaping: {arr_to_reshape}") #flat
print(f"Reshaped arr: \n{reshaped_arr}") #reshaped to 3x3
print("---"*30)

#Stacking and splitting
a = np.array([[1,2],[3,4]])
b = np.array([[5,6]])
v_stacked = np.vstack((a, b))
print(f"A: {a}, \nB: {b}")
print(f"Stacked using vstack: \n{v_stacked}")
print("---"*30)

c = ([[5], [6]])
h_stacked = np.hstack((a, c))
print(f"Stacked using hstack: {h_stacked}")
print("---"*30)

arr_to_split = np.arange(9.0)
split_arr = np.split(arr_to_split, 3)
print(f"Split array: {split_arr}")
print("---"*30)

print("\n \n \nPandas Learning")
print("---"*30)


#3. Intro to pandas
import pandas as pd

#Creating pandas series
s_data = [10, 20, 30, 40, 50]
s_labels = ["a", "b", "c", "d", "e"]
my_series = pd.Series(data=s_data, index=s_labels)
print(f"Pandas Series: \n{my_series}")
print(f"Value at label 'c': {my_series['c']}")
print("---"*30)

#Pandas DataFrame
data_dict = {
    'Name': ['Alice', 'Mike', 'Palmer'],
    'Age': [22, 34, 10],
    'City': ['New York', 'Tucson', 'Petersburg']
}
df = pd.DataFrame(data_dict)
print(f"DataFrame: \n{df}")
print("---"*30)

#DataFrame from numpy array
np_data = np.array([['Tom', 22], ['Mike', 12], ['Jim', 23]])
df_from_np = pd.DataFrame(data=np_data, columns=['Name', 'Age'])
print(f"DataFrame from numpy: \n{df_from_np}")
print("---"*30)

#Basic DF operations
print(f"First 2 rows (head): \n{df.head(2)}")
print("---"*30)
print(f"Last 2 rows (tail): \n{df.tail(2)}")
print("---"*30)
print(f"DataFrame info: \n")
df.info()
print("---"*30)
print(f"Descreptive stats: \n{df.describe(include='all')}")
print("---"*30)

#Selecting columns
print(f"'Name' column: \n{df['Name']}")
print("---"*30)

#Selecting rows
print(f"First row (using iloc): \n{df.iloc[0]}")
print("---"*30)
print(f"Row with index 1 (using loc if index is default): \n{df.loc[1]}")
print("---"*30)

csv_content = "ID,Value1,Value2\n1,10,100\n2,20,200\n3,30,300\n4,40,400\n5,50,500"
with open("month1\week2\data.csv", "w") as f:
    f.write(csv_content)

try:
    df_from_csv = pd.read_csv("month1\week2\data.csv")
    print(f"DataFrame from CSV: \n{df_from_csv}")
    print("---"*30)
    print(f"Info from dataframe: \n")
    df_from_csv.info()
    print("---"*30)
except FileNotFoundError:
    print("Error: data.csv not found, check path.")