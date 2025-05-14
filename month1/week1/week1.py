#1. Basic Python syntax
age = 21
height = 6.4
name = input("Enter your name: ")
is_student = input("Are you a student? (yes/no): ").lower() == "yes"
print(f"Name: {name}, Age: {age}, Height: {height}, Is Student: {is_student}")
print(type(age), type(height), type(name), type(is_student))
#Math
a = 10
b = 3
print(f"a + b = {a + b}")
print(f"a - b = {a - b}")
print(f"a * b = {a * b}")
print(f"a / b = {a / b}") # Float division
print(f"a // b = {a // b}") # Integer division
print(f"a % b = {a % b}")  # Modulus
print(f"a ** b = {a ** b}") # Exponentiation
# Comparison and Logical Operators
x = 5
y = 10
print(f"x < y: {x < y}")
print(f"x == 5 and y > 5: {x == 5 and y > 5}")
print(f"x < 3 or y > 15: {x < 3 or y > 15}")
print(f"not (x == y): {not (x == y)}")

#2. Control Flow
#if/elif/else
money = 69
tank_price = 33
if money >= tank_price:
    print("Get a full tank.")
elif money >= 15:
    print("Get a half tank.")
else:
    print("No gas for you.")

#for loop
numbers = [1, 2, 3, 4, 5]
for num in numbers:
    print(f"Square of {num} is {num**2}")

#while loop
count = 0
while count < 5:
    print(f"Count is: {count}")
    count += 1 

#3. Data Structures (Basics)
#Lists
my_list = [1, "hello", 3.14]
my_list.append(True)
print(f"List items: {my_list}")
print(f"First element: {my_list[0]}")
my_list[1] = "world"
print(f"Modified list: {my_list}")

#Tuples (immutable)
my_tuple = (10, 20, "setup")
print(f"Tuple: {my_tuple}")
print(f"Second element: {my_tuple[1]}")

#Dictionaries
my_dict = {"name": "John", "age": 25, "city": "New York"}
print(f"Dictionary: {my_dict}")
print(f"Name: {my_dict['name']}")
my_dict["age"] = 26
print(f"Updated dictionary: {my_dict}")

#Sets
my_set = {1, 2, 3, 4, 5}
print(f"Set: {my_set}")
my_set.add(6)
print(f"Updated set: {my_set}")
print(f"Is 3 in set? {'3' in my_set}")

#Functions
def greet(name):
    message = f"Hello, {name}"
    return message
greeting = greet(input("Enter your name: "))
print(greeting)

def calculate_area(length, width):
    area = length * width
    return area
rect_width1 = input("Enter the width of the first rectangle: ")
rect_length1 = input("Enter the length of the first rectangle: ")
rect_width1_float = float(rect_width1)
rect_length1_float = float(rect_length1)
area = calculate_area(rect_width1_float, rect_length1_float)
print(f"The area of the first rectangle is {area}")

#4. Modules and Numpy intro
#Math
import math
print(f"Sqrt of 16 is {math.sqrt(16)}")
print(f"Pi is {math.pi}")

#Numpy
import numpy as np

list_a = [1, 2, 3, 4, 5]
numpy_arr_a = np.array(list_a)
print(f"Numpy array: {numpy_arr_a}")
print(f"Type of numpy array: {type(numpy_arr_a)}")
print(f"First element: {numpy_arr_a[0]}")
print(f"Last element: {numpy_arr_a[-1]}")

numpy_arr_b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
print(f"Numpy array b: {numpy_arr_b}")
print(f"Type of numpy array b: {type(numpy_arr_b)}")

zeros_arr = np.zeros((2, 3))
print(f"Zeros array: {zeros_arr}")

ones_arr = np.ones((2, 3))
print(f"Ones array: {ones_arr}")

range_arr = np.arange(0, 10, 0.1) #start, stop, step
print(f"Range array: {range_arr}")

linspace_arr = np.linspace(0, 1, 5) #start, stop, num_points
print(f"Linspace array: {linspace_arr}")

# Array attributes
print(f"Shape of numpy_arr_a: {numpy_arr_a.shape}")
print(f"Shape of zeros_arr: {zeros_arr.shape}")
print(f"Number of dimensions of zeros_arr: {zeros_arr.ndim}")
print(f"Data type of numpy_arr_a elements: {numpy_arr_a.dtype}")
print("---"*30)

arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([6, 7, 8, 9, 10])
print(f"arr1 + arr2 = {arr1 + arr2}")
print(f"arr1 * 2 = {arr1 * 2}")
print(f"arr1 * arr2 = {arr1 * arr2}") # Element-wise multiplication
print("---"*30)
