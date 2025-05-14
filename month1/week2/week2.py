# Advanced Python for Data Handling:
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
even_nums = list(filter(lambda x: x % 2 == 0, numbers))
print(f"Even numbers: {even_nums}")
print("---"*30)

