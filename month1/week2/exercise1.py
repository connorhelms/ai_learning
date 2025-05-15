# Create a list of numbers from 1 to 100. Use a list comprehension to create a new list containing only numbers divisible by 3.
div_by_3 = [x for x in range(100) if x % 3 == 0]
print(div_by_3)