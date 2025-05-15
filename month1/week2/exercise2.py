# Write a lambda function that takes three numbers and returns their product.
x = int(input("Enter number 1: "))
y = int(input("Enter number 2: "))
z = int(input("Enter number 3: "))
sum3 = lambda x, y, z: x + y + z
print(f"Sum of lambda function: {sum3(x, y, z)}")
