# Create a function that calculates the factorial of a number.
def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)

n = int(input("Enter a number: "))
print(f"The factorial of {n} is {factorial(n)}")
