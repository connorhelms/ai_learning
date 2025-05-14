#Write a Python script that asks the user for their name and age, and then prints a message like "Hello [Name], you will be [Age+1] next year."
import datetime
name = input("Enter you name please: ")
age = int(input("Enter your age please: "))
birth_day = input("Enter your birthday (MM/DD): ")
current_year = datetime.datetime.now().year
print(f"Hello {name}, you are currently {age}, on {birth_day}/{current_year} you will turn {age + 1}.")