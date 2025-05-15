# Create a Pandas DataFrame with information about 5 of your favorite books (Title, Author, Year Published, Pages).
    # Display the first 3 books.
    # Display only the 'Title' and 'Author' columns.
    # Get a statistical summary of the 'Pages' column.

import pandas as pd
book_list = {
    'Title': ['0 to 1', 'Technology Republic', 'The Bible', 'The Alchamist', 'The Fish That Ate the Whale'],
    'Author': ['Peter Thiel', 'Alex Karp', 'God', 'Paulo Coelho', 'Caleb Carr'],
    'Year Published': [2014, 2017, 2000, 1994, 2005],
    'Pages': [256, 304, 1176, 158, 544]
}
df = pd.DataFrame(book_list)
print(f"Original DataFrame: \n{df}")
print("---"*30)
print(f"First 3 books in list: \n{df[:3]}")
print("---"*30)
print(f"'Title' column: \n{df['Title']}")
print("---"*30)
print(f"'Author' column: \n{df['Author']}")
print("---"*30)
print(f"Statistical summary of 'Pages' column: \n{df['Pages'].describe()}")
print("---"*30)