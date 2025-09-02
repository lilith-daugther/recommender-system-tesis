import pandas as pd

#leemos la cabecera (nrows=0) para ver los nombres de columna sin cargar datos
books_header  = pd.read_csv('data/archive/books_data.csv',  sep=',', nrows=0)
ratings_header = pd.read_csv('data/archive/books_rating.csv', sep=',', nrows=0)

print("Columns en books_data.csv:") 
print(books_header.columns.tolist(), "\n")

print("Columns en books_rating.csv:")
print(ratings_header.columns.tolist())

"""
Columns en books_data.csv:
['Title', 'description', 'authors', 'image', 'previewLink', 'publisher', 'publishedDate', 'infoLink', 'categories', 'ratingsCount'] 

Columns en books_rating.csv:
['Id', 'Title', 'Price', 'User_id', 'profileName', 'review/helpfulness', 'review/score', 'review/time', 'review/summary', 'review/text']

"""