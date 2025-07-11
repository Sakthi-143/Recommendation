# Recommendation

# Recommendation System Assignment

## Problem Statement

Build a recommender system using cosine similarity scores.

## Table of Contents
1. [Importing Libraries](#importing-libraries)
2. [Importing Dataset](#importing-dataset)
3. [Data Understanding](#data-understanding)
4. [Renaming the Columns](#renaming-the-columns)
5. [Calculating Average Rating of Books](#average-rating-of-books)
6. [Calculating Cosine Similarity between Users](#calculating-cosine-similarity-between-users)
7. [Result](#result)

---

## Importing Libraries <a name="importing-libraries"></a>

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


Importing Dataset <a name="importing-dataset"></a>


from google.colab import files
uploaded = files.upload()

import pandas as pd

# Try different encodings until you find the correct one
encodings_to_try = ['utf-8', 'latin1', 'ISO-8859-1', 'utf-16']

for encoding in encodings_to_try:
    try:
        book = pd.read_csv('book.csv', encoding=encoding)
        print(f"File read successfully with encoding: {encoding}")
        break
    except UnicodeDecodeError:
        print(f"Error decoding with encoding: {encoding}")

# Display the data
print(book)


Data Understanding <a name="data-understanding"></a>


book.head()

book.tail()

book.shape

book.info()

book.describe()

book.isnull().sum()

book.drop(book.columns[[0]], axis=1, inplace=True)

book.nunique()

book.columns


Renaming the Columns <a name="renaming-the-columns"></a>



book.columns = ["UserID", "BookTitle", "BookRating"]

book = book.sort_values(by=['UserID'])




Calculating Average Rating of Books <a name="average-rating-of-books"></a>



AVG = book['BookRating'].mean()
print(AVG)

minimum = book['BookRating'].quantile(0.90)
print(minimum)

q_Books = book.copy().loc[book['BookRating'] >= minimum]
q_Books.shape


Calculating Cosine Similarity between Users <a name="calculating-cosine-similarity-between-users"></a>



from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine, correlation

user_sim = 1 - pairwise_distances(book_df.values, metric='cosine')

user_sim_df = pd.DataFrame(user_sim)

user_sim_df.index = book.UserID.unique()
user_sim_df.columns = book.UserID.unique()

np.fill_diagonal(user_sim, 0)
