import pandas as pd
import matplotlib.pyplot as plt

# Read the csv files
df = pd.read_csv('/content/df.csv')
rating = pd.read_csv('/content/rating.csv')

# Show the first few rows and shape of the dataset
print(df.head())
print(df.shape)

# Drop the 'genres' column from df if it exists
if 'genres' in df.columns:
    df.drop(['genres'], axis=1, inplace=True)

# Show columns of the rating dataset
print(rating.columns)

# Keep only necessary columns in rating
rating = rating.loc[:, ["user id", "movie id", "rating"]]
print(rating.head())

# Merge movie and rating data
df = pd.merge(df, rating, left_on='movie id', right_on='movie id')

# Limit the dataset to the first 1,000,000 rows if needed
df = df.iloc[:1000000]

# Basic statistics
print(df.describe())

# Group by 'title' and calculate mean and count of ratings
ratings = pd.DataFrame(df.groupby("title").mean()['rating'])
ratings['number of ratings'] = pd.DataFrame(df.groupby("title").count()["rating"])
print(ratings.head())

# Sort ratings
print(ratings.sort_values(by='rating', ascending=False))
print(ratings.describe())

# Plot histograms
plt.hist(ratings['rating'])
plt.show()

plt.hist(ratings['number of ratings'], bins=50)
plt.show()

# Create a pivot table
pivot_table = df.pivot_table(index=["user id"], columns=["title"], values="rating")
print(pivot_table.head(5))
print(pivot_table.shape)

# Define the recommendation function
def recommend_movie(movie):
    if movie in pivot_table:
        movie_watched = pivot_table[movie]
        similarity_with_other_movies = pivot_table.corrwith(movie_watched, drop=True)  # Handle missing values
        similarity_with_other_movies = similarity_with_other_movies.dropna().sort_values(ascending=False)
        return similarity_with_other_movies.head()
    else:
        return "Movie not found in the dataset."

# Call the recommendation function
print(recommend_movie('American President, The (1995)'))
print(recommend_movie('Toy Story (1995)'))
print(recommend_movie('Taxi Driver (1976)'))
