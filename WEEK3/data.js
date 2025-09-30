import pandas as pd

# Load dataset
ratings_df = pd.read_csv("u.data", sep="\t", names=["user_id", "movie_id", "rating", "timestamp"])
movies_df = pd.read_csv("u.item", sep="|", names=["movie_id", "title"], usecols=[0, 1], encoding="latin-1")

# Map users and movies to indices
users = sorted(ratings_df["user_id"].unique())
movies = movies_df.sort_values("movie_id")["title"].tolist()

user_to_idx = {u: i for i, u in enumerate(users)}
movie_to_idx = {m: i for i, m in enumerate(movies_df["movie_id"].values)}

# Initialize rating matrix
ratings = [[0] * len(movies) for _ in range(len(users))]

# Fill ratings
for _, row in ratings_df.iterrows():
    u_idx = user_to_idx[row["user_id"]]
    m_idx = movie_to_idx[row["movie_id"]]
    ratings[u_idx][m_idx] = int(row["rating"])

# Write to data.js
with open("data.js", "w", encoding="utf-8") as f:
    f.write("const users = " + str(users) + ";\n\n")
    f.write("const movies = " + str(movies) + ";\n\n")
    f.write("const ratings = " + str(ratings) + ";\n")
