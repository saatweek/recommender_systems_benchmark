import kagglehub
import pandas as pd
import numpy as np
import os
import ast
import math


def encode_genres(s: str) -> np.ndarray:
    """
    encodes genres into predefined bins
    :param s: the 'genres' column of the data has a string which is a dictionary of all the genres the movie belongs to
    :return: 10 bit binary encoded genres
    """
    start_genres = ["drama", "comedy", "action", "romance", "documentary",
              "thriller", "adventure", "fantasy", "crime", "horror"]
    # list of the genres that the movie belongs to
    genre_list = []
    try:
        # the genres column is a string, but it's actually a dictionary of all the genres, so we're
        # extracting the genres and appending it to the genre_list
        genres = ast.literal_eval(s)
        for item in genres:
            genre_list.append(item['name'].lower())
    except:
        pass

    # Encoding the genre list according to the pre-defined start_genres
    encoding = [0] * len(start_genres)
    for genre in genre_list:
        if genre in start_genres:
            encoding[start_genres.index(genre)] = 1

    return np.array(encoding)

def encode_lang(lang: str) -> np.ndarray:
    """
    encodes the language of the movie. The highest occurring languages have predefined encodings, the rest are encoded as [1,1,1]
    :param lang: the "original_language" column from the dataframe.
    :return: numpy array of encodings
    """
    lang = lang.lower()
    if lang == "en":
        return np.array([0, 0, 0])
    elif lang == "fr":
        return np.array([0, 0, 1])
    elif lang == "de":
        return np.array([0, 1, 0])
    elif lang == "ja":
        return np.array([0, 1, 1])
    elif lang == "it":
        return np.array([1, 0, 0])
    elif lang == "es":
        return np.array([1, 1, 0])
    else:
        return np.array([1, 1, 1])

def encode_vote_count(num: int) -> np.ndarray:
    """
    encoding vote_count by putting it into bins of 100 (so, movies with less than 100 reviews go in first bin,
    movies with 100-200 reviews go in second bin and so on
    :param num: int of the number of vote_counts
    :return: numpy array of bins, 1 indicating which bin the number belongs to
    """
    bins = [0] * 10
    try:
        idx = min(math.floor(num / 100), 9)
        bins[idx] = 1
    except:
        bins[-1] = 1
    return np.array(bins)

def encode_vote_avg(avg: float) -> np.ndarray:
    """
    vote_average is a float from 0-5, this function puts them into bins accordingly
    :param avg: vote_average
    :return: binned value of vote_average
    """
    if avg < 1:
        return np.array([0, 0, 0])
    elif avg < 2:
        return np.array([0, 0, 1])
    elif avg < 3:
        return np.array([0, 1, 0])
    elif avg < 4:
        return np.array([0, 1, 1])
    else:
        return np.array([1, 1, 1])

def encode_rating(rating: int) -> np.array:
    rating = int(rating)
    empty_array = [0]*10
    ones = rating*2
    empty_array[:ones] = [1]*ones
    return np.array(empty_array)

def prepare_data(reviews_per_user:int | None = None,
                 top_percentile:float | None = None,
                 num_user: int | None = None,
                 per_user : bool = True,
                 seed: int | None = None):
    """
    Prepares the data to be input for different ML models
    :param per_user: Set True if you're doing analysis on a per-user level and False if using collaborative filtering
    :param num_user: Number of unique users in the final analysis
    :param reviews_per_user: number of reviews you want from each user in the final analysis
    :param top_percentile: if specified, only movies with vote_count in the top_percentile would be considered for training
    :return: Filtered user data tailored to per-user and collaborative code
    """
    user_seed = 77 if seed is None else seed
    review_seed = 9 if seed is None else seed

    # Download dataset
    print("Downloading dataset..")
    path = kagglehub.dataset_download("rounakbanik/the-movies-dataset")

    # Load data
    print("Loading dataset..")
    movies_metadata = pd.read_csv(os.path.join(path, "movies_metadata.csv"), low_memory=False)
    ratings = pd.read_csv(os.path.join(path, "ratings.csv"))

    # We only need the 'id', 'genres', 'original_language', 'vote_count' and 'vote_average' from
    # movies_metadata, so filtering out the rest of the columns for faster processing
    movies_metadata = movies_metadata[['id', 'genres', 'original_language', 'vote_count', 'vote_average']]

    # Converting ids to numeric
    movies_metadata['id'] = pd.to_numeric(movies_metadata['id'], errors='coerce')

    # Handling missing values
    print("Handling missing values...")
    movies_metadata = movies_metadata.dropna(subset=['id'])
    movies_metadata['genres'] = movies_metadata['genres'].fillna('[]')
    movies_metadata['original_language'] = movies_metadata['original_language'].fillna('en')
    movies_metadata['vote_count'] = movies_metadata['vote_count'].fillna(0).astype(int)
    movies_metadata['vote_average'] = movies_metadata['vote_average'].fillna(0)



    # Filter out popular movies if top_percentile is defined
    if top_percentile is not None:
        print(f"Filtering movies in {top_percentile} of vote_count")
        m = ratings['vote_count'].quantile(top_percentile)
        movies_metadata = movies_metadata[movies_metadata['vote_average'] >= m]

    # Sorting rows according to time, and deleting the duplicate rows keeping only the last occurrence
    ratings.sort_values(['userId', 'timestamp'], inplace=True)
    ratings = ratings.drop_duplicates(['userId', 'movieId'], keep='last')

    if reviews_per_user is None or reviews_per_user == 0:
        if num_user is None or num_user == 0:
            # If reviews_per_user and num_user are both None, then assume that we want all the reviews from all the users
            # Hence merging the entirety of both rating and movies_metadata dataset
            print("merging reviews and movies_metadata...")
            merged = ratings.merge(movies_metadata, left_on='movieId', right_on='id', how='inner')
        else:
            # If reviews_per_user is None, but num_user is specified, then we want all the reviews from
            # num_users number of users. Hence, filter num_users number of users from ratings (that have at least 2
            # ratings; 1 for training and 1 for testing), and then merge with movies_metadata


            # Merging the ratings and movies dataset before filtering because some users have reviews for movieId that
            # are not in movies_metadata
            print("Merging rating and movies dataset..")
            merged = ratings.merge(movies_metadata, left_on='movieId', right_on='id', how='inner')
            print(f"reviews_per_user is {reviews_per_user}")
            user_review_counts = merged['userId'].value_counts().reset_index()
            user_review_counts.columns = ['userId', 'num_ratings']
            print(f"There are {len(user_review_counts)} total users")
            heavy_users = user_review_counts[user_review_counts['num_ratings'] >= 2]
            print(f"There are {len(heavy_users)} users with at least 2 reviews")
            sample_users = heavy_users['userId'].sample(n=num_user, random_state=user_seed).tolist()
            merged = merged[merged['userId'].isin(sample_users)]
    else:
        # Merging the ratings and movies dataset before filtering because some users have reviews for movieId that are
        # not in movies_metadata
        print("Merging rating and movies dataset..")
        merged = ratings.merge(movies_metadata, left_on='movieId', right_on='id', how='inner')
        print(f"reviews_per_user is {reviews_per_user}")
        user_review_counts = merged['userId'].value_counts().reset_index()
        user_review_counts.columns = ['userId', 'num_ratings']
        print(f"There are {len(user_review_counts)} total users")
        heavy_users = user_review_counts[user_review_counts['num_ratings'] >= reviews_per_user]
        print(f"There are {len(heavy_users)} users with at least {reviews_per_user} reviews")

        if num_user is not None or num_user == 0:
            # if reviews_per_user is not None, and num_user is also not None, then we want to sample reviews_per_user
            # number of reviews from num_user number of randomly sampled users
            if len(heavy_users) < num_user:
                raise ValueError(f"Only {len(heavy_users)} users have ≥{reviews_per_user} ratings (requested: {num_user})")
            print(f"But we only want {num_user} users..")
            sample_users = heavy_users['userId'].sample(n=num_user, random_state=user_seed).tolist()
            merged = merged[merged['userId'].isin(sample_users)].groupby('userId').sample(n=reviews_per_user, random_state=review_seed)
            print(f"merge.shape after sampling {num_user} users : {merged.shape}")
        else:
            # if the reviews_per_user is not None, but the num_user is None, then we want to sample review_per_user number
            # of reviews from all the users
            merged = merged[merged['userId'].isin(heavy_users['userId'])].groupby('userId').sample(n=reviews_per_user, random_state=review_seed)
            print(f"merge.shape after filtering out users with more than {reviews_per_user} reviews : {merged.shape}")

    #Encoding all the columns with the functions we created earlier
    merged['genres_enc'] = merged['genres'].apply(encode_genres)
    merged['lang_enc'] = merged['original_language'].apply(encode_lang)
    merged['vote_count_enc'] = merged['vote_count'].apply(encode_vote_count)
    merged['vote_avg_enc'] = merged['vote_average'].apply(encode_vote_avg)
    merged['rating'] = merged['rating'].apply(encode_rating)

    # Only keeping the columns we need for analysis
    merged = merged[['userId', 'movieId', 'rating', 'genres_enc', 'lang_enc', 'vote_avg_enc', 'vote_count_enc']]

    # If we don't want to do a per_user analysis then the dataframe is good enough to be returned, no further
    # changes needed
    if not per_user:
        return merged

    sorted_merged_df = merged.sort_values(by=["userId"])

    # Making a list of users, where each user is a list of row["userId"], row["movieId"], row["rating"],
    # row["genres_enc"], row["lang_enc"], row["vote_avg_enc"], row["vote_count_enc"]
    first_pass = True
    previous_userId = None
    Users_data = []
    user = []

    for j, row in sorted_merged_df.iterrows():
        if first_pass:
            first_pass = False
            la = [row["userId"], row["movieId"], row["rating"], row["genres_enc"], row["lang_enc"],
                  row["vote_avg_enc"], row["vote_count_enc"]]
            user.append(la)
            previous_userId = row["userId"]
        else:
            if row["userId"] == previous_userId:
                la = [row["userId"], row["movieId"], row["rating"], row["genres_enc"],
                      row["lang_enc"], row["vote_avg_enc"], row["vote_count_enc"]]
                user.append(la)
            else:
                Users_data.append(user)
                user = []

                la = [row["userId"], row["movieId"], row["rating"], row["genres_enc"],
                      row["lang_enc"], row["vote_avg_enc"], row["vote_count_enc"]]
                user.append(la)
                previous_userId = row["userId"]

    # Add previous user data
    if user:
        Users_data.append(user)

    return Users_data
