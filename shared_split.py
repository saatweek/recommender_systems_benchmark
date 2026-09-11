"""Common per-user benchmark partitions and auditable row fingerprints."""
import hashlib
import json
import math

import numpy as np
import pandas as pd


def split_user_rows(rows, split=0.8):
    if not 0 < split < 1:
        raise ValueError("split must be between 0 and 1")
    boundary = math.floor(len(rows) * split)
    if boundary == 0 or boundary == len(rows):
        raise ValueError("Each user needs training and test reviews")
    return rows[:boundary], rows[boundary:]


def frame_users(df):
    columns = ["userId", "movieId", "rating", "genres_enc", "lang_enc", "vote_avg_enc", "vote_count_enc"]
    return [group[columns].values.tolist() for _, group in
            df.sort_values(by=["userId"]).groupby("userId", sort=False)]


def split_prepared_frame(df, split=0.8):
    train_parts, test_parts = [], []
    for _, group in df.sort_values(by=["userId"]).groupby("userId", sort=False):
        train, test = split_user_rows(group, split)
        train_parts.append(train)
        test_parts.append(test)
    return pd.concat(train_parts), pd.concat(test_parts)


def split_fingerprints(users, split=0.8):
    hashes = [hashlib.sha256(), hashlib.sha256()]
    counts = [0, 0]
    for rows in users:
        for index, partition in enumerate(split_user_rows(rows, split)):
            for row in partition:
                payload = json.dumps([np.asarray(v).tolist() for v in row], separators=(",", ":"))
                hashes[index].update((payload + "\n").encode())
                counts[index] += 1
    return dict(train_fingerprint=hashes[0].hexdigest(), test_fingerprint=hashes[1].hexdigest(),
                train_rows=counts[0], test_rows=counts[1])
