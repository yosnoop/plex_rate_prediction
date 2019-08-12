from plexapi.server import PlexServer
from plexapi.video import Movie
from plexapi.media import Director, Role
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pprint import pprint

import xgboost as xgb
import pandas as pd
import numpy as np


plex = PlexServer()
data = []


def find_movie_sections() -> list:
    result = []
    for section in plex.library.sections():
        if section.type == 'movie':
            result.append(section)

    return result


def get_filmograph(who, sections: list) -> list:
    result = []
    for section in sections:
        search = []
        if isinstance(who, Director):
            search = section.search(None, director=who)
        elif isinstance(who, Role):
            search = section.search(None, actor=who)
        for movie in search:
            result.append(movie)

    return result


def get_average_user_rating(movies: list) -> float:
    if len(movies) == 0:
        raise ValueError('list is empty')
    userRatings = 0
    for movie in movies:
        if movie.userRating is not None:
            userRatings += movie.userRating
    return userRatings / len(movies)


def audience_rating(item: Movie) -> float:
    return item.audienceRating if item.audienceRating else item.rating


def make_movie(item: Movie) -> list:
    audienceRating = audience_rating(item)
    return [
        item.title,
        int(item.rating * 10),
        int(audienceRating * 10),
        item.duration,
        item.year,
        [genre.tag for genre in item.genres],
    ]


def featurize_genre(df: pd.DataFrame) -> pd.DataFrame:
    df_genre = df['genres'].apply(frozenset).to_frame(name='genre')
    for genre in frozenset.union(*df_genre.genre):
        df_genre['genre_' + genre] = \
            df_genre.apply(lambda _: int(genre in _.genre), axis=1)

    return \
        pd.concat(
            [df.drop('genres', axis=1), df_genre.drop('genre', axis=1)],
            axis=1
        )


movie_sections = find_movie_sections()
for item in plex.library.all():
    if not isinstance(item, Movie):
        continue
    if item.rating is None:
        continue

    movie = make_movie(item)
    if item.viewCount > 0 and item.userRating is None:
        item.rate(min((item.viewCount + 1) * 2.0, 10.0))

    userRating = int(item.userRating * 10) if item.userRating else np.nan
    movie.append(userRating)
    '''
    directorRating = item.userRating
    if len(item.directors) > 0:
        filmograph = get_filmograph(item.directors[0], movie_sections)
        if len(filmograph) > 0:
            directorRating = get_average_user_rating(filmograph)
    actorRating = item.userRating
    if len(item.roles) > 0:
        filmograph = get_filmograph(item.roles[0], movie_sections)
        if len(filmograph) > 0:
            actorRating = get_average_user_rating(filmograph)
    '''
    data.append(movie)

df = pd.DataFrame(
                    [movie[1:] for movie in data],
                    columns=[
                        'rating',
                        'audienceRating',
                        'duration',
                        'year',
                        'genres',
                        'userRating',
                        # 'directorRating',
                        # 'actorRating',
                    ]
                )

df = featurize_genre(df)
df_seen = df.loc[df['userRating'].notna()]
df_unseen = df.loc[df['userRating'].isna()].drop('userRating', axis=1)
X = df_seen.drop('userRating', axis=1)
Y = df_seen.userRating

seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = \
    train_test_split(X, Y, test_size=test_size, random_state=seed)
# fit model no training data
model = xgb.XGBClassifier()
model.fit(X_train, y_train)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

model.fit(X, Y)
y_pred = model.predict(df_unseen)
df_prediction = \
    pd.DataFrame(y_pred, columns=['prediction'], index=df_unseen.index)
titles = [data[index][0] for index in df_unseen.index]
df_title = \
    pd.DataFrame(titles, columns=['title'], index=df_unseen.index)

result = pd.concat([df_title, df_unseen, df_prediction], axis=1)
result = result.sort_values(by=['prediction'], ascending=False)
pprint(result)
