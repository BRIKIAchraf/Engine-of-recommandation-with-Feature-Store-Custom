from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Int32, Float32, Int64, String

user = Entity(name="user_id", join_keys=["user_id"], value_type=Int64)
movie = Entity(name="movie_id", join_keys=["movie_id"], value_type=Int64)

user_source = FileSource(
    path="/home/jovyan/work/data/features/user_features",
    timestamp_field="event_timestamp",
)

movie_source = FileSource(
    path="/home/jovyan/work/data/features/movie_features",
    timestamp_field="event_timestamp",
)

user_fv = FeatureView(
    name="user_features",
    entities=[user],
    ttl=timedelta(days=36500), 
    schema=[
        Field(name="gender", dtype=String),
        Field(name="age", dtype=Int64),
        Field(name="occupation", dtype=Int64),
        Field(name="zip", dtype=String),
        Field(name="user_avg_rating", dtype=Float32),
        Field(name="user_nb_movies", dtype=Int64),
        Field(name="user_rating_stddev", dtype=Float32),
    ],
    source=user_source,
)

movie_fv = FeatureView(
    name="movie_features",
    entities=[movie],
    ttl=timedelta(days=36500),
    schema=[
        Field(name="title", dtype=String),
        Field(name="genres", dtype=String),
        Field(name="release_year", dtype=Int32),
        Field(name="movie_avg_rating", dtype=Float32),
        Field(name="movie_total_views", dtype=Int64),
        Field(name="movie_rating_stddev", dtype=Float32),
    ],
    source=movie_source,
)
