import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count, stddev, max as spark_max, to_timestamp, lit, udf
from pyspark.sql.types import IntegerType, FloatType, TimestampType, StringType
import re

def extract_year(title):
    match = re.search(r'\((\d{4})\)', title)
    return int(match.group(1)) if match else 1900

def compute_features():
    spark = SparkSession.builder \
        .appName("FeatureCalculation") \
        .getOrCreate()
        
    raw_dir = "data/raw/ml-1m"
    features_dir = "data/features"
    os.makedirs(features_dir, exist_ok=True)
    
    # Schemas
    ratings_rdd = spark.sparkContext.textFile(f"{raw_dir}/ratings.dat") \
        .map(lambda line: line.split("::")) \
        .map(lambda p: (int(p[0]), int(p[1]), float(p[2]), int(p[3])))
        
    ratings = spark.createDataFrame(ratings_rdd, ["user_id", "movie_id", "rating", "timestamp"])
    
    users_rdd = spark.sparkContext.textFile(f"{raw_dir}/users.dat") \
        .map(lambda line: line.split("::")) \
        .map(lambda p: (int(p[0]), p[1], int(p[2]), int(p[3]), p[4]))
        
    users = spark.createDataFrame(users_rdd, ["user_id", "gender", "age", "occupation", "zip"])
    
    movies_rdd = spark.sparkContext.textFile(f"{raw_dir}/movies.dat") \
        .map(lambda line: line.split("::")) \
        .map(lambda p: (int(p[0]), p[1], p[2]))
        
    movies = spark.createDataFrame(movies_rdd, ["movie_id", "title", "genres"])

    # 1. User Features
    user_stats = ratings.groupBy("user_id").agg(
        avg("rating").alias("user_avg_rating"),
        count("movie_id").alias("user_nb_movies"),
        stddev("rating").alias("user_rating_stddev"),
        spark_max("timestamp").alias("max_ts")
    )
    user_stats = user_stats.fillna(0.0, subset=["user_rating_stddev"])
    
    user_features = users.join(user_stats, on="user_id", how="left")
    user_features = user_features.withColumn("event_timestamp", to_timestamp(col("max_ts")))
    user_features = user_features.drop("max_ts")
    
    # 2. Movie Features
    movie_stats = ratings.groupBy("movie_id").agg(
        avg("rating").alias("movie_avg_rating"),
        count("user_id").alias("movie_total_views"),
        stddev("rating").alias("movie_rating_stddev"),
        spark_max("timestamp").alias("max_ts")
    )
    movie_stats = movie_stats.fillna(0.0, subset=["movie_rating_stddev"])
    
    extract_year_udf = udf(extract_year, IntegerType())
    movies = movies.withColumn("release_year", extract_year_udf(col("title")))
    
    movie_features = movies.join(movie_stats, on="movie_id", how="left")
    movie_features = movie_features.withColumn("event_timestamp", to_timestamp(col("max_ts")))
    movie_features = movie_features.drop("max_ts")
    
    # Save to Parquet
    user_features.write.mode("overwrite") \
        .partitionBy("event_timestamp") \
        .parquet(f"{features_dir}/user_features")
        
    movie_features.write.mode("overwrite") \
        .partitionBy("event_timestamp") \
        .parquet(f"{features_dir}/movie_features")
        
    print("Features calculées et sauvegardées en Parquet.")

if __name__ == "__main__":
    compute_features()
