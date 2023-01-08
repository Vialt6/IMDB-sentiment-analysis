from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from utility.functions import extract_rating_udf

spark = SparkSession.builder \
      .master("local[2]") \
      .appName("dataCleaning.com") \
      .getOrCreate()

film_df = spark.read.parquet("../tmp/film_df")
rating_df = spark.read.parquet("../tmp/rating_df")


film_rating_df = film_df.join(rating_df, on="id")
film_rating_df = film_rating_df.select("title","rating","id")
rating_film_10 = film_rating_df.filter(col("rating") == 10)
rating_film_10.show()

rating_film_1 = film_rating_df.filter(col("rating") == 1)
rating_film_1.show()