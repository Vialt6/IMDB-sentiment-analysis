from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType
from utility.functions import extract_director_udf
from pyspark.sql.functions import col
spark = SparkSession.builder \
      .master("local[2]") \
      .appName("top_20_bigrams.com") \
      .getOrCreate()


director_df = spark.read.parquet("../tmp/director_df")
sentiment_df = spark.read.parquet("../tmp/sentiment_score")
director_sentiment_df = director_df.join(sentiment_df, on="id")
director_sentiment_df = director_sentiment_df.select("director","pos","neg","id")
director_sentiment_df.show()
#director_sentiment_df.write.mode("overwrite").parquet("D:/progetti/progetto/tmp/director_sentiment_df")


top_10_pos_director = director_sentiment_df.orderBy("neg", ascending=0).limit(10)
top_10_pos_director = top_10_pos_director.select("director","neg")
top_10_pos_director.show(truncate=False)

#top_10_pos_director.write.mode("overwrite").parquet("D:/progetti/progetto/tmp/top_10_pos_director")