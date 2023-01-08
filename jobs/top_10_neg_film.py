from pyspark.sql import SparkSession
spark = SparkSession.builder \
      .master("local[2]") \
      .appName("top_20_bigrams.com") \
      .getOrCreate()

film_sentiment_df = spark.read.parquet("../tmp/film_sentiment_df")

#select the top 10 film based on negativity score
top_10_neg_film = film_sentiment_df.orderBy("neg", ascending=0).limit(10)
top_10_neg_film = top_10_neg_film.select("title","neg")
top_10_neg_film.show(truncate=False)