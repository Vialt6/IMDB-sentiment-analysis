from pyspark.sql import SparkSession
spark = SparkSession.builder \
      .master("local[2]") \
      .appName("top_20_bigrams.com") \
      .getOrCreate()


film_df = spark.read.parquet("../tmp/film_df")
sentiment_df = spark.read.parquet("../tmp/sentiment_score")

#join between the two df
film_sentiment_df = film_df.join(sentiment_df, on="id")
film_sentiment_df = film_sentiment_df.select("title","pos","neg","compound","id")
film_sentiment_df.show()
#film_sentiment_df.write.mode("overwrite").parquet("D:/progetti/progetto/tmp/film_sentiment_df")


#select the top 10 film based on positivity score
top_10_pos_film = film_sentiment_df.orderBy("compound", ascending=0).limit(10)
top_10_pos_film = top_10_pos_film.select("title","compound")
top_10_pos_film.show(truncate=False)



top_10_pos_film.write.mode("overwrite").parquet("D:/progetti/progetto/tmp/top_10_pos_film")