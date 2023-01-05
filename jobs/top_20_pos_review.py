from pyspark.sql import SparkSession

spark = SparkSession.builder \
      .master("local[2]") \
      .appName("top_20_bigrams.com") \
      .getOrCreate()

#Define top 20 review based on positivity score
sentiment_score_df = spark.read.parquet("../tmp/sentiment_score")

top_20_pos_review = sentiment_score_df.orderBy(sentiment_score_df.pos.desc()).limit(20)
top_20_pos_review.show()

top_20_pos_review.write.mode("overwrite").parquet("D:/progetti/progetto/tmp/top_20_pos_review")