from pyspark.sql import SparkSession
from pyspark.sql.functions import monotonically_increasing_id, row_number, lit, desc
from pyspark.sql.window import Window

spark = SparkSession.builder \
      .master("local[2]") \
      .appName("top_20_bigrams.com") \
      .getOrCreate()

#Define top 20 review based on positivity score
sentiment_score_df = spark.read.parquet("../tmp/sentiment_score")

top_20_pos_review = sentiment_score_df.orderBy("pos", ascending=0).limit(20)
top_20_pos_review.write.mode("overwrite").parquet("D:/progetti/progetto/tmp/top_20_pos_review")