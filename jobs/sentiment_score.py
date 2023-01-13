import nltk
import pandas as pd
from pyspark.sql import SparkSession
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pyspark.sql.functions import row_number, lit
from pyspark.sql.window import Window

spark = SparkSession.builder \
      .master("local[2]") \
      .appName("top_20_bigrams.com") \
      .getOrCreate()

#using nltk library to extract negative neutral and positive score based on the reviews

clean_df = spark.read.parquet("../tmp/clean_df")
clean_df=clean_df.toPandas()
#clean_df = clean_df.toPandas()
sid = SentimentIntensityAnalyzer()
clean_df["sentiment"] = clean_df["review"].apply(lambda x: sid.polarity_scores(x))
reviews_df = pd.concat([clean_df.drop(['sentiment'], axis=1), clean_df['sentiment'].apply(pd.Series)], axis=1)

sentiment_score = spark.createDataFrame(reviews_df)

#Add id column
sentiment_score = sentiment_score.withColumn("id", row_number().over(Window.partitionBy(lit('')).orderBy(lit(''))))
sentiment_score.show()
#sentiment_score.write.mode("overwrite").parquet("D:/progetti/progetto/tmp/sentiment_score")

