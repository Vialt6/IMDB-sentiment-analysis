import re

import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import StructType, StructField, StringType
from utility.functions import clean_text_udf

spark = SparkSession.builder \
      .master("local[2]") \
      .appName("dataCleaning.com") \
      .getOrCreate()


#Reading dataset
movie_df = pd.read_csv('../IMDB Dataset.csv')
classNameContent = StructType([StructField("review", StringType(), True),
                               StructField("sentiment",  StringType(), True)])
df = spark.createDataFrame(movie_df, classNameContent)
df.createTempView("MovieReviews")
#df.cache()
#df.show()


#Clean_df is the dataset after cleaning review column
clean_df = df.select(clean_text_udf("review").alias("review"), "sentiment")
#clean_df.cache()
#clean_df.persist()
clean_df.show()

#clean_df.write.mode("overwrite").parquet("tmp/clean_df")