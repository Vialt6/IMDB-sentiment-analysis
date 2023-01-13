import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import row_number, lit
from pyspark.sql.types import StructType, StructField, StringType
from utility.functions import clean_text_up_udf
from pyspark.sql.window import Window

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

#Adding Id column to initial df
df_id = df.withColumn("id", row_number().over(Window.partitionBy(lit('')).orderBy(lit(''))))

df_id.write.mode("overwrite").csv("D:/progetti/progetto/file/df_with_id")