import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType
from utility.functions import clean_text_up_udf, extract_title_udf
from pyspark.sql.functions import col
spark = SparkSession.builder \
      .master("local[2]") \
      .appName("top_20_bigrams.com") \
      .getOrCreate()


df = spark.read.parquet("../tmp/df_with_id")
df.show()
film_df = df.withColumn("title", extract_title_udf("review"))
filtered_df = film_df.filter(col("title").isNotNull())
#filtered_df = filtered_df.filter(filtered_df.title != 'Films')
filtered_df.show()
#filtered_df.write.mode("overwrite").parquet("D:/progetti/progetto/tmp/film_df")