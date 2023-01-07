
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType
from utility.functions import clean_text_up_udf
from pyspark.sql.functions import col

from utility.functions import extract_rating_udf

spark = SparkSession.builder \
      .master("local[2]") \
      .appName("dataCleaning.com") \
      .getOrCreate()


df = spark.read.parquet("../tmp/df_with_id")

rating_df = df.withColumn("rating", extract_rating_udf("review"))
filtered_df = rating_df.filter(col("rating").isNotNull())
filtered_df.show()

filtered_df.write.mode("overwrite").parquet("D:/progetti/progetto/tmp/rating_df")