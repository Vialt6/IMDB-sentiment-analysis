import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType
from utility.functions import extract_director_udf
from pyspark.sql.functions import col
spark = SparkSession.builder \
      .master("local[2]") \
      .appName("top_20_bigrams.com") \
      .getOrCreate()


df = spark.read.parquet("../tmp/df_with_id")

dir_df = df.withColumn("director", extract_director_udf("review"))
filtered_df = dir_df.filter(col("director").isNotNull())
filtered_df.show()
pandasDF=filtered_df.toPandas()
print(pandasDF.shape)
filtered_df.write.mode("overwrite").parquet("D:/progetti/progetto/tmp/director_df")