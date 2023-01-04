from pyspark.sql import SparkSession
from pyspark.sql.functions import desc, first, udf, StringType, split, explode, collect_list, regexp_replace, col, \
    array, expr, size, length

spark = SparkSession.builder \
      .master("local[2]") \
      .appName("top_20_bigrams.com") \
      .getOrCreate()

clean_df = spark.read.parquet("../tmp/clean_df")

only_negative_df = clean_df.filter(clean_df.sentiment == "negative")
#only_negative_df.write.mode("overwrite").parquet("D:/progetti/progetto/tmp/only_negative")

num_of_charsN = only_negative_df.withColumn("number_of_character", length("review"))
num_of_charsN.show()
#num_of_charsN.write.mode("overwrite").parquet("D:/progetti/progetto/tmp/num_chars_negative")