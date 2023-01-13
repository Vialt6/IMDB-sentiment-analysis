from pyspark.sql import SparkSession
from pyspark.sql.functions import desc, first, udf, StringType, split, explode, collect_list, regexp_replace, col, \
    array, expr, size, length

spark = SparkSession.builder \
      .master("local[2]") \
      .appName("top_20_bigrams.com") \
      .getOrCreate()

clean_df = spark.read.parquet("../tmp/clean_df")

only_positive_df = clean_df.filter(clean_df.sentiment == "positive")

#number of words in positive reviews
num_of_wordP = (only_positive_df.withColumn('wordCount', size(split(col("review"),' '))))

#num_of_wordP.write.mode("overwrite").parquet("D:/progetti/progetto/tmp/num_words_positive")