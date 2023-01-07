from pyspark.sql import SparkSession
from pyspark.sql.functions import desc, first, udf, StringType, split, explode, collect_list, regexp_replace, col, \
    array, expr, size, length
from utility.functions import wordCount

spark = SparkSession.builder \
      .master("local[2]") \
      .appName("top_20_words_negative.com") \
      .getOrCreate()


#Read the clean df
only_negative_df = spark.read.parquet("../tmp/only_negative")

#def top_20_words_df():
shakeWordsSplitDF = (only_negative_df
                        .select(split(only_negative_df.review, '\s+').alias('split')))
shakeWordsSingleDF = (shakeWordsSplitDF
                        .select(explode(shakeWordsSplitDF.split).alias('word')))
shakeWordsDF = shakeWordsSingleDF.where(shakeWordsSingleDF.word != '')
#shakeWordsDF.persist()
#shakeWordsDF.show()

shakeWordsDFCount = shakeWordsDF.count()
WordsAndCountsDF = wordCount(shakeWordsDF)
WordsAndCountsDF

top_20_words_negative = WordsAndCountsDF.orderBy("count", ascending=0).limit(20).cache()
top_20_words_negative.write.mode("overwrite").parquet("D:/progetti/progetto/tmp/top_20_words_negative")