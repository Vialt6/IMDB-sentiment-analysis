from pyspark.sql import SparkSession
from pyspark.sql.functions import desc, first, udf, StringType, split, explode, collect_list, regexp_replace, col, \
    array, expr, size, length
from utility.functions import wordCount

spark = SparkSession.builder \
      .master("local[2]") \
      .appName("top_20_words.com") \
      .getOrCreate()


#Read the clean df
clean_df = spark.read.parquet("../tmp/clean_df")

#def top_20_words_df():
shakeWordsSplitDF = (clean_df
                        .select(split(clean_df.review, '\s+').alias('split')))
shakeWordsSingleDF = (shakeWordsSplitDF
                        .select(explode(shakeWordsSplitDF.split).alias('word')))
shakeWordsDF = shakeWordsSingleDF.where(shakeWordsSingleDF.word != '')
#shakeWordsDF.persist()
#shakeWordsDF.show()

shakeWordsDFCount = shakeWordsDF.count()
WordsAndCountsDF = wordCount(shakeWordsDF)
WordsAndCountsDF

top_20_words = WordsAndCountsDF.orderBy("count", ascending=0).limit(20).cache()
#top_20_words.cache()

#top_20_words.write.mode("overwrite").parquet("D:/progetti/progetto/tmp/top_20_words")