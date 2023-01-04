from pyspark.ml.feature import Tokenizer
from pyspark.ml.feature import NGram
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode

spark = SparkSession.builder \
      .master("local[2]") \
      .appName("top_20_trigrams.com") \
      .getOrCreate()


#Read the clean df
clean_df = spark.read.parquet("../tmp/clean_df")



#First tokenize the clean_df. Sample output: [a, b, c,...]
#                                            [d, e, f...]
tokenizer = Tokenizer(inputCol="review", outputCol="word")
vector_df = tokenizer.transform(clean_df).select("word")

#Top 20 trigram: a group of three consecutive written units such as letters, syllables, or words.
#Same as top 20 bigram

ngram = NGram(n=3, inputCol="word", outputCol="trigrams")
trigramDataFrame = ngram.transform(vector_df)
trigrams_df=trigramDataFrame.select(explode("trigrams").alias("trigram")).groupBy("trigram").count()
top_20_trigrams = trigrams_df.orderBy("count", ascending=0).limit(20)
#top_20_trigrams.cache()
#top_20_trigrams.show()

top_20_trigrams.write.mode("overwrite").parquet("D:/progetti/progetto/tmp/top_20_trigrams")