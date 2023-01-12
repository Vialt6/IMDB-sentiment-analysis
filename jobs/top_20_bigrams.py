from pyspark.ml.feature import Tokenizer
from pyspark.ml.feature import NGram
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode

spark = SparkSession.builder \
      .master("local[2]") \
      .appName("top_20_bigrams.com") \
      .getOrCreate()

clean_df = spark.read.parquet("../tmp/clean_df")


#Top 20 bigram: a pair of consecutive written units such as letters, syllables, or words.
#using the pyspark.ml library "ngram" for feature extraction

#First tokenize the clean_df. Sample output: [a, b, c,...]
#                                            [d, e, f...]
tokenizer = Tokenizer(inputCol="review", outputCol="word")
vector_df = tokenizer.transform(clean_df).select("word")
vector_df.show()

ngram = NGram(n=2, inputCol="word", outputCol="bigrams")
#Df with 2 columns "Vector" and "word"
bigramDataFrame = ngram.transform(vector_df)
bigrams_df=bigramDataFrame.select(explode("bigrams").alias("bigram")).groupBy("bigram").count()
top_20_bigrams = bigrams_df.orderBy("count", ascending=0).limit(20).cache()

top_20_bigrams.show()

#top_20_bigrams.write.mode("overwrite").parquet("D:/progetti/progetto/tmp/top_20_bigrams")