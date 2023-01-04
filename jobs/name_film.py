from pyspark.ml.feature import Tokenizer
from pyspark.ml.feature import NGram
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode

spark = SparkSession.builder \
      .master("local[2]") \
      .appName("top_20_bigrams.com") \
      .getOrCreate()

clean_df = spark.read.parquet("../tmp/clean_df")
pattern_rating = "\\(?[+-]?\\d+)?\\/\\d+\\)?" #"#?"+"[^.?!(]"+"\\(18|19|20)\\d{2}\\)"+ pattern_rating+"(#IMDb)?")
pattern= ""
film_df = clean_df.filter(clean_df.review.contains("watched"))
film_df.show()