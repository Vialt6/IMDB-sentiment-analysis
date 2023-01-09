from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import CountVectorizer, Tokenizer
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, LogisticRegressionModel
from pyspark.ml.pipeline import PipelineModel
spark = SparkSession.builder \
      .master("local[2]") \
      .appName("top_20_bigrams.com") \
      .getOrCreate()

clean_df = spark.read.parquet("../tmp/clean_df")

clean_df = clean_df.withColumn('sentiment', regexp_replace('sentiment', 'positive', '1'))
clean_df = clean_df.withColumn('sentiment', regexp_replace('sentiment', 'negative', '0'))
clean_df = clean_df.withColumn("sentiment", clean_df.sentiment.cast(IntegerType()))


tokenizer = Tokenizer(inputCol="review", outputCol="tokenized_review")
tokenizedTrain = tokenizer.transform(clean_df)
#tokenizedTrain.show(truncate=False, n=5)

cv = CountVectorizer(inputCol="tokenized_review", outputCol="features")
model = cv.fit(tokenizedTrain)
result = model.transform(tokenizedTrain)
#result.show(truncate=False)

split_data = result.randomSplit([0.7,0.3])
train_set = split_data[0]
test_set = split_data[1]

lr = LogisticRegression(labelCol="sentiment", featuresCol="features", maxIter=10, regParam=0.01)

#model = lr.fit(train_set)
path_lr = "D:/progetti/progetto/models/logreg"
#model.save(path_lr)
model = LogisticRegressionModel.load(path_lr)
prediction = model.transform(test_set)
prediction = prediction.select("prediction","sentiment")
prediction.show(5)

Total_True=prediction.filter(prediction['prediction']==prediction['sentiment']).count()
Alldata=prediction.count()
Accuracy=Total_True/Alldata
print("Accuracy Score is:", Accuracy*100, '%')


"""rf = RandomForestClassifier(featuresCol = 'sentiment', labelCol = 'features')
rfModel = rf.fit(train_set)
rfModel.save("D:/progetti/progetto/models/logreg")
prediction_rf = rfModel.transform(test_set)
prediction_rf = prediction_rf.select("prediction","sentiment")
prediction_rf.show()

Total_True=prediction_rf.filter(prediction_rf['prediction']==prediction_rf['sentiment']).count()
Alldata=prediction_rf.count()
Accuracy_rf=Total_True/Alldata
print("Accuracy Score is:", Accuracy_rf*100, '%')"""