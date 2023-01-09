from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import CountVectorizer, Tokenizer
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, LogisticRegressionModel, NaiveBayes, RandomForestClassificationModel, \
    NaiveBayesModel
from pyspark.mllib.evaluation import MulticlassMetrics, BinaryClassificationMetrics

spark = SparkSession.builder \
      .master("local[2]") \
      .appName("top_20_bigrams.com") \
      .getOrCreate()

clean_df = spark.read.parquet("../tmp/clean_df")

clean_df = clean_df.withColumn('sentiment', regexp_replace('sentiment', 'positive', '1'))
clean_df = clean_df.withColumn('sentiment', regexp_replace('sentiment', 'negative', '0'))
clean_df = clean_df.withColumn("sentiment", clean_df.sentiment.cast(IntegerType()))
clean_df = clean_df.withColumnRenamed("sentiment","label")

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

lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=10, regParam=0.01)

#model = lr.fit(train_set)
path_lr = "D:/progetti/progetto/models/logreg"
#model.save(path_lr)
model = LogisticRegressionModel.load(path_lr)
lr_prediction = model.transform(test_set)
lr_prediction = lr_prediction.select("prediction","label")
lr_prediction.show(5)

evaluatorMulti = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction", metricName='areaUnderROC')
lr_accuracy= evaluatorMulti.evaluate(lr_prediction, {evaluatorMulti.metricName: "accuracy"})
lr_f1 = evaluatorMulti.evaluate(lr_prediction, {evaluatorMulti.metricName: "f1"})
lr_weightedPrecision = evaluatorMulti.evaluate(lr_prediction, {evaluatorMulti.metricName: "weightedPrecision"})
lr_weightedRecall = evaluatorMulti.evaluate(lr_prediction, {evaluatorMulti.metricName: "weightedRecall"})
lr_accuracy = round(lr_accuracy,2)
lr_f1 = round(lr_f1,2)
lr_weightedRecall = round(lr_weightedRecall,2)
lr_weightedPrecision = round(lr_weightedPrecision,2)

print(lr_accuracy)
print(lr_f1)
print(lr_weightedPrecision)
print(lr_weightedRecall)
print("Accuracy Score is:", lr_accuracy*100, '%')


nb = NaiveBayes(labelCol="label",featuresCol="features", smoothing=1)
path_nb = "D:/progetti/progetto/models/naiveBayes"
#nbModel = nb.fit(train_set)
#nbModel.save(path_nb)
nbModel = NaiveBayesModel.load(path_nb)
predictions = nbModel.transform(test_set)
prediction = predictions.select("label","prediction")
#prediction.show()

evaluatorMulti = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction", metricName='areaUnderROC')
nb_accuracy= evaluatorMulti.evaluate(prediction, {evaluatorMulti.metricName: "accuracy"})
nb_f1 = evaluatorMulti.evaluate(prediction, {evaluatorMulti.metricName: "f1"})
nb_weightedPrecision = evaluatorMulti.evaluate(prediction, {evaluatorMulti.metricName: "weightedPrecision"})
nb_weightedRecall = evaluatorMulti.evaluate(prediction, {evaluatorMulti.metricName: "weightedRecall"})
nb_accuracy = round(nb_accuracy,2)
nb_f1 = round(nb_f1,2)
nb_weightedRecall = round(nb_weightedRecall,2)
nb_weightedPrecision = round(nb_weightedPrecision,2)

print(nb_accuracy)
print(nb_f1)
print(nb_weightedPrecision)
print(nb_weightedRecall)


from pyspark.ml.classification import RandomForestClassifier
rf = RandomForestClassifier(labelCol="label", \
                            featuresCol="features", \
                            numTrees = 100, \
                            maxDepth = 4, \
                            maxBins = 32)

path_rf = "D:/progetti/progetto/models/random_forest"
# Train model with Training Data
#rfModel = rf.fit(train_set)
#rfModel.save(path_rf)
rfModel = RandomForestClassificationModel.load(path_rf)
predictions = rfModel.transform(test_set)
rf_prediction = predictions.select("label","prediction")
evaluatorMulti = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction", metricName='areaUnderROC')
rf_accuracy= evaluatorMulti.evaluate(rf_prediction, {evaluatorMulti.metricName: "accuracy"})
rf_f1 = evaluatorMulti.evaluate(rf_prediction, {evaluatorMulti.metricName: "f1"})
rf_weightedPrecision = evaluatorMulti.evaluate(rf_prediction, {evaluatorMulti.metricName: "weightedPrecision"})
rf_weightedRecall = evaluatorMulti.evaluate(rf_prediction, {evaluatorMulti.metricName: "weightedRecall"})
rf_accuracy = round(rf_accuracy,2)
rf_f1 = round(rf_f1,2)
rf_weightedRecall = round(rf_weightedRecall,2)
rf_weightedPrecision = round(rf_weightedPrecision,2)

print(rf_accuracy)
print(rf_f1)
print(rf_weightedPrecision)
print(rf_weightedRecall)

classifiers = ["Logistic Regression", "Naive Bayes", "Random Forest"]
log_reg_score = [lr_accuracy, lr_f1, lr_weightedPrecision, lr_weightedRecall]
nb_score = [nb_accuracy, nb_f1, nb_weightedPrecision, nb_weightedRecall]
rf_score = [rf_accuracy, rf_f1, rf_weightedPrecision, rf_weightedRecall]

#Creating list for result dataframe
accuracy_list = [lr_accuracy, nb_accuracy, rf_accuracy]
f1_list = [lr_f1, nb_f1, rf_f1]
precision_list = [lr_weightedPrecision, nb_weightedPrecision, rf_weightedPrecision]
recall_list = [lr_weightedRecall, nb_weightedRecall, rf_weightedRecall]
#Creating dataframe for results
columns = ["classifier","accuracy","f1","precision","recall"]
result_df = spark.createDataFrame(zip(classifiers, accuracy_list,f1_list,precision_list,recall_list), columns)

result_df.show()

result_df.write.mode("overwrite").parquet("D:/progetti/progetto/tmp/result_model_df")