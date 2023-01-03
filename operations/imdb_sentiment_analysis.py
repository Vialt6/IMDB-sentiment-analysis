from io import BytesIO

import ngram as ngram
import nltk
from matplotlib import pyplot as plt
from pyspark.sql.functions import desc, first, udf, StringType, split, explode, collect_list, regexp_replace, col, \
    array, expr, size, length
from pyspark.sql import SparkSession
import re
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from pyspark.sql.types import StructType, StructField, IntegerType
import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd
import seaborn as sns
from pyspark.ml.feature import Tokenizer
from pyspark.ml.feature import NGram
from wordcloud import WordCloud
import base64

spark = SparkSession.builder \
      .master("local[2]") \
      .appName("SparkByExamples.com") \
      .getOrCreate()

#Reading dataset
movie_df = pd.read_csv('../IMDB Dataset.csv')
classNameContent = StructType([StructField("review", StringType(), True),
                               StructField("sentiment",  StringType(), True)])
df = spark.createDataFrame(movie_df, classNameContent)
df.createTempView("MovieReviews")
#df.cache()
df.show()

#df.groupby("sentiment").count().show()

#Define some functions for "review" column

#Removing html tags from review
def html_parser(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

html_parser_udf = udf(lambda x : html_parser(x), StringType())

#removing text beetwen square brackets
def remove_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

remove_square_brackets_udf = udf(lambda x : remove_square_brackets(x), StringType())

#removing url
def remove_url(text):
    return re.sub(r'http\S+', '', text)

remove_url_udf = udf(lambda x : remove_url(x), StringType())


stop = set(stopwords.words("english"))

#removing stopwards
def remove_stopwords(text):
    final_text = []
    for i in text.split():
        if i.strip().lower() not in stop and i.strip().lower().isalpha():
            final_text.append(i.strip().lower())
    return " ".join(final_text)
remove_stopwords_udf = udf(lambda x : remove_stopwords(x), StringType())

#applying all the cleaning functions
def clean_text(text):
    text = html_parser(text)
    text = remove_square_brackets(text)
    text = remove_url(text)
    text = remove_stopwords(text)
    return text
clean_text_udf = udf(lambda x : clean_text(x), StringType())

#Clean_df is the dataset after cleaning review column
clean_df = df.select(clean_text_udf("review").alias("review"), "sentiment")
#clean_df.cache()
#clean_df.persist()
clean_df.show()


#Word count

#Common words in text
#applying split funcion and explode to extract words from each column to see the Top words

def wordCount(wordListDF):
    return (wordListDF.groupBy('word').count())

wordCountUDF = udf(lambda x : wordCount(x), IntegerType())

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

top_20_words = WordsAndCountsDF.orderBy("count", ascending=0).limit(20)
#top_20_words.cache()
#top_20_words.show()

#Top 20 bigram: a pair of consecutive written units such as letters, syllables, or words.
#using the pyspark.ml library "ngram" for feature extraction

#First tokenize the clean_df. Sample output: [a, b, c,...]
#                                            [d, e, f...]
tokenizer = Tokenizer(inputCol="review", outputCol="word")
vector_df = tokenizer.transform(clean_df).select("word")
#vector_df.show()

ngram = NGram(n=2, inputCol="word", outputCol="bigrams")
#Df with 2 columns "Vector" and "word"
bigramDataFrame = ngram.transform(vector_df)
bigrams_df=bigramDataFrame.select(explode("bigrams").alias("bigram")).groupBy("bigram").count()
top_20_bigrams = bigrams_df.orderBy("count", ascending=0).limit(20)
#top_20_bigrams.cache()
#top_20_bigrams.show()

#Top 20 trigram: a group of three consecutive written units such as letters, syllables, or words.
#Same as top 20 bigram

ngram = NGram(n=3, inputCol="word", outputCol="trigrams")
trigramDataFrame = ngram.transform(vector_df)
trigrams_df=trigramDataFrame.select(explode("trigrams").alias("trigram")).groupBy("trigram").count()
top_20_trigrams = trigrams_df.orderBy("count", ascending=0).limit(20)
#top_20_trigrams.cache()
#top_20_trigrams.show()

#Only positive and only negative df
only_positive_df = clean_df.filter(clean_df.sentiment == "positive")
only_negative_df = clean_df.filter(clean_df.sentiment == "negative")


#number of words and characters in positive reviews
num_of_wordP = (only_positive_df.withColumn('wordCount', size(split(col("review"),' '))))
num_of_charsP = only_positive_df.withColumn("number_of_character", length("review"))
#num_of_wordP.sort(f.desc("wordCount")).show(20)


#number of words and characters in negative reviews
num_of_wordN = (only_negative_df.withColumn('wordCount',size(split(col("review"),' '))))
num_of_charsN = only_negative_df.withColumn("number_of_character", length("review"))
#num_of_wordN.sort(f.desc("wordCount")).show(20)






#Convert the "spark df" to a pandas df
pdf1 = df.toPandas()
clean_df_pandas=clean_df.toPandas()
only_positive_pandas = only_positive_df.toPandas()
only_negative_pandas = only_negative_df.toPandas()
top_20_words_pandas = top_20_words.toPandas()
top_20_bigrams_pandas=top_20_bigrams.toPandas()
top_20_trigrams_pandas=top_20_trigrams.toPandas()
word_countPositive_pd = num_of_wordP.toPandas()
word_countNegative_pd = num_of_wordN.toPandas()
chars_countPositive_pd = num_of_charsP.toPandas()
chars_countNegative_pd = num_of_charsN.toPandas()

#Dash app
app = dash.Dash()

#Figure section

#First histogram
fig = px.histogram(
    pdf1,
    x="sentiment",
    width=700, height=700
    #size_max=60
)

#Barplot top 20 words
fig_top_20_words = px.bar(
    top_20_words_pandas,
    x="count",
    y="word",
    title='Top 20 words in Text',
    orientation='h',
    width=700, height=700,
    color='word'
)

#Barpolot top 20 bigrams
fig_top_20_bigrams = px.bar(
    top_20_bigrams_pandas,
    x="count",
    y="bigram",
    title='Top 20 bigrams in Text',
    orientation='h',
    width=700, height=700,
    color='bigram'
)

#Barplot top 20 trigrams
fig_top_20_trigrams = px.bar(
    top_20_trigrams_pandas,
    x="count",
    y="trigram",
    title='Top 20 trigrams in Text',
    orientation='h',
    width=700, height=700,
    color='trigram'
)

fig_count_positive = px.histogram(
    word_countPositive_pd,
    title='Number of words in positive reviews',
    x="wordCount",
    width=700, height=700
)


fig_count_negative = px.histogram(
    word_countNegative_pd,
    title='Number of words in negative reviews',
    x="wordCount",
    width=700, height=700
)

fig_chararcter_positive = px.histogram(
    chars_countPositive_pd,
    title='Number of characters in negative reviews',
    x="number_of_character",
    width=700, height=700
)

fig_chararcter_negative = px.histogram(
    chars_countNegative_pd,
    title='Number of characters in negative reviews',
    x="number_of_character",
    width=700, height=700
)




#Making wordcloud for positive and negative
def plot_wordcloud(df,title,column):
    plt.figure(figsize = (20,20))
    wc = WordCloud(max_words = 500 , width = 800 , height = 300).generate(" ".join(df[df.sentiment == column].review))
    plt.title(title, fontsize=13)
    return wc.to_image()

#need to convert to an img
@app.callback(dash.dependencies.Output('image_wc_positive', 'src'), [dash.dependencies.Input('image_wc_positive','id')])
def make_image_positive_wc(b):
    img=BytesIO()
    plot_wordcloud(df=only_positive_pandas,title="Positive Wordcloud",column="positive").save(img,format='PNG')
    return 'data:image/png;base64,{}'.format(base64.b64encode(img.getvalue()).decode())

@app.callback(dash.dependencies.Output('image_wc_negative', 'src'), [dash.dependencies.Input('image_wc_negative','id')])
def make_image_negative_wc(b):
    img=BytesIO()
    plot_wordcloud(df=only_negative_pandas,title="Negative Wordcloud",column="negative").save(img,format='PNG')
    return 'data:image/png;base64,{}'.format(base64.b64encode(img.getvalue()).decode())



#Dash Layout section
app.layout=html.Div(children = [
    html.H1(children='IMDB Dashboard'),
    dcc.Graph(id="sentiment", figure=fig),
    dcc.Graph(id="words", figure=fig_top_20_words),
    dcc.Graph(id="bigrams", figure=fig_top_20_bigrams),
    dcc.Graph(id="trigrams", figure=fig_top_20_trigrams),
    html.Img(id="image_wc_positive"),
    html.Img(id="image_wc_negative"),
    dcc.Graph(id="word_count_positive", figure=fig_count_positive),
    dcc.Graph(id="word_count_negative", figure=fig_count_negative),
    dcc.Graph(id="charsCountPositive", figure=fig_chararcter_positive),
    dcc.Graph(id="charsCountNegative", figure=fig_chararcter_negative)
])

if __name__ == "__main__":
    app.run_server(debug=True)