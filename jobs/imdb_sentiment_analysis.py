from io import BytesIO
from matplotlib import pyplot as plt
from pyspark.sql.functions import desc, first, udf, StringType, split, explode, collect_list, regexp_replace, col, \
    array, expr, size, length
from pyspark.sql import SparkSession
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
#df.show()

#Read the clean df
clean_df = spark.read.parquet("../tmp/clean_df")
#clean_df.show()
#Word count

#Common words in text
#applying split funcion and explode to extract words from each column to see the Top words
top_20_words = spark.read.parquet("../tmp/top_20_words")


#Top 20 bigram: a pair of consecutive written units such as letters, syllables, or words.
#using the pyspark.ml library "ngram" for feature extraction
top_20_bigrams = spark.read.parquet("../tmp/top_20_bigrams")

#Top 20 trigram: a group of three consecutive written units such as letters, syllables, or words.
#Same as top 20 bigram
top_20_trigrams = spark.read.parquet("../tmp/top_20_trigrams")

#Only positive and onnly negative
only_positive_df = spark.read.parquet("../tmp/only_positive")
only_negative_df = spark.read.parquet("../tmp/only_negative")

#number of words and characters in positive reviews
num_of_wordP = spark.read.parquet("../tmp/num_words_positive")
num_of_charsP = spark.read.parquet("../tmp/num_chars_positive")
#num_of_wordP.sort(f.desc("wordCount")).show(20)


#number of words and characters in negative reviews
num_of_wordN = spark.read.parquet("../tmp/num_words_negative")
num_of_charsN = spark.read.parquet("../tmp/num_chars_negative")
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
    wc = WordCloud(max_words = 500 , width = 800 , height = 300).generate(" ".join(df.review))
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