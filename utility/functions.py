from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from pyspark.sql.functions import udf
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
import re
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


def wordCount(wordListDF):
    return (wordListDF.groupBy('word').count())

wordCountUDF = udf(lambda x : wordCount(x), IntegerType())