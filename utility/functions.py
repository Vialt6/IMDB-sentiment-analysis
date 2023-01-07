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

#removing stopwards and set text to lowercase
def remove_stopwords(text):
    final_text = []
    for i in text.split():
        if i.strip().lower() not in stop and i.strip().lower().isalpha():
            final_text.append(i.strip().lower())
    return " ".join(final_text)
remove_stopwords_udf = udf(lambda x : remove_stopwords(x), StringType())


#only remove stopwards
def remove_stopwords_up(text):
    final_text = []
    for i in text.split():
        if i.strip().lower() not in stop and i.strip().lower().isalpha():
            final_text.append(i.strip())
    return " ".join(final_text)



#applying all the cleaning functions
def clean_text(text):
    text = html_parser(text)
    text = remove_square_brackets(text)
    text = remove_url(text)
    text = remove_stopwords(text)
    return text
clean_text_udf = udf(lambda x : clean_text(x), StringType())


#applying all the cleaning functions with words up
def clean_text_up(text):
    text = html_parser(text)
    text = remove_square_brackets(text)
    text = remove_url(text)
    text = remove_stopwords_up(text)
    return text
clean_text_up_udf = udf(lambda x : clean_text_up(x), StringType())



def wordCount(wordListDF):
    return (wordListDF.groupBy('word').count())

wordCountUDF = udf(lambda x : wordCount(x), IntegerType())


def extract_rating(review):
    # Search for a string "X/10" in the review, Where X is a number between 1 and 10
    match = re.search(r'(\d+)/10', review)
    if match:
        # Estrai il numero X dalla stringa e converdilo in intero
        rating = int(match.group(1))
        if rating>=1 and rating<=10:
            return rating
    else:
        return None


# extract_rating_udf
extract_rating_udf = udf(extract_rating)

def extract_title(review):
  # Cerca le parole chiave "recensione di" o "recensione del" seguite da un titolo tra virgolette
  match = re.search(r'(watched) "(.+?)"', review)
  if match:
    # Estrai il titolo dalla stringa
    title = match.group(2)
    return title
  else:
    return None

# Crea una funzione utente (udf) da questa funzione
extract_title_udf = udf(extract_title)

