#aqui va cargar la data, quitar la columna anio y la vectorizacion
import pandas as pd
import re
from nltk.corpus import stopwords
import gensim
from collections import Counter
import spacy
import gensim
# cambiador de formatos para factorizar un dataset
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
from nltk.tokenize import word_tokenize
warnings.filterwarnings('ignore')

spanish_stopwords = list(stopwords.words('spanish'))
spanish_stopwords.extend(("xq", "oye", "dale", "dele", "ba", "abc", "nan", "na"))
# print(spanish_stopwords)
def cargar_data(archivo):
    try:
        datos = pd.read_csv(archivo)
        return datos
    except Exception as e:
        return f"Error al cargar el archivo: {e}"

def quitar_columnas(data):
    columns_to_keep = ["TEXTO_TOKEN", "LEMMA"]
    for column in data.columns:
        if column not in columns_to_keep:
            data = data.drop(column, axis=1)
    return data

def elegir_columna(data, column):
    if isinstance(column, int):
        return data.iloc[:, column]
    elif isinstance(column, str):
        return data[column]
    else:
        return None

def limpiar_tokenizar(texto):
    texto = re.sub(r'\W', ' ', str(texto))
    tokens = word_tokenize(texto.lower())
    return tokens

def eliminar_vacios(data):
    return data.dropna()
def lemmatize_text(data, column):
    nlp = spacy.load("es_core_news_sm")
    data['LEMMA'] = data['TEXTO_TOKEN'].apply(lambda x: ' '.join(x))
    data['LEMMA'] = data['LEMMA'].apply(lambda x: [token.lemma_ for token in nlp(x) if not token.is_stop])
    return data

def process_words(data, column):
    WORDS = Counter(words(open('big.txt', encoding='utf-8').read()))
    data['CORRECION'] = data['LEMMA'].apply(lambda x: process_words_list(x, WORDS))
    return data

def lda_gensim(data, column, cluster):
    processed_docs = [doc for doc in data['CORRECION']]
    dictionary = gensim.corpora.Dictionary(processed_docs)
    dictionary.filter_extremes(no_below=15, no_above=0.1, keep_n=100000)
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=cluster, id2word=dictionary, passes=10, workers=2)
    return lda_model

def eliminar_vacios(data):
    data = data[data['TEXTO_TOKEN'].apply(len) > 3]
    return data
