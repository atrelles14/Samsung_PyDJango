# apiData/apps/utils.py
import spacy
import pandas as pd
import gensim
from .models import Opinion

nlp = spacy.load("es_core_news_sm")

def lemmatize_text(text):
    if isinstance(text, list):
        text = ' '.join(text)
    doc = nlp(text)
    return [token.lemma_ for token in doc if not token.is_stop]

def perform_lda(data, n_topics=8):
    processed_docs = data['texto'].apply(lemmatize_text)
    dictionary = gensim.corpora.Dictionary(processed_docs)
    dictionary.filter_extremes(no_below=15, no_above=0.1, keep_n=100000)
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=n_topics, id2word=dictionary, passes=10, workers=2)
    return dictionary, lda_model
