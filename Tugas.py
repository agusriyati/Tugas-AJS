import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
)
from sklearn import svm
from nltk.corpus import stopwords
import re
import nltk

nltk.download("stopwords")
nltk.download("punkt")
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle


Home, Learn, Proses, Model, Implementasi = st.tabs(
    ["Home", "Learn Data", "Preprocessing dan TF-IDF", "Model", "Implementasi"]
)

with Home:
    st.title(
        """SENTIMEN ANALISIS TWITTER TERHADAP BERITA GENOSIDA ISRAEL TERHADAP PALESTINA"""
    )
    st.subheader("Kelompok 2")
    st.text(
        """
            1. AGUSRIYATI E1E122087
            2. FEBRI HAERANI E1E122089"""
    )

with Learn:
    st.title(
        """SENTIMEN ANALISIS TWITTER TERHADAP BERITA GENOSIDA ISRAEL TERHADAP PALESTINA"""
    )
    st.write(
        "Genosida adalah tindakan sistematis untuk menghancurkan atau memusnahkan sebagian atau seluruh kelompok etnis, agama, atau nasional. Ini bisa mencakup pembunuhan massal, pemindahan paksa, penahanan, penyiksaan, dan tindakan lainnya yang bertujuan untuk menghancurkan kelompok tersebut secara fisik atau budaya. Istilah ini pertama kali digunakan oleh Raphael Lemkin pada tahun 1944 untuk menggambarkan kejahatan Nazi terhadap Yahudi selama Holocaust, tetapi sekarang digunakan secara lebih umum untuk merujuk pada kejahatan serupa di tempat-tempat lain di dunia.."
    )
    st.write(
        "Dalam Klasifikasi ini data yang digunakan adalah ulasan atau komentar dari aplikasi Twitter dengan topik BERITA GENOSIDA ISRAEL TERHADAP PALESTINA."
    )
    st.title("Klasifikasi data inputan berupa : ")
    st.write("1. text : data komentar atau ulasan yang diambil dari twitter")
    st.write("2. Label: kelas keluaran [1: positif, -1: Negatif]")

    st.title("""Asal Data""")
    st.write(
        "Dataset yang digunakan adalah data hasil crowling twitter dengan kata kunci 'GENOSIDA ISRAEL TERHADAP PALESTINA' yang disimpan di https://raw.githubusercontent.com/nuskhatulhaqqi/data_mining/main/resesi_2023%20(1).csv"
    )
    st.write("Total datanya adalah 132 dengan atribut 2")
    # uploaded_files = st.file_uploader("Upload file CSV", accept_multiple_files=True)
    # if uploaded_files is not None :
    data = pd.read_csv("genosida (4).csv")
    # else:
    #    for uploaded_file in uploaded_files:
    #       data = pd.read_csv(uploaded_file)
    #       st.write("Nama File Anda = ", uploaded_file.name)
    #       st.dataframe(data)


with Proses:
    st.title("""Preprosessing""")
    clean_tag = re.compile("@\S+")
    clean_url = re.compile("https?:\/\/.*[\r\n]*")
    clean_hastag = re.compile("#\S+")
    clean_symbol = re.compile("[^a-zA-Z]")

    def clean_punct(text):
        text = clean_tag.sub("", text)
        text = clean_url.sub("", text)
        text = clean_hastag.sub(" ", text)
        text = clean_symbol.sub(" ", text)
        return text

    # Buat kolom tambahan untuk data description yang telah diremovepunctuation
    preprocessing = data["Text"].apply(clean_punct)
    clean = pd.DataFrame(preprocessing)
    "### Melakukan Cleaning "
    clean

    def clean_lower(lwr):
        lwr = lwr.lower()  # lowercase text
        return lwr

    # Buat kolom tambahan untuk data description yang telah dicasefolding
    clean = clean["Text"].apply(clean_lower)
    casefolding = pd.DataFrame(clean)
    "### Melakukan Casefolding "
    casefolding

    def to_list(text):
        t_list = []
        for i in range(len(text)):
            t_list.append(text[i])
        return t_list

    casefolding1 = to_list(clean)

    "### Melakukan Tokenisasi "

    def tokenisasi(text):
        tokenize = []
        for i in range(len(text)):
            token = word_tokenize(text[i])
            tokendata = []
            for x in token:
                tokendata.append(x)
            tokenize.append(tokendata)
        return tokendata

    tokenisasi = tokenisasi(casefolding1)
    tokenisasi

    "### Melakukan Stopword Removal "

    def stopword(text):
        stopword = []
        for i in range(len(text)):
            listStopword = set(
                stopwords.words("indonesian") + stopwords.words("english")
            )
            removed = []
            for x in text[i]:
                if x not in listStopword:
                    removed.append(x)
            stopword.append(removed)
        return removed

    stopword = stopword(tokenisasi)
    stopword
    "### Melakukan Stemming "

    def stemming(text):
        stemming = []
        for i in range(len(text)):
            factory = StemmerFactory()
            stemmer = factory.create_stemmer()
            katastem = []
            for x in text[i]:
                katastem.append(stemmer.stem(x))
            stemming.append(katastem)
        return stemming

    # kk = pd.DataFrame(stemming)
    # kk.to_csv('hasil_stemming.csv')
    kkk = pd.read_csv("hasil_stemming.csv")
    kkk

    "### Hasil Proses Pre-Prosessing "

    def gabung(test):
        join = []
        for i in range(len(stemming)):
            joinkata = " ".join(stemming[i])
            join.append(joinkata)
        hasilpreproses = pd.DataFrame(join, columns=["Text"])
        hasilpreproses.to_csv("hasilpreproses.csv")
        return hasilpreproses

    hasilpreproses = pd.read_csv("hasilpreproses.csv")
    hasilpreproses

    st.title("""TF-IDF""")
    tr_idf_model = TfidfVectorizer()
    tf_idf_vector = tr_idf_model.fit_transform(hasilpreproses["Text"])
    tf_idf_array = tf_idf_vector.toarray()
    words_set = tr_idf_model.get_feature_names_out()
    df_tf_idf = pd.DataFrame(tf_idf_array, columns=words_set)
    df_tf_idf


with Model:
    st.title("""Modeling""")
    y = data.Label
    # split data
    X_train, X_test, y_train, y_test = train_test_split(
        df_tf_idf, y, test_size=0.2, random_state=4
    )
    clf = svm.SVC(kernel="linear")
    clf.fit(X_train, y_train)
    X_pred = clf.predict(X_test)
    akurasi = round(100 * accuracy_score(y_test, X_pred))
    st.subheader("Metode Yang Digunakan Adalah Support Vector Machine")
    st.write(
        "Akurasi Terbaik Dari Skenario Uji Coba Diperoleh Sebesar : {0:0.2f} %".format(
            akurasi
        )
    )

    with open("vec_pickle", "wb") as r:
        pickle.dump(clf, r)
    with open("svm_pickle", "wb") as r:
        pickle.dump(tr_idf_model, r)


with Implementasi:
    st.title("""Implementasi Data""")

    inputan = st.text_input("Masukkan Ulasan")

    def submit():
        # input
        clean_symbol, casefolding, token, stopword, katastem, joinkata = preproses(
            inputan
        )

        # loaded_model = pickle.load(open(svm_pickle, 'rb'))
        with open("vec_pickle", "rb") as r:
            d = pickle.load(r)
        with open("svm_pickle", "rb") as r:
            data = pickle.load(r)

        X_pred = d.predict((data.transform([joinkata])).toarray())
        if X_pred[0] == 1:
            h = "Positif"
        else:
            h = "Negatif"
        hasil = f"Berdasarkan data yang Anda masukkan, maka ulasan masuk dalam kategori  : {h}"
        st.success(hasil)
        st.subheader("Preprocessing")
        st.write("Cleansing :", clean_symbol)
        st.write("Case Folding :", casefolding)
        st.write("Tokenisasi :", token)
        st.write("Stopword :", stopword)
        st.write("Steeming :", katastem)
        st.write("Siap Proses :", joinkata)

    all = st.button("Submit")
    if all:

        def preproses(inputan):
            clean_tag = re.sub("@\S+", "", inputan)
            clean_url = re.sub("https?:\/\/.*[\r\n]*", "", clean_tag)
            clean_hastag = re.sub("#\S+", " ", clean_url)
            clean_symbol = re.sub("[^a-zA-Z]", " ", clean_hastag)
            casefolding = clean_symbol.lower()
            token = word_tokenize(casefolding)
            listStopword = set(
                stopwords.words("indonesian") + stopwords.words("english")
            )
            stopword = []
            for x in token:
                if x not in listStopword:
                    stopword.append(x)
            factory = StemmerFactory()
            stemmer = factory.create_stemmer()
            katastem = []
            for x in stopword:
                katastem.append(stemmer.stem(x))
            joinkata = " ".join(katastem)
            return clean_symbol, casefolding, token, stopword, katastem, joinkata

        st.balloons()
        submit()
