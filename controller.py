from flask import render_template, request, redirect, flash, session,Markup, jsonify, Blueprint, url_for
import werkzeug.utils
import os
import pandas as pd
from werkzeug import url_encode
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy import func
from flask_login import login_user, current_user, logout_user, login_required
from app import app
from app.admin.forms import  admin_F, editadmin_F, loginadmin_F
from app.models import Tusers
from app import db, bcrypt, engine
import re
import string
import nltk
nltk.download('stopwords', quiet=True)
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory,StopWordRemover, ArrayDictionary
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split,GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.naive_bayes  import MultinomialNB
from sklearn.svm import SVC, LinearSVC
import pickle
import numpy as np

factory = StemmerFactory()
stemmer = factory.create_stemmer()
list_stopwords = stopwords.words('indonesian')
# list_stopwords.extend(open('stopword.txt', 'r'))
list_stopwords.extend(['brt','rt', 'cc','b', 'yg', 'kpd', 'jkw', 'dgn','pdhl','ah','bgmn','pa','pak','bpk','thn','utk','aa','aja','sih','dg','ga', 'shg'])
# # convert list to dictionary
list_stopwords = set(list_stopwords)
# from flask_sqlalchemy import SQLAlchemy




# INISIASI BLUEPRINT
radmin = Blueprint('radmin', __name__)

# FITUR LOGIN
@app.route('/')
@app.route('/index')
def index():
    form = loginadmin_F()
    return render_template('login.html',form=form)

@app.route('/login',methods=['GET','POST'] )
def login():
    form = loginadmin_F()
    if current_user.is_authenticated:
        return redirect(url_for('radmin.dashboard'))

    if form.validate_on_submit():
        cekemail=Tusers.query.filter_by(email=form.email.data).first()
        if cekemail and bcrypt.check_password_hash(cekemail.password, form.password.data):
            login_user(cekemail)
            # print(cekemail)
            return redirect(url_for('radmin.dashboard'))
        else:
            flash('Login Gagal, pastikan email dan password benar', 'danger')
    return render_template('login.html',form=form)

#FITUR LOGOUT
@radmin.route('/logout',)
def logout():
    logout_user()
    return redirect('login')

#FITUR REGISTER USERS
@app.route('/register',methods=['GET','POST'] )
def register():
    form = admin_F()
    if form.validate_on_submit():
        pass_hash = bcrypt.generate_password_hash(form.password.data).decode('UTF-8')
        add_user = Tusers(username=form.username.data,email=form.email.data, password=pass_hash)
        db.session.add(add_user)
        db.session.commit()
        flash(f'Akun {form.username.data} berhasil terdaftar', 'success')
        return redirect('/register')
    return render_template('register.html',form=form)


# FITUR EDIT USERS
@radmin.route('/edituser', methods=['GET','POST'] )
@login_required
def edituser():
    form = editadmin_F()
    if form.validate_on_submit():
        pass_hash = bcrypt.generate_password_hash(form.password.data).decode('UTF-8')
        current_user.username = form.username.data
        current_user.email = form.email.data
        current_user.password = pass_hash
        db.session.commit()
        flash("Data Berhasil di ubah", 'warning')
        return redirect(url_for('radmin.edituser'))
    elif request.method=="GET":
        form.username.data =current_user.username
        form.email.data =current_user.email
        form.password.data = current_user.password

    return render_template('edituser.html', form=form)


#FITUR LAMAN DASHBOARD
@radmin.route("/dashboard")
@login_required
def dashboard():
    return render_template('index.html')


#FITUR UPLOAD FILE
@radmin.route("/upload")
@login_required
def upload():

    return render_template('upload.html')

# #FITUR PROSES UPLOAD CSV
# @radmin.route('/uploadcsv', methods=['GET','POST'])
# @login_required
# def uploadcsv():
#     if request.method == 'POST':
#         f = request.files['csvfile']
#         f.save(os.path.join('app/data', 'DATA.csv'))
#         df = pd.read_csv('app/data/DATA.csv')
#         return redirect('proses')
#     else:
#         return render_template('analisis.html', data=df.to_html(header=False))

# #FITUR PROSES SENTIMEN
# @radmin.route('/proses', methods=['GET','POST'])
# @login_required
# def proses():
#     df = pd.read_csv('app/data/DATA.csv', encoding='utf-8', delimiter=",")
#     df['tokens'] = df['Text'].apply(lambda row: get_keywords(row))
#     data = pd.DataFrame(df)
#     tokens = df['tokens']
#     filename = 'app/data/svm_model.sav'
#     load_model = pickle.load(open(filename, 'rb'))
#     transformer = TfidfTransformer()
#     cv = TfidfVectorizer(sublinear_tf=True, vocabulary=pickle.load(open("app/data/svm_fet.pkl", "rb")))
#     tokens = transformer.fit_transform(cv.fit_transform(tokens))
#     polarity = load_model.predict(tokens)
#     df = pd.DataFrame()
#     df = data
#     data['Label'] = polarity
#     df = pd.DataFrame(data)
#     df['Label'] = df['Label'].replace({2: 'Positif', 1: 'Negatif'})
#     aw = df[["Author","Date","Text","Label"]]
#     ad = df.to_csv('app/data/Data-cleaning.csv')
#     with engine.connect() as conn, conn.begin():
#             aw.to_sql('tb_sentimen', conn, if_exists='replace')
#     return render_template('analisis.html',data=aw.to_html(table_id='dataTable',
#      classes='table', border='0', justify='center'))


#CREATE MODEL SENTIMEN ANALISIS
@radmin.route('/uploadcsv', methods=['GET','POST'])
@login_required
def uploadcsv():
    df = pd.read_csv('app/data/DATA.csv', encoding='utf-8', delimiter=",")
    if request.method == 'POST':
        f = request.files['csvfile']
        f.save(os.path.join('app/data', 'DATA.csv'))
        df = pd.read_csv('app/data/DATA.csv' ,encoding='utf-8')
        df['tokens'] = df['Text'].apply(lambda row: get_keywords(row))
        #BARU LAGI INI MAH
        X = df['tokens']
        y = df['Label']
        def feature_extration (data):
            tfv = TfidfVectorizer(sublinear_tf=True)
            fet = tfv.fit_transform(data)
            pickle.dump(tfv.vocabulary_, open('app/data/svm_fet.pkl', 'wb'))
            return fet
        data = np.array(X)
        label = np.array(y)
        features = feature_extration(data)
        print(features)
        X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.80)
        for c in [0.01, 0.05, 0.25, 0.5, 1]:
            sv = SVC(C=c)
            sv.fit(X_train, y_train)
            print('Accurasi dari C=%s: %s' % (c, accuracy_score(y_test, sv.predict(X_test))))
        val_pred = sv.predict(X_test)
        print(accuracy_score(y_test, val_pred))
        print(classification_report(y_test, val_pred))

        final_model = SVC(C=1)
        final_model.fit(features,label)
        val_pred = final_model.predict(X_test)
        print(classification_report(y_test, val_pred))
        print('____________')
        print(accuracy_score(y_test, val_pred))
        print('____________')
        print(confusion_matrix(y_test, val_pred))


        filename = 'svm_model.sav'
        pickle.dump(final_model, open(filename, 'wb'))
        adaw = pd.DataFrame(df)
        ada = adaw.to_csv('app/data/Data-cleaning.csv', encoding='utf-8')
    return render_template('analisis.html',data=adaw.to_html(table_id='dataTable', classes='table', border='0', justify='center'))




# Text Preprocessing
def get_keywords(row):
    # prepare regex for char filtering
      tokens = row.split()
      #Hapus At Users
      tokens = [re.sub("@[A-Za-z0-9-_]+","",w) for w in tokens]
      #Hapus Tag
      tokens = [re.sub("#[A-Za-z0-9-_]+","",w) for w in tokens]
      #Hapus URL
      tokens = [re.sub(r"http\S+","", w) for w in tokens]
      #Case Folding
      tokens = [w.lower() for w in tokens]
      #Stemming Sastrawi
      tokens = [stemmer.stem(w) for w in tokens]
      # remove punctuation from each word
      re_punc = re.compile('[%s]' % re.escape(string.punctuation))
      tokens = [re_punc.sub('', w) for w in tokens]
      tokens = [word for word in tokens if word.isalpha()]
      #Stopword NLTK Corpus Indonesian
      tokens = [w for w in tokens if w not in list_stopwords]
      tokens = ' '.join(tokens) #Merubah Split
      return tokens

#FITUR HASIL ANALISIS
@radmin.route("/analisis")
@login_required
def analisis():
    with engine.connect() as conn, conn.begin():
        sql = 'SELECT * FROM tb_sentimen;'
        df = pd.read_sql_query(sql, conn)
        # data = pd.read_csv('app/data/Data-cleaning.csv', encoding='utf-8')
        df['Label'] = df['Label'].replace({2: 'Positif', 1: 'Negatif'})
        df = df[["Author","Date","Text","Label"]]
    return render_template('analisis.html', data=df.to_html(table_id='dataTable',
    classes='table', border='0', justify='center', index=False))


#FITUR GRAFIK HASIL ANALISIS
@radmin.route("/grafik")
@login_required
def grafik():
    with engine.connect() as conn, conn.begin():
        sql = 'SELECT * FROM tb_sentimen;'
        df = pd.read_sql_query(sql, conn)
        # print(df.head())
        #menghitung jumlah tweet
        index = df.index
        jml_tweet = len(index)
        #Menghitung Jumlah Pengguna
        Author = df.Author
        Author = len(Author)
        #menghitung jumlah positif
        jml_pos = df['Label']=='Positif'
        jml_pos = sum(jml_pos)
        #menghitung jumlah negatif
        jml_neg = df['Label']=='Negatif'
        jml_neg = sum(jml_neg)
       
        #membuat Chart Frequncy Author
        author_freq = df['Author'].value_counts().reset_index()
        author_freq.columns = ['Author','Total']
        total = pd.DataFrame(author_freq)
        tot_author = total.Author.head(5)
        tot_author = tot_author.tolist()
        tot_total = total.Total.head(5)
        tot_total = tot_total.tolist()

    return render_template('grafik.html', data=df.to_html(index=False,table_id='dataTable', classes='table', border='0', justify='center'),jml_tweet=jml_tweet, Author=Author,jml_pos=jml_pos, jml_neg=jml_neg, tot_author=tot_author,tot_total=tot_total)



# KODINGAN MEDAN
@radmin.route('/medan', methods=['GET','POST'])
@login_required
def medan():
    if request.method == 'POST':
        f = request.files['csvfile']
        f.save(os.path.join('app/data/medan', 'DATA.csv'))
        df = pd.read_csv('app/data/medan/DATA.csv')
        return redirect('proses_medan')
    else:
        return render_template('analisis.html', data=df.to_html(header=False))

#FITUR PROSES SENTIMEN
@radmin.route('/proses_medan', methods=['GET','POST'])
@login_required
def proses_medan():
    df = pd.read_csv('app/data/medan/DATA.csv', encoding='utf-8', delimiter=",")
    df['tokens'] = df['Text'].apply(lambda row: get_keywords(row))
    data = pd.DataFrame(df)
    tokens = df['tokens']
    filename = 'app/data/svm_model.sav'
    load_model = pickle.load(open(filename, 'rb'))
    transformer = TfidfTransformer()
    cv = TfidfVectorizer(sublinear_tf=True, vocabulary=pickle.load(open("app/data/svm_fet.pkl", "rb")))
    tokens = transformer.fit_transform(cv.fit_transform(tokens))
    polarity = load_model.predict(tokens)
    df = pd.DataFrame()
    df = data
    data['Label'] = polarity
    df = pd.DataFrame(data)
    df['Label'] = df['Label'].replace({2: 'Positif', 1: 'Negatif'})
    aw = df[["Author","Date","Text","Label"]]
    ad = df.to_csv('app/data//medan/Data-cleaning.csv')
    with engine.connect() as conn, conn.begin():
            aw.to_sql('tb_medan', conn, if_exists='replace')
    return render_template('view-medan.html',data=aw.to_html(table_id='dataTable',
     classes='table', border='0', justify='center'))

@radmin.route("/hasil_medan")
@login_required
def hasil_medan():
    with engine.connect() as conn, conn.begin():
        sql = 'SELECT * FROM tb_medan;'
        df = pd.read_sql_query(sql, conn)
    data = pd.read_csv('app/data/medan/Data-cleaning.csv', encoding='utf-8')
    data = data[["Author","Date","Text","Label"]]
    print(data.head())
    return render_template('view-medan.html', data=data.to_html(table_id='dataTable',
    classes='table', border='0', justify='center', index=False))


@radmin.route("/graf_medan")
@login_required
def graf_medan():
    with engine.connect() as conn, conn.begin():
        sql = 'SELECT * FROM tb_medan;'
        df = pd.read_sql_query(sql, conn)
        # print(df.head())
        #menghitung jumlah tweet
        index = df.index
        jml_tweet = len(index)
        #Menghitung Jumlah Pengguna
        Author = df.Author
        Author = len(Author)
        #menghitung jumlah positif
        jml_pos = df['Label']=='Positif'
        jml_pos = sum(jml_pos)
        #menghitung jumlah negatif
        jml_neg = df['Label']=='Negatif'
        jml_neg = sum(jml_neg)
        #menghitung jumlah netral
        #membuat Chart Frequncy Author
        author_freq = df['Author'].value_counts().reset_index()
        author_freq.columns = ['Author','Total']
        total = pd.DataFrame(author_freq)
        tot_author = total.Author.head(5)
        tot_author = tot_author.tolist()
        tot_total = total.Total.head(5)
        tot_total = tot_total.tolist()
    return render_template('graf_medan.html', data=df.to_html(index=False,table_id='dataTable', classes='table', border='0', justify='center'),jml_tweet=jml_tweet, Author=Author,jml_pos=jml_pos, jml_neg=jml_neg, tot_author=tot_author,tot_total=tot_total)

# KODINGAN MAKASAR
@radmin.route('/makasar', methods=['GET','POST'])
@login_required
def makasar():
    if request.method == 'POST':
        f = request.files['csvfile']
        f.save(os.path.join('app/data/makasar', 'DATA.csv'))
        df = pd.read_csv('app/data/makasar/DATA.csv')
        return redirect('proses_makasar')
    else:
        return render_template('analisis.html', data=df.to_html(header=False))

#FITUR PROSES SENTIMEN
@radmin.route('/proses_makasar', methods=['GET','POST'])
@login_required
def proses_makasar():
    df = pd.read_csv('app/data/makasar/DATA.csv', encoding='utf-8', delimiter=",")
    df['tokens'] = df['Text'].apply(lambda row: get_keywords(row))
    data = pd.DataFrame(df)
    tokens = df['tokens']
    filename = 'app/data/svm_model.sav'
    load_model = pickle.load(open(filename, 'rb'))
    transformer = TfidfTransformer()
    cv = TfidfVectorizer(sublinear_tf=True, vocabulary=pickle.load(open("app/data/svm_fet.pkl", "rb")))
    tokens = transformer.fit_transform(cv.fit_transform(tokens))
    polarity = load_model.predict(tokens)
    df = pd.DataFrame()
    df = data
    data['Label'] = polarity
    df = pd.DataFrame(data)
    df['Label'] = df['Label'].replace({2: 'Positif', 1: 'Negatif'})
    aw = df[["Author","Date","Text","Label"]]
    ad = df.to_csv('app/data//makasar/Data-cleaning.csv')
    with engine.connect() as conn, conn.begin():
            aw.to_sql('tb_makasar', conn, if_exists='replace')
    return render_template('view-makasar.html',data=aw.to_html(table_id='dataTable',
     classes='table', border='0', justify='center'))

@radmin.route("/hasil_makasar")
@login_required
def hasil_makasar():
    with engine.connect() as conn, conn.begin():
        sql = 'SELECT * FROM tb_makasar'
        df = pd.read_sql_query(sql, conn)
    data = pd.read_csv('app/data/makasar/Data-cleaning.csv', encoding='utf-8')
    data = data[["Author","Date","Text","Label"]]
    print(data.head())
    return render_template('view-makasar.html', data=data.to_html(table_id='dataTable',
    classes='table', border='0', justify='center', index=False))


@radmin.route("/graf_makasar")
@login_required
def graf_makasar():
    with engine.connect() as conn, conn.begin():
        sql = 'SELECT * FROM tb_makasar;'
        df = pd.read_sql_query(sql, conn)
        # print(df.head())
        #menghitung jumlah tweet
        index = df.index
        jml_tweet = len(index)
        #Menghitung Jumlah Pengguna
        Author = df.Author
        Author = len(Author)
        #menghitung jumlah positif
        jml_pos = df['Label']=='Positif'
        jml_pos = sum(jml_pos)
        #menghitung jumlah negatif
        jml_neg = df['Label']=='Negatif'
        jml_neg = sum(jml_neg)
        #menghitung jumlah netral
        #membuat Chart Frequncy Author
        author_freq = df['Author'].value_counts().reset_index()
        author_freq.columns = ['Author','Total']
        total = pd.DataFrame(author_freq)
        tot_author = total.Author.head(5)
        tot_author = tot_author.tolist()
        tot_total = total.Total.head(5)
        tot_total = tot_total.tolist()
    return render_template('graf_makasar.html', data=df.to_html(index=False,table_id='dataTable', classes='table', border='0', justify='center'),jml_tweet=jml_tweet, Author=Author,jml_pos=jml_pos, jml_neg=jml_neg, tot_author=tot_author,tot_total=tot_total)

# KODINGAN MANADO
@radmin.route('/manado', methods=['GET','POST'])
@login_required
def manado():
    if request.method == 'POST':
        f = request.files['csvfile']
        f.save(os.path.join('app/data/manado', 'DATA.csv'))
        df = pd.read_csv('app/data/manado/DATA.csv')
        return redirect('proses_manado')
    else:
        return render_template('analisis.html', data=df.to_html(header=False))

#FITUR PROSES SENTIMEN
@radmin.route('/proses_manado', methods=['GET','POST'])
@login_required
def proses_manado():
    df = pd.read_csv('app/data/manado/DATA.csv', encoding='utf-8', delimiter=",")
    df['tokens'] = df['Text'].apply(lambda row: get_keywords(row))
    data = pd.DataFrame(df)
    tokens = df['tokens']
    filename = 'app/data/svm_model.sav'
    load_model = pickle.load(open(filename, 'rb'))
    transformer = TfidfTransformer()
    cv = TfidfVectorizer(sublinear_tf=True, vocabulary=pickle.load(open("app/data/svm_fet.pkl", "rb")))
    tokens = transformer.fit_transform(cv.fit_transform(tokens))
    polarity = load_model.predict(tokens)
    df = pd.DataFrame()
    df = data
    data['Label'] = polarity
    df = pd.DataFrame(data)
    df['Label'] = df['Label'].replace({2: 'Positif', 1: 'Negatif'})
    aw = df[["Author","Date","Text","Label"]]
    ad = df.to_csv('app/data//manado/Data-cleaning.csv')
    with engine.connect() as conn, conn.begin():
            aw.to_sql('tb_manado', conn, if_exists='replace')
    return render_template('view-manado.html',data=aw.to_html(table_id='dataTable',
     classes='table', border='0', justify='center'))

@radmin.route("/hasil_manado")
@login_required
def hasil_manado():
    with engine.connect() as conn, conn.begin():
        sql = 'SELECT * FROM tb_manado'
        df = pd.read_sql_query(sql, conn)
    data = pd.read_csv('app/data/manado/Data-cleaning.csv', encoding='utf-8')
    data = data[["Author","Date","Text","Label"]]
    print(data.head())
    return render_template('view-manado.html', data=data.to_html(table_id='dataTable',
    classes='table', border='0', justify='center', index=False))


@radmin.route("/graf_manado")
@login_required
def graf_manado():
    with engine.connect() as conn, conn.begin():
        sql = 'SELECT * FROM tb_manado;'
        df = pd.read_sql_query(sql, conn)
        # print(df.head())
        #menghitung jumlah tweet
        index = df.index
        jml_tweet = len(index)
        #Menghitung Jumlah Pengguna
        Author = df.Author
        Author = len(Author)
        #menghitung jumlah positif
        jml_pos = df['Label']=='Positif'
        jml_pos = sum(jml_pos)
        #menghitung jumlah negatif
        jml_neg = df['Label']=='Negatif'
        jml_neg = sum(jml_neg)
        #menghitung jumlah netral
        #membuat Chart Frequncy Author
        author_freq = df['Author'].value_counts().reset_index()
        author_freq.columns = ['Author','Total']
        total = pd.DataFrame(author_freq)
        tot_author = total.Author.head(5)
        tot_author = tot_author.tolist()
        tot_total = total.Total.head(5)
        tot_total = tot_total.tolist()
    return render_template('graf_manado.html', data=df.to_html(index=False,table_id='dataTable', classes='table', border='0', justify='center'),jml_tweet=jml_tweet, Author=Author,jml_pos=jml_pos, jml_neg=jml_neg, tot_author=tot_author,tot_total=tot_total)

# KODINGAN SEMARANG
@radmin.route('/semarang', methods=['GET','POST'])
@login_required
def semarang():
    if request.method == 'POST':
        f = request.files['csvfile']
        f.save(os.path.join('app/data/semarang', 'DATA.csv'))
        df = pd.read_csv('app/data/semarang/DATA.csv')
        return redirect('proses_semarang')
    else:
        return render_template('analisis.html', data=df.to_html(header=False))

#FITUR PROSES SENTIMEN
@radmin.route('/proses_semarang', methods=['GET','POST'])
@login_required
def proses_semarang():
    df = pd.read_csv('app/data/semarang/DATA.csv', encoding='utf-8', delimiter=",")
    df['tokens'] = df['Text'].apply(lambda row: get_keywords(row))
    data = pd.DataFrame(df)
    tokens = df['tokens']
    filename = 'app/data/svm_model.sav'
    load_model = pickle.load(open(filename, 'rb'))
    transformer = TfidfTransformer()
    cv = TfidfVectorizer(sublinear_tf=True, vocabulary=pickle.load(open("app/data/svm_fet.pkl", "rb")))
    tokens = transformer.fit_transform(cv.fit_transform(tokens))
    polarity = load_model.predict(tokens)
    df = pd.DataFrame()
    df = data
    data['Label'] = polarity
    df = pd.DataFrame(data)
    df['Label'] = df['Label'].replace({2: 'Positif', 1: 'Negatif'})
    aw = df[["Author","Date","Text","Label"]]
    ad = df.to_csv('app/data//semarang/Data-cleaning.csv')
    with engine.connect() as conn, conn.begin():
            aw.to_sql('tb_semarang', conn, if_exists='replace')
    return render_template('view-semarang.html',data=aw.to_html(table_id='dataTable',
     classes='table', border='0', justify='center'))

@radmin.route("/hasil_semarang")
@login_required
def hasil_semarang():
    with engine.connect() as conn, conn.begin():
        sql = 'SELECT * FROM tb_semarang'
        df = pd.read_sql_query(sql, conn)
    data = pd.read_csv('app/data/semarang/Data-cleaning.csv', encoding='utf-8')
    data = data[["Author","Date","Text","Label"]]
    print(data.head())
    return render_template('view-semarang.html', data=data.to_html(table_id='dataTable',
    classes='table', border='0', justify='center', index=False))


@radmin.route("/graf_semarang")
@login_required
def graf_semarang():
    with engine.connect() as conn, conn.begin():
        sql = 'SELECT * FROM tb_semarang;'
        df = pd.read_sql_query(sql, conn)
        # print(df.head())
        #menghitung jumlah tweet
        index = df.index
        jml_tweet = len(index)
        #Menghitung Jumlah Pengguna
        Author = df.Author
        Author = len(Author)
        #menghitung jumlah positif
        jml_pos = df['Label']=='Positif'
        jml_pos = sum(jml_pos)
        #menghitung jumlah negatif
        jml_neg = df['Label']=='Negatif'
        jml_neg = sum(jml_neg)
        #menghitung jumlah netral
        #membuat Chart Frequncy Author
        author_freq = df['Author'].value_counts().reset_index()
        author_freq.columns = ['Author','Total']
        total = pd.DataFrame(author_freq)
        tot_author = total.Author.head(5)
        tot_author = tot_author.tolist()
        tot_total = total.Total.head(5)
        tot_total = tot_total.tolist()
    return render_template('graf_semarang.html', data=df.to_html(index=False,table_id='dataTable', classes='table', border='0', justify='center'),jml_tweet=jml_tweet, Author=Author,jml_pos=jml_pos, jml_neg=jml_neg, tot_author=tot_author,tot_total=tot_total)


#KODINGAN SURABAYA
@radmin.route('/surabaya', methods=['GET','POST'])
@login_required
def surabaya():
    if request.method == 'POST':
        f = request.files['csvfile']
        f.save(os.path.join('app/data/surabaya', 'DATA.csv'))
        df = pd.read_csv('app/data/surabaya/DATA.csv')
        return redirect('proses_surabaya')
    else:
        return render_template('analisis.html', data=df.to_html(header=False))

#FITUR PROSES SENTIMEN
@radmin.route('/proses_surabaya', methods=['GET','POST'])
@login_required
def proses_surabaya():
    df = pd.read_csv('app/data/surabaya/DATA.csv', encoding='utf-8', delimiter=",")
    df['tokens'] = df['Text'].apply(lambda row: get_keywords(row))
    data = pd.DataFrame(df)
    tokens = df['tokens']
    filename = 'app/data/svm_model.sav'
    load_model = pickle.load(open(filename, 'rb'))
    transformer = TfidfTransformer()
    cv = TfidfVectorizer(sublinear_tf=True, vocabulary=pickle.load(open("app/data/svm_fet.pkl", "rb")))
    tokens = transformer.fit_transform(cv.fit_transform(tokens))
    polarity = load_model.predict(tokens)
    df = pd.DataFrame()
    df = data
    data['Label'] = polarity
    df = pd.DataFrame(data)
    df['Label'] = df['Label'].replace({2: 'Positif', 1: 'Negatif'})
    aw = df[["Author","Date","Text","Label"]]
    ad = df.to_csv('app/data//surabaya/Data-cleaning.csv')
    with engine.connect() as conn, conn.begin():
            aw.to_sql('tb_surabaya', conn, if_exists='replace')
    return render_template('view-surabaya.html',data=aw.to_html(table_id='dataTable',
     classes='table', border='0', justify='center'))

@radmin.route("/hasil_surabaya")
@login_required
def hasil_surabaya():
    with engine.connect() as conn, conn.begin():
        sql = 'SELECT * FROM tb_surabaya'
        df = pd.read_sql_query(sql, conn)
    data = pd.read_csv('app/data/surabaya/Data-cleaning.csv', encoding='utf-8')
    data = data[["Author","Date","Text","Label"]]
    print(data.head())
    return render_template('view-surabaya.html', data=data.to_html(table_id='dataTable',
    classes='table', border='0', justify='center', index=False))


@radmin.route("/graf_surabaya")
@login_required
def graf_surabaya():
    with engine.connect() as conn, conn.begin():
        sql = 'SELECT * FROM tb_surabaya;'
        df = pd.read_sql_query(sql, conn)
        # print(df.head())
        #menghitung jumlah tweet
        index = df.index
        jml_tweet = len(index)
        #Menghitung Jumlah Pengguna
        Author = df.Author
        Author = len(Author)
        #menghitung jumlah positif
        jml_pos = df['Label']=='Positif'
        jml_pos = sum(jml_pos)
        #menghitung jumlah negatif
        jml_neg = df['Label']=='Negatif'
        jml_neg = sum(jml_neg)
        #menghitung jumlah netral
        #membuat Chart Frequncy Author
        author_freq = df['Author'].value_counts().reset_index()
        author_freq.columns = ['Author','Total']
        total = pd.DataFrame(author_freq)
        tot_author = total.Author.head(5)
        tot_author = tot_author.tolist()
        tot_total = total.Total.head(5)
        tot_total = tot_total.tolist()
    return render_template('graf_surabaya.html', data=df.to_html(index=False,table_id='dataTable', classes='table', border='0', justify='center'),jml_tweet=jml_tweet, Author=Author,jml_pos=jml_pos, jml_neg=jml_neg, tot_author=tot_author,tot_total=tot_total)




#KODINGAN SURAKARTA
@radmin.route('/surakarta', methods=['GET','POST'])
@login_required
def surakarta():
    if request.method == 'POST':
        f = request.files['csvfile']
        f.save(os.path.join('app/data/surakarta', 'DATA.csv'))
        df = pd.read_csv('app/data/surakarta/DATA.csv')
        return redirect('proses_surakarta')
    else:
        return render_template('analisis.html', data=df.to_html(header=False))

#FITUR PROSES SENTIMEN
@radmin.route('/proses_surakarta', methods=['GET','POST'])
@login_required
def proses_surakarta():
    df = pd.read_csv('app/data/surakarta/DATA.csv', encoding='utf-8', delimiter=",")
    df['tokens'] = df['Text'].apply(lambda row: get_keywords(row))
    data = pd.DataFrame(df)
    tokens = df['tokens']
    filename = 'app/data/svm_model.sav'
    load_model = pickle.load(open(filename, 'rb'))
    transformer = TfidfTransformer()
    cv = TfidfVectorizer(sublinear_tf=True, vocabulary=pickle.load(open("app/data/svm_fet.pkl", "rb")))
    tokens = transformer.fit_transform(cv.fit_transform(tokens))
    polarity = load_model.predict(tokens)
    df = pd.DataFrame()
    df = data
    data['Label'] = polarity
    df = pd.DataFrame(data)
    df['Label'] = df['Label'].replace({2: 'Positif', 1: 'Negatif'})
    aw = df[["Author","Date","Text","Label"]]
    ad = df.to_csv('app/data/surakarta/Data-cleaning.csv')
    with engine.connect() as conn, conn.begin():
            aw.to_sql('tb_surakarta', conn, if_exists='replace')
    return render_template('view-surakarta.html',data=aw.to_html(table_id='dataTable',
     classes='table', border='0', justify='center'))

@radmin.route("/hasil_surakarta")
@login_required
def hasil_surakarta():
    with engine.connect() as conn, conn.begin():
        sql = 'SELECT * FROM tb_surakarta'
        df = pd.read_sql_query(sql, conn)
    data = pd.read_csv('app/data/surakarta/Data-cleaning.csv', encoding='utf-8')
    data = data[["Author","Date","Text","Label"]]
    print(data.head())
    return render_template('view-surakarta.html', data=data.to_html(table_id='dataTable',
    classes='table', border='0', justify='center', index=False))


@radmin.route("/graf_surakarta")
@login_required
def graf_surakarta():
    with engine.connect() as conn, conn.begin():
        sql = 'SELECT * FROM tb_surakarta;'
        df = pd.read_sql_query(sql, conn)
        # print(df.head())
        #menghitung jumlah tweet
        index = df.index
        jml_tweet = len(index)
        #Menghitung Jumlah Pengguna
        Author = df.Author
        Author = len(Author)
        #menghitung jumlah positif
        jml_pos = df['Label']=='Positif'
        jml_pos = sum(jml_pos)
        #menghitung jumlah negatif
        jml_neg = df['Label']=='Negatif'
        jml_neg = sum(jml_neg)
        #menghitung jumlah netral
        #membuat Chart Frequncy Author
        author_freq = df['Author'].value_counts().reset_index()
        author_freq.columns = ['Author','Total']
        total = pd.DataFrame(author_freq)
        tot_author = total.Author.head(5)
        tot_author = tot_author.tolist()
        tot_total = total.Total.head(5)
        tot_total = tot_total.tolist()
    return render_template('graf_surakarta.html', data=df.to_html(index=False,table_id='dataTable', classes='table', border='0', justify='center'),jml_tweet=jml_tweet, Author=Author,jml_pos=jml_pos, jml_neg=jml_neg, tot_author=tot_author,tot_total=tot_total)


@radmin.route("/all_graf")
@login_required
def all_graf():
    with engine.connect() as conn, conn.begin():
        #Query Select Medan untuk Grafik All
        medan = 'SELECT * FROM tb_medan;'
        medan = pd.read_sql_query(medan, conn)
        medan_pos = medan['Label']=='Positif'
        medan_pos = sum(medan_pos)
        medan_neg = medan['Label']=='Negatif'
        medan_neg = sum(medan_neg)
        index = medan.index
        jml_medan = len(index)
        #Query Select Manado untuk Grafik All
        manado = 'SELECT * FROM tb_manado;'
        manado = pd.read_sql_query(manado, conn)
        index = manado.index
        jml_manado = len(index)
        manado_pos = manado['Label']=='Positif'
        manado_pos = sum(manado_pos)
        manado_neg = manado['Label']=='Negatif'
        manado_neg = sum(manado_neg)
        #Query Select Makasar untuk Grafik All
        maks = 'SELECT * FROM tb_makasar;'
        maks = pd.read_sql_query(maks, conn)
        index = maks.index
        jml_maks = len(index)
        maks_pos = maks['Label']=='Positif'
        maks_pos = sum(maks_pos)
        maks_neg = maks['Label']=='Negatif'
        maks_neg = sum(maks_neg)
        #Query Select Semarang untuk Grafik All
        sem = 'SELECT * FROM tb_semarang;'
        sem = pd.read_sql_query(sem, conn)
        index = sem.index
        jml_sem = len(index)
        sem_pos = sem['Label']=='Positif'
        sem_pos = sum(sem_pos)
        sem_neg = sem['Label']=='Negatif'
        sem_neg = sum(sem_neg)
        #Query Select Surabaya untuk Grafik All
        sby = 'SELECT * FROM tb_surabaya;'
        sby = pd.read_sql_query(sby, conn)
        index = sby.index
        jml_sby = len(index)
        sby_pos = sby['Label']=='Positif'
        sby_pos = sum(sby_pos)
        sby_neg = sby['Label']=='Negatif'
        sby_neg = sum(sby_neg)
        #Query Select Surabaya untuk Grafik All
        solo = 'SELECT * FROM tb_surakarta;'
        solo = pd.read_sql_query(solo, conn)
        index = solo.index
        jml_solo = len(index)
        solo_pos = solo['Label']=='Positif'
        solo_pos = sum(solo_pos)
        solo_neg = solo['Label']=='Negatif'
        solo_neg = sum(solo_neg)
        

    return render_template('all_graf.html', medan_pos=medan_pos, medan_neg=medan_neg, manado_pos=manado_pos, manado_neg=manado_neg,
    maks_pos=maks_pos, maks_neg=maks_neg, sem_pos=sem_pos, sem_neg=sem_neg, sby_pos=sby_pos, sby_neg=sby_neg, solo_pos=solo_pos, solo_neg=solo_neg, jml_medan=jml_medan, 
    jml_manado=jml_manado, jml_maks=jml_maks, jml_sem=jml_sem, jml_sby=jml_sby, jml_solo=jml_solo)
