import requests
from flask import Flask, render_template, request

from bs4 import BeautifulSoup

import scipy.sparse

# custom import
from utils import cleanDoc, count_vectorizer, \
         tfidf_vectorizer,load_model, getLabel, check_url_format,\
            load_url_model

app = Flask(__name__)



# load classifier model
model = load_model()
url_model = load_url_model()


# count vec and tfidf vec predict
def make_prediction_count_tfidf(corpus, url):

    tfidf_corpus = tfidf_vectorizer(corpus)
    count_corpus = count_vectorizer(url)

    X = scipy.sparse.hstack([tfidf_corpus, count_corpus])
    
    result = model.predict(X).item()

    return getLabel(result)


def make_url_prediction(url):
    count_corpus = count_vectorizer(url, True)
    print('url')
    result = url_model.predict(count_corpus).item()
    return getLabel(result)




headers = {"User-Agent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36'}

@app.route('/', methods=['GET', 'POST'])
def index():
    errors = []
    results = {}
    if request.method == "POST":
        # get url that the user has entered
        # Note: Make sure, that your URL includes http:// or https://. 
        # Otherwise our application won’t detect, that it’s a valid URL.
        try:
            url = request.form['url']
            url = check_url_format(url.strip())
            r = requests.get(url,headers=headers)
        except:
            errors.append(
                "Unable to get URL. Please make sure it's valid and try again."
            )
            return render_template('index.html', errors=errors)

        if r:
            # text processing
            raw = BeautifulSoup(r.text, 'html.parser').get_text()
            corpus = cleanDoc([raw])
            # nltk.data.path.append('./nltk_data/')  # set the path
            # save the results
            if len(corpus[0]) >= 300:
                results = [url, corpus[0][:300],
                            make_prediction_count_tfidf(corpus, url)]
            
            else:
                results = [url, 'None',
                        make_url_prediction(url)]
            print(results)
        else:
             
            results = [url, 'None',
                        make_url_prediction(url)]

    return render_template('index.html', errors=errors, results=results)

if __name__ == '__main__':
    app.run()