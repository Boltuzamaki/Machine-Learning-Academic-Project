
from django.shortcuts import render
from django.http import HttpResponse
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import pickle

modulePath = os.path.dirname(__file__)  # get current directory
filePath = os.path.join(modulePath, 'tfidftransformer.tfidf')
with open(filePath, 'rb') as f:
	tfidfvec=pickle.load(f)
filePath = os.path.join(modulePath, 'finalized_model.model')
with open(filePath, 'rb') as f:
	model=pickle.load(f)

dicts = {0:"tech",
        1:"business",
        2:"sport",
        3:"entertainment",
        4:"politics"}

def text_lowercase(text): 
    return text.lower() 

def home(request):
    return render(request, 'index.html')


def classify(request):
    #Get the text
    djtext = request.GET.get('text', 'default')
    if djtext != "default":
    	text = [text_lowercase(djtext)] 
    	features = tfidfvec.transform(text).toarray() 
    	predicted = model.predict(features)
    	predicted = dicts[predicted[0]]
    	
    if djtext=="default":
    	predicted = "No Text Provided please try again"

    params = {'Category': predicted}
    return render(request, 'result.html', params)



