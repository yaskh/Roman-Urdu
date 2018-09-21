import pandas as pd 
import gensim 
from nltk.tokenize import word_tokenize
import re
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import warnings
from sklearn.metrics import accuracy_score
warnings.filterwarnings("ignore")
data = pd.read_csv('file.csv')



def hashing(word):
    word = re.sub(r'ain$', r'ein', word)
    word = re.sub(r'ai', r'ae', word)
    word = re.sub(r'ay$', r'e', word)
    word = re.sub(r'ey$', r'e', word)
    word = re.sub(r'ie$', r'y', word)
    word = re.sub(r'^es', r'is', word)
    word = re.sub(r'a+', r'a', word)
    word = re.sub(r'j+', r'j', word)
    word = re.sub(r'd+', r'd', word)
    word = re.sub(r'u', r'o', word)
    word = re.sub(r'o+', r'o', word)
    word = re.sub(r'ee+', r'i', word)
    if not re.match(r'ar', word):
        word = re.sub(r'ar', r'r', word)
    word = re.sub(r'iy+', r'i', word)
    word = re.sub(r'ih+', r'eh', word)
    word = re.sub(r's+', r's', word)
    if re.search(r'[rst]y', 'word') and word[-1] != 'y':
        word = re.sub(r'y', r'i', word)
    if re.search(r'[bcdefghijklmnopqrtuvwxyz]i', word):
        word = re.sub(r'i$', r'y', word)
    if re.search(r'[acefghijlmnoqrstuvwxyz]h', word):
        word = re.sub(r'h', '', word)
    word = re.sub(r'k', r'q', word)
    return word

def array_cleaner(array):
  X = []
  for sentence in array:
    clean_sentence = ''
    review = re.sub('[^a-zA-Z]', ' ', sentence)
    review = review.lower()
    words = review.split(' ')
    for word in words:
        clean_sentence = clean_sentence +' '+ word
    X.append(clean_sentence)
  return X

X = data.iloc[:,0]
Y = data.iloc[:,1]

train_X, X_test, train_Y, y_test = train_test_split(X, Y, test_size = 0.20, random_state = 123)
    
train_X = array_cleaner(train_X)
X_test = array_cleaner(X_test)
num_features = 50
min_word_count = 5
num_workers = 4     
context = 10        
downsampling = 1e-3 

# Initializing the train model
from gensim.models import word2vec
print("Training model....")
model = word2vec.Word2Vec(train_X,\
                          workers=num_workers,\
                          size=num_features,\
                          min_count=min_word_count,\
                          window=context,
                          sample=downsampling)


model.init_sims(replace=True)
model_name = "Roman_Urdu_Model"
model.save(model_name)

# Function to average all word vectors 
def featureVecMethod(words, model, num_features):
    # Pre-initialising empty numpy array for speed
    featureVec = np.zeros(num_features,dtype="float32")
    nwords = 0
    
    #Converting Index2Word which is a list 
    index2word_set = set(model.wv.index2word)
    
    for word in  words:
        if word in index2word_set:
            nwords = nwords + 1
            featureVec = np.add(featureVec,model[word])
    
    # Dividing the result by number of words to get average
    featureVec = np.divide(featureVec, nwords)
    return featureVec
# Function for calculating the average feature vector
def getAvgFeatureVecs(reviews, model, num_features):
    counter = 0
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    for review in reviews:
        # Printing a status message every 1000th review
        if counter%1000 == 0:
            print("Review %d of %d"%(counter,len(reviews)))
            
        reviewFeatureVecs[counter] = featureVecMethod(review, model, num_features)
        counter = counter+1
        
    return reviewFeatureVecs
# Calculating average feature vector for training set

trainDataVecs = getAvgFeatureVecs(train_X, model, num_features)


# calculating average feature vector for test
testDataVecs = getAvgFeatureVecs(X_test, model, num_features)

#classifier
forest = RandomForestClassifier(n_estimators = 5)
#classifier = SGDClassifier()
print("Fitting random forest to training data....")    
forest = forest.fit(trainDataVecs, train_Y)
#classifier.fit(trainDataVecs,train_Y)
#testdata=["kamiyab"]
#testdata = array_cleaner(testdata)
#testDataVecs = getAvgFeatureVecs(testdata, model, num_features)
#result = forest.predict(testDataVecs)
#print(result)
y_pred = forest.predict(testDataVecs)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred,y_test)
print(accuracy_score(y_test, y_pred))