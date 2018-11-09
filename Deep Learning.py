import pandas as pd 
import re 
import numpy as np 


dataset = pd.read_csv("Roman Urdu DataSet.csv")

X = dataset.iloc[:,0].values
y = dataset.iloc[:,1].values

#Corrects the spelling of Negative on one datapoint 
for i in range(len(y)):
    if y[i] == "Positive" or y[i] == 'Negative' or y[i] == 'Neutral':
        continue
    else:
        y[i] = 'Negative'
        
#    new = []
#    y_2 = []
#    for i in range(len(X)):
#        if not X[i] in new:
#            new.append(X[i])
#            y_2.append(y[i])
#           

#X = new
#y = y_2

corpus = []
for i in range(len(X)):
    review = re.sub('[^a-zA-Z]', ' ', str(X[i]))
    review = review.lower()
    review = review.split()
    review = ' '.join(review)
    corpus.append(review)
    

#from sklearn.feature_extraction.text import CountVectorizer
#cv = CountVectorizer(max_features = 2500)
#X = cv.fit_transform(corpus).toarray()


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_df = 10,min_df = 2, max_features = 7500)
X = vectorizer.fit_transform(corpus).toarray()




from sklearn import preprocessing
from keras.utils import np_utils
# encode class values as integers
encoder = preprocessing.LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)
# convert integers to dummy variables (i.e. one hot encoded)
y = np_utils.to_categorical(encoded_Y)

# Splitting the dataset into the Training set and Test set
#from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


from keras.models import Sequential
from keras.layers import Dense
# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 400, init = 'uniform', activation = 'relu', input_dim = 7500))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 400, init = 'uniform', activation = 'relu'))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 400, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 3, init = 'uniform', activation = 'softmax'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit_generator(X, y, batch_size = 10, epochs = 10,multiprocessing = True)

classifier.save("classifier.h5")
