
# coding: utf-8

# Acknowledgements for code used for templates and modeling
# 
# Scikit-Lean Libraries and Documentation:
# Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
# 
# Zacstewart's github: 
# Text Classification with Scikit Learn https://gist.github.com/zacstewart/5978000 Copyright 2014
# 
# Randomized Logistic Regression - http://scikit-learn.org/0.13/modules/generated/sklearn.linear_model.RandomizedLogisticRegression.html for feature selection
# 
# Most informative features
# http://stackoverflow.com/questions/11116697/how-to-get-most-informative-features-for-scikit-learn-classifiers

# In[1]:

import json
import pandas
import matplotlib as mlab
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.stats import expon as ex
import nltk
from datetime import datetime
import re
from sklearn import svm
import csv
import sys
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import cross_validation
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import linear_model
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import tree
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.externals.six import StringIO  
import pydot
import os
from sklearn import preprocessing
from sklearn.cross_validation import KFold
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
#building machine learning systems with python book - investigate


# In[2]:

reviews = {}
users = {}
businesses = {}
ids = []
features=[]
target = []
featurefile = "features_1417670000.26.csv"
targetfile = "target_1417669996.94.csv"


# In[7]:

business_file = open('/Users/Dania/Desktop/YelpDatasetChallenge/yelp_dataset_challenge_academic_dataset/data_business.json')
for line in business_file:
    biz = json.loads(line)
    biz_id = biz[u'business_id']
    businesses[biz_id] = biz
print(len(businesses))
business_file.close()


# In[3]:

review_file = open('/Users/Dania/Desktop/YelpDatasetChallenge/yelp_dataset_challenge_academic_dataset/data_review.json')
for line in review_file:
    review=json.loads(line)
    rev_id=review[u'review_id']
    ids.append(rev_id)
    reviews[rev_id]=review
print len(reviews)
review_file.close()    


# In[9]:

user_file = open('/Users/Dania/Desktop/YelpDatasetChallenge/yelp_dataset_challenge_academic_dataset/data_user.json')
for line in user_file:
    user=json.loads(line)
    u_id=user[u'user_id']
    users[u_id]=user
print len(users)    
user_file.close()    


# In[10]:

def reviewAgeCalc(day):
    dt=datetime.strptime(day,'%Y-%m-%d')
    return ((datetime.now().date()-dt.date()).total_seconds()/86400)


# In[4]:

#collect and tokenize text, further cleaning could be done here
corpus = []
for rid in ids:
    review = reviews[rid]
    text = (review[u'text'])
    text = text.lower()
    text = re.sub('\!+', "!", text)
    text = re.sub('\.+', ".", text)
    text = re.sub('\?+', "?", text)
    text = re.sub('(\\n)+', '\n', text)
    reviews[rid][u'text']=text
    corpus.append(text)
    
corpus = np.array(corpus)


# In[12]:

def reviewTextMetadata(text):
    #clean text
    tokens=nltk.word_tokenize(text)
    count_stop= tokens.count('?') + tokens.count(".") + tokens.count("!")
    count_stop=max(count_stop, 1)
    average_sentence_length=len(tokens)/count_stop
    review_length=len(tokens)
    paragraph_count=text.count('\\n')
    
    return review_length, average_sentence_length, paragraph_count


# Order for feature vector:
#     From review
#         1. Review Length
#         2. Average Sentence Length
#         3. Review Stars
#         4. Paragraph Count
#         5. Age of Review (days)    
#     From business
#         6. Overall Business Stars
#         7. Review Count
#     From business and review    
#         8. Difference between review stars and business stars
#     From user
#         9. Number of friends
#         10. Count of reviews

# In[17]:

#do once, then can bring back with csv files
def generateFeatureMatrix():
    features=[]
    target=[]
    for rid in ids:
        #from review
        review=reviews[rid]
        target.append(review[u'votes'][u'useful']) #adding to target vector
        reviewLength, reviewAverageSentenceLength, reviewParagraphCount = reviewTextMetadata(review[u'text'])
        reviewStars = review[u'stars']
        reviewAge = reviewAgeCalc(review[u'date'])
        #from business
        bid=review[u'business_id']
        business=businesses[bid]
        businessStars=business[u'stars']
        businessReviewCount=business[u'review_count']
        starDifference = (reviewStars-businessStars)
        #from user
        uid=review[u'user_id']
        userFriends=len(users[uid][u'friends'])
        userReviewCount=users[uid][u'review_count']
        featureArray=[reviewLength,
                      reviewAverageSentenceLength, 
                      reviewStars,
                      reviewParagraphCount,
                      reviewAge,
                      businessStars, 
                      businessReviewCount, 
                      starDifference,
                      userFriends,
                      userReviewCount]
        features.append(featureArray)
        if(len(features)!=len(target)):
            print len(features)
            print len (target)
            print review
            break

    print len(features) 
    
    ts=time.time()
    featuresFilename = 'features_' + str(ts) + ".csv"
    f=open(featuresFilename, 'wb')
    writer=csv.writer(f)
    writer.writerows(features)
    f.close()
    print featuresFilename
    
    targetFilename = 'target_' + str(ts) + ".csv"
    writeTargetFile(target, targetFilename)


# In[5]:

def writeTargetFile(targetVector, filename):
    ts=time.time() 
    f=open(filename, 'wb')
    target1=target[0:10]
    writer=csv.writer(f, quoting=csv.QUOTE_ALL)
    writer.writerow(targetVector)
    f.close()
    print filename


# In[6]:

def importFeatureMatrixFromCSV(featurefile, targetfile):
    features=[]
    with open(featurefile, 'rb') as f:
        reader=csv.reader(f)
        for row in reader:
            frow=[]
            for item in row:
                f= float(item)
                frow.append(f)
            features.append(frow)
    print len(features) #confirm correct functionality 
        
    target = importTargetVectorFromCSV(targetfile)
    
    return features, target


# In[7]:

def importTargetVectorFromCSV(targetfile):
    target=[]
    with open(targetfile, 'rb') as f:
        reader=csv.reader(f)
        for row in reader:
            for item in row:
                t= int(item)
                target.append(t)
    print len(target) #confirm correct functionality
    
    return target


# In[8]:

features, target = importFeatureMatrixFromCSV(featurefile, targetfile)


# In[9]:

#Constants for data partition
TOTAL_REVIEW_COUNT = 1125458-1
FIRST_QUARTER= 281354  
FIRST_THIRD = 375152  
HALF=562729   
TWO_THIRDS = 750305 
THIRD_QUARTER = 844093
FOUR_FIFTHS = 894740
Column_titles = ['revlength', 'avsentlength', 'revstar', 'paragraphcount', 'age', 'bizstar', 'bizrevcount', 'stardiff', 'numfriends', 'userReviews']


# Write features to CSV files for safe keeping

# In[10]:

#create target file that treats useful as binary - ie. a review receives any number of useful votes or it doesn't
targetBinary=[]
for t in target:
    if t>0:
        targetBinary.append(1)
    else:
        targetBinary.append(0)
print len(targetBinary)


# In[11]:

featureMatrix = np.mat(features)

featuresTrain = featureMatrix[0:TWO_THIRDS-1]
featuresTest = featureMatrix[TWO_THIRDS:TOTAL_REVIEW_COUNT]
targetTrain = target[0:TWO_THIRDS-1]
targetTest = target[TWO_THIRDS: TOTAL_REVIEW_COUNT]
targetBinaryTrain = targetBinary[0:TWO_THIRDS-1]
targetBinaryTest = targetBinary[TWO_THIRDS: TOTAL_REVIEW_COUNT]

featuresSplitTest = featureMatrix[TWO_THIRDS: FOUR_FIFTHS-1]
targetBinarySplitTest = targetBinary[TWO_THIRDS: FOUR_FIFTHS-1]
featuresSplitValidation = featureMatrix[FOUR_FIFTHS: TOTAL_REVIEW_COUNT]
targetBinarySplitValidation = targetBinary[FOUR_FIFTHS: TOTAL_REVIEW_COUNT]

corpusTrain = corpus[0:TWO_THIRDS-1]
corpusTest = corpus[TWO_THIRDS:TOTAL_REVIEW_COUNT]
corpusSplitTest = corpus[TWO_THIRDS:FOUR_FIFTHS-1]
corpusSplitValidation = corpus[FOUR_FIFTHS:TOTAL_REVIEW_COUNT]


# In[12]:

corpusMiniTrain = corpusTrain[0:FIRST_THIRD-1]
targetBinaryMiniTrain = targetBinaryTrain[0:FIRST_THIRD-1]


# In[105]:

writeTargetFile(targetBinarySplitTest, "TargetTrainingforD3.csv")


# In[15]:

#train tree with no normalization
dtc = tree.DecisionTreeClassifier()
dtc.fit(featuresTrain, targetTrain)
dtc_test = dtc.predict(featuresTrain)
print(classification_report(targetTrain, dtc_test))
dtc_cross_test = dtc.predict(featuresTest)
print(classification_report(targetTest, dtc_cross_test))


# In[31]:

#train tree with no normalization and binary target
dtcb = tree.DecisionTreeClassifier()
dtcb.fit(featuresTrain, targetBinaryTrain)
dtc_test = dtcb.predict(featuresTrain)
print(classification_report(targetBinaryTrain, dtc_test))
dtc_cross_test = dtcb.predict(featuresTest)
print(classification_report(targetBinaryTest, dtc_cross_test))


# In[11]:

dtc_reducedFeatures = tree.DecisionTreeClassifier()
featureReducedTrain = np.delete(featuresTrain, np.s_[2:4],1)
featureReducedTest = np.delete(featuresTest, np.s_[2:4],1)

print np.shape(featureReducedTrain)
dtc_reducedFeatures.fit(featureReducedTrain, targetBinaryTrain)
dtcbr_test = dtc_reducedFeatures.predict(featureReducedTrain)
print classification_report(targetBinaryTrain, dtcbr_test)
dtcbr_cross_test = dtc_reducedFeatures.predict(featureReducedTest)
print classification_report(targetBinaryTest, dtcbr_cross_test)
print dtc_reducedFeatures.feature_importances_

dtcbr_


# In[107]:

dtc_normalized = tree.DecisionTreeClassifier()
#normalizedTrain = preprocessing.normalize(featuresTrain)
#normalizedTest = preprocessing.normalize(featuresTest)
dtc_normalized.fit(normalizedTrain, targetBinaryTrain)
dtc_normalized_test = dtc_normalized.predict(featuresTrain)
print classification_report(targetBinaryTrain, dtc_normalized_test)
dtc_normalized_cross_test = dtc_normalized.predict(normalizedTest)
print classification_report(targetBinaryTest, dtc_normalized_cross_test)
print dtc_normalized.feature_importances_

dtc_normalized_validate_test = dtc_normalized.predict(featuresSplitTest)
print classification_report(targetBinarySplitTest, dtc_normalized_validate_test)
writeTargetFile(dtc_normalized_validate_test, "DecisionTreeMetadata_NormalizedforD3.csv")


# In[35]:

#information about this machine
print dtcb
print dtcb.tree_.feature
print dtcb.tree_.value
print dtcb.feature_importances_
#This tree's top three features: Age of review, review length, number of friends of reviewer
#close fourth: number of total reviews for business
#no single feature is very telling.


# In[70]:

dtc_p1 = tree.DecisionTreeClassifier(max_depth=10)
dtc_p1.fit(featuresTrain, targetBinaryTrain)


# In[71]:

dtc_p1_test = dtc_p1.predict(featuresTrain)
print "Accuracy on training set"
print classification_report(targetBinaryTrain, dtc_p1_test)

print "Accuracy on full test set"
dtc_p1_cross_test = dtc_p1.predict(featuresTest)
print classification_report(targetBinaryTest, dtc_p1_cross_test)

print "Accuracy on half test set"
dtc_p1_validate_test = dtc_p1.predict(featuresSplitTest)
print classification_report(targetBinarySplitTest, dtc_p1_validate_test)

print "Accuracy on validation set"
dtc_p1_validate_val = dtc_p1.predict(featuresSplitValidation)
print classification_report(targetBinarySplitValidation, dtc_p1_validate_val)

print "Feature Importance"
print dtc_p1.feature_importances_

with open("/Users/Dania/Desktop/dtree.dot", 'w') as f:
    f = tree.export_graphviz(dtc_p1, out_file=f)


# In[106]:

writeTargetFile(dtc_p1_validate_test, "DecisionTreeMetadata_MaxDepthforD3.csv")


# In[39]:

#Bernoulli Naive-Bayes
#The Bernoulli version of Naive Bayes requires a binary outcome. 
bnb = BernoulliNB()
bnb.fit(featuresTrain, targetBinaryTrain)
bnb_test = bnb.predict(featuresTrain)
print(classification_report(targetBinaryTrain, bnb_test))
bnb_cross_test = bnb.predict(featuresTest)
print(classification_report(targetBinaryTest, bnb_cross_test))

bnb_validate_test = bnb.predict(featuresSplitTest)
print classification_report(targetBinarySplitTest, bnb_validate_test)

writeTargetFile(bnb_validate_test, )

print bnb
print bnb.coef_
print bnb.class_count_


# In[63]:

#Multinomial Naive Bayes throws an error whenever negative features are used. Currently, the Features Matrix contains 
#negative values in the difference field. This will be run with the absolute value of the Features Matrix
mnb = MultinomialNB()
absFeaturesTrain = np.absolute(featuresTrain)
absFeaturesTest = np.absolute(featuresTest)
mnb.fit(absFeaturesTrain, targetTrain)
mnb_test = mnb.predict(absFeaturesTrain)
print(classification_report(targetTrain, mnb_test))
mnb_cross_test = mnb.predict(absFeaturesTest)
print(classification_report(targetTest, mnb_cross_test))
#resulted in really terrible accuracy! Completely inappropriate algorithm here. Can't even predict the training set well!


# In[79]:

rfc = RandomForestClassifier(n_estimators = 8, max_depth = 10)
rfc.fit(featuresTrain, targetBinaryTrain)

rfc_test = rfc.predict(featuresTrain)
print "Accuracy on training set"
print classification_report(targetBinaryTrain, rfc_test)

print "Accuracy on full test set"
rfc_cross_test = rfc.predict(featuresTest)
print classification_report(targetBinaryTest, rfc_cross_test)


# In[ ]:




# In[71]:




# In[ ]:

#December 3, need to figure out what problem is here
lr = linear_model.LinearRegression()
lr.fit(featuresTrain, targetTrain)
lr_test = lr.predict(featuresTrain)
print(classification_report(targetTrain, lr_test))
lr_cross_test = lr.predict(featuresTest)
print(classification_report(targetTest, lr_cross_test))


# In[ ]:

#Support Vector Regressions - Run time too long
#svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_lin = SVR(kernel='linear', C=1e3)
svr_poly = SVR(kernel='poly', C=1e3, degree=2)
svr_rbf.fit(featuresTrain, targetTrain)
svr_rbf_test = svr_rbf.predict(featuresTrain)
print(classification_report(targetTrain, svr_rbf_test))
svr_rbf_cross_test = svr_rbf.predict(featuresTest)
print(classification_report(targetTest, svr_rbf_cross_test))


# In[ ]:


















# This part of the code looks to build classifiers solely from natural language processesing

# In[13]:

class LemmaTokenizer(object):
  def __init__(self):
    self.wnl = WordNetLemmatizer()
  def __call__(self, doc):
    return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

vectorizer = CountVectorizer(tokenizer = LemmaTokenizer(), stop_words ='english', min_df = 1)
transformer = TfidfTransformer()
sgd = linear_model.SGDClassifier()


# In[14]:

pipeline_mnb = Pipeline([('vectorizer',  CountVectorizer(min_df=1, stop_words='english')), ('transformer', TfidfTransformer()), 
  ('classifier',  MultinomialNB()) ])

pipeline_mnb.fit(corpusMiniTrain, targetBinaryMiniTrain)
score = pipeline_mnb.score(corpusSplitTest, targetBinarySplitTest)
mnb_validate_test = pipeline_mnb.predict(corpusSplitTest)


# In[25]:




# In[25]:

print score
print classification_report(targetBinarySplitTest, mnb_validate_test)


# In[26]:

pipeline_bnb = Pipeline([('vectorizer',  CountVectorizer(min_df=1, stop_words='english')), ('transformer', TfidfTransformer()), 
  ('classifier',  BernoulliNB()) ])

pipeline_bnb.fit(corpusMiniTrain, targetBinaryMiniTrain)
score_bnb = pipeline_mnb.score(corpusSplitTest, targetBinarySplitTest)
text_bnb_validate_test = pipeline_mnb.predict(corpusSplitTest)

print score_bnb
print classification_report(targetBinarySplitTest, text_bnb_validate_test)


# In[35]:

writeTargetFile(text_bnb_validate_test, "TextBernoulliNBforD3.csv")


# In[30]:

bnb_final_vect = CountVectorizer(min_df=1, stop_words='english')
bnb_final_model =  BernoulliNB()
pipeline_bnb2 = Pipeline([('vectorizer', bnb_final_vect), ('transformer', TfidfTransformer()), 
  ('classifier', bnb_final_model) ])

pipeline_bnb2.fit(corpusMiniTrain, targetBinaryMiniTrain)
score_bnb2 = pipeline_bnb2.score(corpusSplitTest, targetBinarySplitTest)
text_bnb2_validate_test = pipeline_bnb2.predict(corpusSplitTest)

print score_bnb2
print classification_report(targetBinarySplitTest, text_bnb2_validate_test)


# In[34]:

print len(bnb_final_vect.vocabulary_)
print type(bnb_final_vect.vocabulary_)
for 


# In[79]:

#takes 16 hours to train. Impractacle
pipeline_sgd = Pipeline([('vectorizer', vectorizer), ('transformer', transformer), ('sgd', sgd)]) 

svmtarget = np.array(targetBinary[0:250000])
svmcorpus = np.array(corpus[0:250000])

vectorizersvm = CountVectorizer(stop_words ='english', min_df = 1)
pipeline_svc = Pipeline([('vectorizer', vectorizersvm), ('transformer', transformer), ('svc', svm)])

#pipeline_svc.fit(svmcorpus, svmtarget)

score = pipeline_svc.score(npcorpustest, nptargettest)
print score


# In[114]:

vectorizerTree = CountVectorizer(tokenizer = LemmaTokenizer(), stop_words ='english', min_df = 1, max_features = 50000)
text_dtc = tree.DecisionTreeClassifier(max_depth = 10000)
#pipeline_text_dtc = Pipeline([('vectorizer', vectorizerTree), ('text_dtc', text_dtc)])

#pipeline_text_dtc.fit(corpusMiniTrain, targetBinaryMiniTrain)

#score = pipeline_text_dtc.score(corpusSplitTest, targetBinarySplitTest)
#print score

text_dtc_test = pipeline_text_dtc.predict(text_dtc_test)
print classification_report(targetBinaryTest, text_dtc_test)


# In[15]:

vectorizerTree = CountVectorizer(tokenizer = LemmaTokenizer(), stop_words ='english', min_df = 1, max_features = 5000)
text_dtc = tree.DecisionTreeClassifier(max_depth = 10000)
dictionary = vectorizerTree.fit_transform(corpusMiniTrain)


# In[18]:

#print vectorizerTree
documentTermMatrix_corpusValTest = vectorizerTree.transform(corpusSplitTest)
documentTermMatrix_corpusValTest = documentTermMatrix_corpusValTest.toarray()
print type(documentTermMatrix_corpusValTest)


# In[16]:

#SPARSEdocumentTermMatrix_corpusMiniTrain = vectorizerTree.transform(corpusMiniTrain)

#print type(SPARSEdocumentTermMatrix_corpusMiniTrain)
print type(dictionary)
print dictionary


# In[17]:

text_dtc = tree.DecisionTreeClassifier(max_depth = 5000)
documentTermMatrix_corpusMiniTrain = dictionary.toarray()
print type(documentTermMatrix_corpusMiniTrain)
print documentTermMatrix_corpusMiniTrain
text_dtc.fit(documentTermMatrix_corpusMiniTrain, targetBinaryMiniTrain)
print text_dtc


# In[19]:

dtc_validate_test = text_dtc.predict(documentTermMatrix_corpusValTest)
print classification_report(targetBinarySplitTest, dtc_validate_test)


# In[20]:

writeTargetFile(dtc_validate_test, "DecisionTreeText_5000features.csv")


# In[109]:

#too big to train. Feature set is too large for these conditions
text_dtc = tree.DecisionTreeClassifier(max_depth = 5000)
pipeline_text_dtc = Pipeline([('vectorizer', vectorizer), ('transformer', transformer), ('text_dtc', text_dtc)])

#pipeline_text_dtc.fit(corpusTrain, targetBinaryTrain)

score = pipeline_text_dtc.score(corpusSplitTest, targetBinarySplitTest)
print score


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[71]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



