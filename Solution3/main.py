__author__ = 'rfischer'


from sknn.mlp import Classifier, Layer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from sklearn.ensemble import BaggingClassifier

import pandas
import numpy as np
import time
import datetime
import logging
import sys
import pickle

logging.basicConfig(
            format="%(message)s",
            level=logging.DEBUG,
            stream=sys.stdout)



def fetchData(data):
    dates = [int(time.mktime(datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timetuple())) for s in data["Dates"]]
    locationX = [int(x*10000000000000) for x in data["X"]]
    locationY = [int(y*10000000000000) for y in data["Y"]]
    districts = list(set(data["PdDistrict"]))

    x_train = []
    for i in range(0, len(dates)):
        x_train.append([dates[i], locationX[i], locationY[i], districts.index(data["PdDistrict"][i])])

    x_train = np.array(x_train)

    return x_train

def predict():
    pipelines = [
            pickle.load(open('financial_nn.pkl', 'rb')),
            pickle.load(open('drug_nn.pkl', 'rb')),
            pickle.load(open('violence_nn.pkl', 'rb')),
            pickle.load(open('financial_svm.pkl', 'rb')),
            pickle.load(open('drug_svm.pkl', 'rb')),
            pickle.load(open('violent_svm.pkl', 'rb')),
            pickle.load(open('nn.pkl', 'rb'))]

    decision_pipeline = pickle.load(open('decision_nn.pkl', 'rb'))

    print("You want to convert new one?")
    response = sys.stdin.readline().lower()[0]
    if response == "y":
        print("Load CSV")
        data = pandas.read_csv('../Data/test.csv', sep=',')

        print("Convert CSV")
        x_train = fetchData(data)
        pickle.dump(x_train, open('train_xtrain.pkl', 'wb'))
    else:
        x_train = pickle.load(open('train_xtrain.pkl', 'rb'))


    print("Should pre Predict?")
    response = sys.stdin.readline().lower()[0]
    if response == "y":
        newXTrain = []
        for i in range(0, len(x_train)):
            if i > 0 and i % 1000 == 0:
                print (i, " / ", len(x_train), i/len(x_train)*100)

            tmp = []
            input = np.array([x_train[i]])
            for pipeline in pipelines:
                result = pipeline.predict_proba(input).tolist()
                if hasattr(result, "__len__"):
                    result = result[0]
                tmp.extend(result)

            newXTrain.append(x_train[i].tolist() + tmp)
        pickle.dump(newXTrain, open('predict_xtrain2.pkl', 'wb'))
    else:
        newXTrain = pickle.load(open('predict_xtrain2.pkl', 'rb'))

    print("Predicting")
    result = decision_pipeline.predict_proba(np.array(newXTrain))

    print("Preparing")

    indices = np.array([[x] for x in range(0, result.shape[0])])
    result = np.append(indices, result, axis=1)

    with open('result.csv', 'wb') as f:
        f.write('Id,ARSON,ASSAULT,BAD CHECKS,BRIBERY,BURGLARY,DISORDERLY CONDUCT,DRIVING UNDER THE INFLUENCE,DRUG/NARCOTIC,DRUNKENNESS,EMBEZZLEMENT,EXTORTION,FAMILY OFFENSES,FORGERY/COUNTERFEITING,FRAUD,GAMBLING,KIDNAPPING,LARCENY/THEFT,LIQUOR LAWS,LOITERING,MISSING PERSON,NON-CRIMINAL,OTHER OFFENSES,PORNOGRAPHY/OBSCENE MAT,PROSTITUTION,RECOVERED VEHICLE,ROBBERY,RUNAWAY,SECONDARY CODES,SEX OFFENSES FORCIBLE,SEX OFFENSES NON FORCIBLE,STOLEN PROPERTY,SUICIDE,SUSPICIOUS OCC,TREA,TRESPASS,VANDALISM,VEHICLE THEFT,WARRANTS,WEAPON LAWS\n')
        np.savetxt(f, result, delimiter=",", fmt='%i,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f')

def learnSVM(crimes, x_train, data):
    pipeline = Pipeline([
            ('min/max scaler', MinMaxScaler(feature_range=(-1.0, 1.0))),
            ('svc',  BaggingClassifier(svm.SVC(C=1.0, cache_size=200, class_weight="balanced", coef0=0.0, verbose=True),max_samples=10000))
    ])
    pipeline.fit(x_train, [1 if x in crimes else 0 for x in data["Category"]])
    return pipeline

def filterByCategory(x_train, categories, data, possibleCategory):
    newXTrain = []
    newCategories = []
    for i in range(0, len(x_train)):
        if data["Category"][i] in possibleCategory:
            newXTrain.append(x_train[i])
            newCategories.append(categories[i])

    newXTrain = np.array(newXTrain)
    newCategories = np.array(newCategories)

    return newXTrain, newCategories

def learn():
    financialCrimes = ["ASSAULT", "BURGLARY", "EMBEZZLEMENT", "EXTORTION", "FORGERY/COUNTERFEITING", "FRAUD", "GAMBLING", "LARCENY/THEFT", "ROBBERY", "STOLEN PROPERTY", "VEHICLE THEFT"]
    drugCrimes = ["DRIVING UNDER THE INFLUENCE", "DRUG/NARCOTIC", "DRUNKENNESS", "LIQUOR LAWS"]
    violentCrimes = ["ARSON", "ASSAULT", "FAMILY OFFENSES", "SEX OFFENSES FORCIBLE", "KIDNAPPING", "SUICIDE", "VANDALISM", "EXTORTION"]

    print("Load CSV")
    data = pandas.read_csv('../Data/train.csv', sep=',')

    print("Do you want to prepare data or use existing prepared data?")
    response = sys.stdin.readline().lower()[0]

    if response == "y":
        print("Convert CSV")

        classes = [
            "ARSON","ASSAULT","BAD CHECKS","BRIBERY","BURGLARY","DISORDERLY CONDUCT","DRIVING UNDER THE INFLUENCE",
            "DRUG/NARCOTIC","DRUNKENNESS","EMBEZZLEMENT","EXTORTION","FAMILY OFFENSES","FORGERY/COUNTERFEITING",
            "FRAUD","GAMBLING","KIDNAPPING","LARCENY/THEFT","LIQUOR LAWS","LOITERING","MISSING PERSON",
            "NON-CRIMINAL","OTHER OFFENSES","PORNOGRAPHY/OBSCENE MAT","PROSTITUTION","RECOVERED VEHICLE",
            "ROBBERY","RUNAWAY","SECONDARY CODES","SEX OFFENSES FORCIBLE","SEX OFFENSES NON FORCIBLE",
            "STOLEN PROPERTY","SUICIDE","SUSPICIOUS OCC","TREA","TRESPASS","VANDALISM","VEHICLE THEFT",
            "WARRANTS","WEAPON LAWS"
        ]
        categories = np.array([classes.index(x) for x in data["Category"]])
        x_train = fetchData(data)

        pickle.dump(x_train, open('x_train.pkl', 'wb'))
        pickle.dump(categories, open('categories.pkl', 'wb'))
    else:
        print("Load prepared data")
        x_train = pickle.load(open('x_train.pkl', 'rb'))
        categories = pickle.load(open('categories.pkl', 'rb'))

    print("Do you want to learn SVMs?")
    response = sys.stdin.readline().lower()[0]
    if response == "y":
        print("Learn Decision SVMs")
        # Train SVM for financial crimes
        financial_svm = learnSVM(financialCrimes, x_train, data)
        pickle.dump(financial_svm, open('financial_svm.pkl', 'wb'))

        drug_svm = learnSVM(drugCrimes, x_train, data)
        pickle.dump(drug_svm, open('drug_svm.pkl', 'wb'))

        violent_svm = learnSVM(violentCrimes, x_train, data)
        pickle.dump(violent_svm, open('violent_svm.pkl', 'wb'))


    print("Do you want to learn financial NN?")
    response = sys.stdin.readline().lower()[0]
    if response == "y":
        financial_xtrain, financial_categories = filterByCategory(x_train=x_train, categories=categories, data=data,possibleCategory=financialCrimes)
        financial_pipeline = Pipeline([
                ('min/max scaler', MinMaxScaler(feature_range=(-1.0, 1.0))),
                ('neural network',  Classifier(
                    layers=[
                        Layer("Sigmoid", units=40),
                        Layer("Softmax")
                    ],
                    learning_rate=0.001,
                    n_iter=25
                ))
        ])
        financial_pipeline.fit(financial_xtrain, financial_categories)
        pickle.dump(financial_pipeline, open('financial_nn.pkl', 'wb'))


    print("Do you want to learn drug NN?")
    response = sys.stdin.readline().lower()[0]
    if response == "y":
        drug_xtrain, drug_categories = filterByCategory(x_train=x_train, categories=categories, data=data,possibleCategory=drugCrimes)
        drug_pipeline = Pipeline([
                ('min/max scaler', MinMaxScaler(feature_range=(-1.0, 1.0))),
                ('neural network',  Classifier(
                    layers=[
                        Layer("Sigmoid", units=40),
                        Layer("Softmax")
                    ],
                    learning_rate=0.001,
                    n_iter=25
                ))
        ])
        drug_pipeline.fit(drug_xtrain, drug_categories)
        pickle.dump(drug_pipeline, open('drug_nn.pkl', 'wb'))

    print("Do you want to learn violence NN?")
    response = sys.stdin.readline().lower()[0]
    if response == "y":
        violence_xtrain, violence_categories = filterByCategory(x_train=x_train, categories=categories, data=data,possibleCategory=violentCrimes)
        violence_pipeline = Pipeline([
                ('min/max scaler', MinMaxScaler(feature_range=(-1.0, 1.0))),
                ('neural network',  Classifier(
                    layers=[
                        Layer("Sigmoid", units=40),
                        Layer("Softmax")
                    ],
                    learning_rate=0.001,
                    n_iter=25
                ))
        ])
        violence_pipeline.fit(violence_xtrain, violence_categories)
        pickle.dump(violence_pipeline, open('violence_nn.pkl', 'wb'))

    print("Do you want to learn fallback NN?")
    response = sys.stdin.readline().lower()[0]
    if response == "y":
        pipeline = Pipeline([
                ('min/max scaler', MinMaxScaler(feature_range=(-1.0, 1.0))),
                ('neural network',  Classifier(
                    layers=[
                        Layer("Sigmoid", units=40),
                        Layer("Softmax")
                    ],
                    learning_rate=0.001,
                    n_iter=25
                ))
        ])
        pipeline.fit(x_train, categories)
        pickle.dump(pipeline, open('nn.pkl', 'wb'))

    print("Do you want to learn Decision NN?")
    response = sys.stdin.readline().lower()[0]
    if response == "y":
        print("Load Classifiers")

        pipelines = [
            pickle.load(open('financial_nn.pkl', 'rb')),
            pickle.load(open('drug_nn.pkl', 'rb')),
            pickle.load(open('violence_nn.pkl', 'rb')),
            pickle.load(open('financial_svm.pkl', 'rb')),
            pickle.load(open('drug_svm.pkl', 'rb')),
            pickle.load(open('violent_svm.pkl', 'rb')),
            pickle.load(open('nn.pkl', 'rb'))]

        print ("Build Classifier Data")
        newXTrain = [];
        response = sys.stdin.readline().lower()[0]
        if response == "y":
            for i in range(0, len(x_train)):
                if i > 0 and i % 1000 == 0:
                    print (i, " / ", len(x_train), i/len(x_train)*100)

                tmp = []
                input = np.array([x_train[i]])
                for pipeline in pipelines:
                    result = pipeline.predict_proba(input).tolist()
                    if hasattr(result, "__len__"):
                        result = result[0]
                    tmp.extend(result)

                newXTrain.append(x_train[i].tolist() + tmp)
            pickle.dump(newXTrain, open('newXTrain.pkl', 'wb'))
        else:
            newXTrain = pickle.load(open('newXTrain.pkl', 'rb'))
            if 0:
                i = len(x_train) - 1
                tmp = []
                input = np.array([x_train[i]])
                for pipeline in pipelines:
                    result = pipeline.predict_proba(input).tolist()
                    if hasattr(result, "__len__"):
                        result = result[0]
                    tmp.extend(result)

                newXTrain.append(x_train[i].tolist() + tmp)

        print len(newXTrain), len(categories)
        x_train = np.array(newXTrain)

        print("Train Decision Classifier")

        decision_pipeline = Pipeline([
                ('min/max scaler', MinMaxScaler(feature_range=(-1.0, 1.0))),
                ('neural network',  Classifier(
                    layers=[
                        Layer("Sigmoid", units=70),
                        Layer("Softmax")
                    ],
                    learning_rate=0.001,
                    n_iter=100
                ))
        ])
        decision_pipeline.fit(x_train, categories)
        pickle.dump(decision_pipeline, open('decision_nn.pkl', 'wb'))


    print("Finished learning - Hooray!")


def test():
    pass
#predict()
#test()
learn()