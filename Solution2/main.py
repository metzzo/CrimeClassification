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

classes = [
    "ARSON","ASSAULT","BAD CHECKS","BRIBERY","BURGLARY","DISORDERLY CONDUCT","DRIVING UNDER THE INFLUENCE",
    "DRUG/NARCOTIC","DRUNKENNESS","EMBEZZLEMENT","EXTORTION","FAMILY OFFENSES","FORGERY/COUNTERFEITING",
    "FRAUD","GAMBLING","KIDNAPPING","LARCENY/THEFT","LIQUOR LAWS","LOITERING","MISSING PERSON",
    "NON-CRIMINAL","OTHER OFFENSES","PORNOGRAPHY/OBSCENE MAT","PROSTITUTION","RECOVERED VEHICLE",
    "ROBBERY","RUNAWAY","SECONDARY CODES","SEX OFFENSES FORCIBLE","SEX OFFENSES NON FORCIBLE",
    "STOLEN PROPERTY","SUICIDE","SUSPICIOUS OCC","TREA","TRESPASS","VANDALISM","VEHICLE THEFT",
    "WARRANTS","WEAPON LAWS"
]
financialCrimes = ["ASSAULT", "BURGLARY", "EMBEZZLEMENT", "EXTORTION", "FORGERY/COUNTERFEITING", "FRAUD", "GAMBLING", "LARCENY/THEFT", "ROBBERY", "STOLEN PROPERTY", "VEHICLE THEFT"]
drugCrimes = ["DRIVING UNDER THE INFLUENCE", "DRUG/NARCOTIC", "DRUNKENNESS", "LIQUOR LAWS"]
violentCrimes = ["ARSON", "ASSAULT", "FAMILY OFFENSES", "SEX OFFENSES FORCIBLE", "KIDNAPPING", "SUICIDE", "VANDALISM", "EXTORTION"]
miscCrimes = ["BAD CHECKS","BRIBERY", "DISORDERLY CONDUCT", "LOITERING","MISSING PERSON","NON-CRIMINAL","OTHER OFFENSES","PORNOGRAPHY/OBSCENE MAT","PROSTITUTION","RECOVERED VEHICLE", "RUNAWAY","SECONDARY CODES","SEX OFFENSES NON FORCIBLE", "SUSPICIOUS OCC","TREA","TRESPASS" "WARRANTS","WEAPON LAWS"]
crimeTypes = [financialCrimes, drugCrimes, violentCrimes, miscCrimes]


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
    financial = pickle.load(open('financial_nn.pkl', 'rb'))
    drug = pickle.load(open('drug_nn.pkl', 'rb'))
    violence = pickle.load(open('violence_nn.pkl', 'rb'))
    misc = pickle.load(open('misc_nn.pkl', 'rb'))
    decider_classificator = pickle.load(open('category_decider.pkl', 'rb'))


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


    print("Should categorize?")
    response = sys.stdin.readline().lower()[0]
    if response == "y":
        category_decision = decider_classificator.predict(x_train)
        pickle.dump(category_decision, open('predict_category.pkl', 'wb'))
    else:
        category_decision = pickle.load(open('predict_category.pkl', 'rb'))

    print("Predicting")

    result = []

    for i in range(0, len(x_train)):
        if i > 0 and i % 1000 == 0:
            print (i, " / ", len(x_train))

        category = category_decision[i]
        tmpMapResult = {}
        for c in classes:
            tmpMapResult[c] = 0

        if category == 0:
            predictResult = financial.predict_proba(np.array([x_train[i]]))[0]
            tmpClasses = financialCrimes
        elif category == 1:
            predictResult = drug.predict_proba(np.array([x_train[i]]))[0]
            tmpClasses = drugCrimes
        elif category == 2:
            predictResult = violence.predict_proba(np.array([x_train[i]]))[0]
            tmpClasses = violentCrimes
        elif category == 3:
            predictResult = misc.predict_proba(np.array([x_train[i]]))[0]
            tmpClasses = miscCrimes

        for index, r in enumerate(predictResult):
            tmpMapResult[tmpClasses[index]] = r

        tmpResult = [0] * len(classes);

        for index, c in enumerate(classes):
            tmpResult[index] = tmpMapResult[c]

        result.append(np.array(tmpResult))

    print("Preparing")

    result = np.array(result)

    indices = np.array([[x] for x in range(0, result.shape[0])])
    result = np.append(indices, result, axis=1)

    with open('result.csv', 'wb') as f:
        f.write('Id,ARSON,ASSAULT,BAD CHECKS,BRIBERY,BURGLARY,DISORDERLY CONDUCT,DRIVING UNDER THE INFLUENCE,DRUG/NARCOTIC,DRUNKENNESS,EMBEZZLEMENT,EXTORTION,FAMILY OFFENSES,FORGERY/COUNTERFEITING,FRAUD,GAMBLING,KIDNAPPING,LARCENY/THEFT,LIQUOR LAWS,LOITERING,MISSING PERSON,NON-CRIMINAL,OTHER OFFENSES,PORNOGRAPHY/OBSCENE MAT,PROSTITUTION,RECOVERED VEHICLE,ROBBERY,RUNAWAY,SECONDARY CODES,SEX OFFENSES FORCIBLE,SEX OFFENSES NON FORCIBLE,STOLEN PROPERTY,SUICIDE,SUSPICIOUS OCC,TREA,TRESPASS,VANDALISM,VEHICLE THEFT,WARRANTS,WEAPON LAWS\n')
        np.savetxt(f, result, delimiter=",", fmt='%i,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f')

def learnSVM(crimes, x_train, data):
    pipeline = Pipeline([
            ('min/max scaler', MinMaxScaler(feature_range=(-1.0, 1.0))),
            #('svc',  BaggingClassifier(svm.SVC(C=1.0, cache_size=200, coef0=0.0, verbose=True),max_samples=10000)) #c=1, max_samples=10000
            #('svc',  svm.LinearSVC(C=.1, verbose=True)) #c=1, max_samples=10000
            ('neural network', Classifier(
                    layers=[
                        Layer("Sigmoid", units=40),
                        Layer("Softmax")
                    ],
                    learning_rate=0.001,
                    n_iter=25
            ))
    ])
    pipeline.fit(x_train, np.array([1 if x in crimes else 0 for x in data["Category"]]))
    #pipeline.fit(x_train, [1 if x in crimes else 0 for x in data["Category"]])
    return pipeline

def learnDecisionNN(crimes, x_train, data):
    pipeline = Pipeline([
            ('min/max scaler', MinMaxScaler(feature_range=(-1.0, 1.0))),
            #('svc',  BaggingClassifier(svm.SVC(C=1.0, cache_size=200, coef0=0.0, verbose=True),max_samples=10000)) #c=1, max_samples=10000
            #('svc',  svm.LinearSVC(C=.1, verbose=True)) #c=1, max_samples=10000
            ('neural network', Classifier(
                    layers=[
                        Layer("Sigmoid", units=40),
                        Layer("Softmax")
                    ],
                    learning_rate=0.001,
                    n_iter=100
            ))
    ])
    pipeline.fit(x_train, np.array([1 if x in crimes[1] else (2 if x in crimes[2] else (0 if x in crimes[0] else 3)) for x in data["Category"]]))
    #pipeline.fit(x_train, [1 if x in crimes else 0 for x in data["Category"]])
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

    print("Do you want to learn category decider?")
    response = sys.stdin.readline().lower()[0]
    if response == "y":
        print("Do you want to test category decider?")
        response = sys.stdin.readline().lower()[0]
        if response == "y":
            decider_classificator = learnDecisionNN(crimeTypes, x_train[:len(x_train)/3*2], data[:len(data)/3*2])
            pickle.dump(decider_classificator, open('category_decider_test.pkl', 'wb'))
            result = decider_classificator.predict(x_train[-len(data)/3*1:])
            expected = [1 if x in drugCrimes else (2 if x in violentCrimes else (0 if x in financialCrimes else 3)) for x in data["Category"][-len(data)/3*1:]]
            #expected = [1 if x in financialCrimes else 0 for x in data["Category"]]
            correct = 0
            for index, row in enumerate(result):
                if row == expected[index]:
                    correct += 1

            print "Correct ", correct, len(result)
        else:
            decider_classificator = learnDecisionNN(crimeTypes, x_train, data)
            pickle.dump(decider_classificator, open('category_decider.pkl', 'wb'))

    print("Do you want to learn misc NN?")
    response = 'y' # sys.stdin.readline().lower()[0]
    if response == "y":
        misc_xtrain, misc_categories = filterByCategory(x_train=x_train, categories=categories, data=data,possibleCategory=miscCrimes)
        misc_pipeline = Pipeline([
                ('min/max scaler', MinMaxScaler(feature_range=(-1.0, 1.0))),
                ('neural network',  Classifier(
                    layers=[
                        Layer("Sigmoid", units=40),
                        Layer("Softmax")
                    ],
                    learning_rate=0.001,
                    n_iter=100
                ))
        ])
        misc_pipeline.fit(misc_xtrain, misc_categories)
        pickle.dump(misc_pipeline, open('misc_nn.pkl', 'wb'))

    print("Do you want to learn financial NN?")
    response = 'y' # sys.stdin.readline().lower()[0]
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
                    n_iter=100
                ))
        ])
        financial_pipeline.fit(financial_xtrain, financial_categories)
        pickle.dump(financial_pipeline, open('financial_nn.pkl', 'wb'))


    print("Do you want to learn drug NN?")
    response = 'y' # sys.stdin.readline().lower()[0]
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
                    n_iter=100
                ))
        ])
        drug_pipeline.fit(drug_xtrain, drug_categories)
        pickle.dump(drug_pipeline, open('drug_nn.pkl', 'wb'))

    print("Do you want to learn violence NN?")
    response = 'y' # sys.stdin.readline().lower()[0]
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
                    n_iter=100
                ))
        ])
        violence_pipeline.fit(violence_xtrain, violence_categories)
        pickle.dump(violence_pipeline, open('violence_nn.pkl', 'wb'))


    print("Finished learning - Hooray!")


#learn()
predict()
