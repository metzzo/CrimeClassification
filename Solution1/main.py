__author__ = 'rfischer'


from sknn.mlp import Classifier, Layer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm

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
    pipeline = pickle.load(open('nn.pkl', 'rb'))

    print("Load CSV")
    data = pandas.read_csv('../Data/test.csv', sep=',')

    print("Convert CSV")
    x_train = fetchData(data)

    result = pipeline.predict_proba(x_train)
    print(result)

    indices = np.array([[x] for x in range(0, result.shape[0])])
    result = np.append(indices, result, axis=1)

    with open('result.csv', 'wb') as f:
        f.write('Id,ARSON,ASSAULT,BAD CHECKS,BRIBERY,BURGLARY,DISORDERLY CONDUCT,DRIVING UNDER THE INFLUENCE,DRUG/NARCOTIC,DRUNKENNESS,EMBEZZLEMENT,EXTORTION,FAMILY OFFENSES,FORGERY/COUNTERFEITING,FRAUD,GAMBLING,KIDNAPPING,LARCENY/THEFT,LIQUOR LAWS,LOITERING,MISSING PERSON,NON-CRIMINAL,OTHER OFFENSES,PORNOGRAPHY/OBSCENE MAT,PROSTITUTION,RECOVERED VEHICLE,ROBBERY,RUNAWAY,SECONDARY CODES,SEX OFFENSES FORCIBLE,SEX OFFENSES NON FORCIBLE,STOLEN PROPERTY,SUICIDE,SUSPICIOUS OCC,TREA,TRESPASS,VANDALISM,VEHICLE THEFT,WARRANTS,WEAPON LAWS\n')
        np.savetxt(f, result, delimiter=",", fmt='%i,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f')

def learn():
    print("Load CSV")
    data = pandas.read_csv('../Data/train.csv', sep=',')

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

    print("Convert to Array of Array")

    # Features
    # Location (Long/Lat), District, Time, Resolution

    print("Learn")

    # Train SVM
    pipeline = Pipeline([
            ('min/max scaler', MinMaxScaler(feature_range=(-1.0, 1.0))),
            ('svc',  svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0))
    ])
    pipeline.fit(x_train, categories)


    # Train Neuralnetwork
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


    # Train Nearest Neighbor

    pickle.dump(pipeline, open('nn.pkl', 'wb'))

# predict()
learn()