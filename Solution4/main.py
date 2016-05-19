__author__ = 'rfischer'

from sknn.mlp import Classifier, Layer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn import neighbors
from sklearn import svm
from sklearn.ensemble import BaggingClassifier
import operator

import pandas
import numpy as np
import time
import datetime
import logging
import sys
import pickle
import math

logging.basicConfig(
            format="%(message)s",
            level=logging.DEBUG,
            stream=sys.stdout)

print("Loading data... ")
trainData = pandas.read_csv('../Data/train.csv', sep=',')
testData = pandas.read_csv('../Data/test.csv', sep=',')

# allData = pandas.concat([trainData, testData])

classes = [
    "ARSON","ASSAULT","BAD CHECKS","BRIBERY","BURGLARY","DISORDERLY CONDUCT","DRIVING UNDER THE INFLUENCE",
    "DRUG/NARCOTIC","DRUNKENNESS","EMBEZZLEMENT","EXTORTION","FAMILY OFFENSES","FORGERY/COUNTERFEITING",
    "FRAUD","GAMBLING","KIDNAPPING","LARCENY/THEFT","LIQUOR LAWS","LOITERING","MISSING PERSON",
    "NON-CRIMINAL","OTHER OFFENSES","PORNOGRAPHY/OBSCENE MAT","PROSTITUTION","RECOVERED VEHICLE",
    "ROBBERY","RUNAWAY","SECONDARY CODES","SEX OFFENSES FORCIBLE","SEX OFFENSES NON FORCIBLE",
    "STOLEN PROPERTY","SUICIDE","SUSPICIOUS OCC","TREA","TRESPASS","VANDALISM","VEHICLE THEFT",
    "WARRANTS","WEAPON LAWS"
]
print ("Do you want to prepare new data?")
prepareNewData = sys.stdin.readline().lower()[0] == 'y'

print ("Do you want to train category classifier?")
trainCategoryClassifier = sys.stdin.readline().lower()[0] == 'y'

print ("Do you want to train top crime classifier?")
trainTopCrimeClassifier = sys.stdin.readline().lower()[0] == 'y'

print ("Do you want to train not top crime classifier?")
trainNotTopCrimeClassifier = sys.stdin.readline().lower()[0] == 'y'

print ("Do you want to calculate crime histograms?")
calcCrimeHistograms = sys.stdin.readline().lower()[0] == 'y'

print ("Do you want to train nearest neighbor?")
trainNearestNeighbor = sys.stdin.readline().lower()[0] == 'y'

def distance(lon1, lat1, lon2, lat2):
    return math.sqrt((lon1 - lon2)*(lon1 - lon2) + (lat1 - lat2)*(lat1 - lat2))

def calculateCrimeHistograms(data):
    # Neighbor Feature Set
    crimeDistances = [1000, 500, 100, 10, 5, 1]
    crimeHistograms = []

    distMatrix = np.zeros((len(data), len(data)))

    locationX = [x for x in data["X"]]
    locationY = [y for y in data["Y"]]

    categories = np.array([classes.index(x) for x in data["Category"]])

    for i in range(0, len(data)):
        #if i % 100 == 0:
        print i, len(data)

        crimeHistogram = [0] * len(crimeDistances)

        for j in range(0, len(data)):
            dist = distMatrix[i][j]
            if dist is None:
                dist = distMatrix[j][i] = distMatrix[i][j] = distance(locationX[i], locationY[i], locationX[j], locationY[j]) * 1000

            for index, maxDist in enumerate(crimeDistances):
                crimeHistogram[index] = [0]*len(categories)
                if dist < maxDist:
                    crimeHistogram[index][categories[j]] += 1
                else:
                    break

        # normalize histogram
        for histogram in crimeHistogram:
            sum = float(np.sum(histogram)) # sum should not be 0 because there is always i itself in the "near location"
            for k in range(0, len(histogram)):
                histogram[k] /= sum
            print histogram

    return crimeHistograms


def convertData(data):
    if "Category" in data:
        categories = np.array([classes.index(x) for x in data["Category"]])
    else:
        categories = np.array([0] * len(data["Dates"]))

    parsedDates = [datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timetuple() for s in data["Dates"]]

    dates = [time.mktime(s) for s in parsedDates]
    year = [float(s.tm_year) for s in parsedDates]
    month = [float(s.tm_mon) for s in parsedDates]
    dayOfYear = [float(s.tm_yday) for s in parsedDates]
    dayOfWeek = [float(s.tm_wday) for s in parsedDates]
    dayOfMonth = [float(s.tm_mday) for s in parsedDates]
    hour = [float(s.tm_hour) for s in parsedDates]
    min = [float(s.tm_min) for s in parsedDates]

    locationX = [float(x*10000000000000) for x in data["X"]]
    locationY = [float(y*10000000000000) for y in data["Y"]]
    districts = list(set(data["PdDistrict"]))

    x_train = []
    for i in range(0, len(dates)):
        x_train.append([dates[i], year[i], month[i], dayOfYear[i], dayOfWeek[i], dayOfMonth[i], hour[i], min[i], locationX[i], locationY[i], float(districts.index(data["PdDistrict"][i]))])

    x_train = np.array(x_train)

    return x_train, categories

def fetchData(data, name):
    if prepareNewData:
        x_train, categories = convertData(data)

        pickle.dump(x_train, open(name + '_x_train.pkl', 'wb'))
        pickle.dump(categories, open(name + '_categories.pkl', 'wb'))
    else:
        x_train = pickle.load(open(name + '_x_train.pkl', 'rb'))
        categories = pickle.load(open(name + '_categories.pkl', 'rb'))


    return x_train, categories

def findTopCrimeClasses():
    crimesBucket = {}
    crimeCount = 0

    for cat in trainData["Category"]:
        crimesBucket[cat] = 1.0 + (crimesBucket[cat] if cat in crimesBucket else 0.0)
        crimeCount += 1

    for key in crimesBucket:
        crimesBucket[key] /= crimeCount

    sorted_cats = sorted(crimesBucket.items(), key=operator.itemgetter(1))

    topCrimes = []
    percentage = 0
    for cat in reversed(sorted_cats):
        if percentage > 0.5:
            break
        percentage += cat[1]
        topCrimes.append(classes.index(cat[0]))

    notTopCrimes = [classes.index(cat) for cat in classes if classes.index(cat) not in topCrimes]

    return topCrimes, notTopCrimes

trainXData, trainCategories = fetchData(trainData, "train")
testXData, testCategories = fetchData(testData, "test")

topCrimes, notTopCrimes = findTopCrimeClasses()

def filterByCategory(x_train, categories, possibleCategory):
    newXTrain = []
    newCategories = []

    for i in range(0, len(x_train)):
        if categories[i] in possibleCategory:
            newXTrain.append(x_train[i])
            newCategories.append(categories[i])

    return np.array(newXTrain), np.array(newCategories)

def learn():
    if calcCrimeHistograms:
        print("Calculate Crime Histograms")
        crimeHistograms = calculateCrimeHistograms(trainData)
        pickle.dump(crimeHistograms, open('crime_histogram.pkl', 'wb'))

    if trainNearestNeighbor:
        print("Train Nearest Neighbor")
        nearestneighbor_classificator = neighbors.KNeighborsClassifier(10, 'distance')
        newTrainXData = [[[d[8], d[9]]] for d in trainXData]
        nearestneighbor_classificator.fit(newTrainXData, trainCategories)

        pickle.dump(nearestneighbor_classificator, open('nearestneighbor_top.pkl', 'wb'))


    if trainCategoryClassifier:
        print("Training category classifier")
        decider_classificator = Pipeline([
            ('min/max scaler', MinMaxScaler(feature_range=(-1.0, 1.0))),
            ('neural network', Classifier(
                    layers=[
                        Layer("Sigmoid", units=30),
                        Layer("Softmax")
                    ],
                    learning_rate=0.001,
                    n_iter=80
            ))
        ])
        expected = np.array([int(x in topCrimes) for x in trainCategories])
        decider_classificator.fit(trainXData, expected)
        pickle.dump(decider_classificator, open('category_decider.pkl', 'wb'))

    if trainTopCrimeClassifier:
        print("Training top crime classifier")
        topCrime_classificator = Pipeline([
            ('min/max scaler', MinMaxScaler(feature_range=(-1.0, 1.0))),
            ('neural network', Classifier(
                    layers=[
                        Layer("Sigmoid", units=30),
                        Layer("Softmax")
                    ],
                    learning_rate=0.001,
                    n_iter=80
            ))
        ])
        newXTrain, newCategories = filterByCategory(x_train=trainXData, categories=trainCategories, possibleCategory=topCrimes)
        topCrime_classificator.fit(newXTrain, newCategories)
        pickle.dump(topCrime_classificator, open('topcrime_decider.pkl', 'wb'))

    if trainNotTopCrimeClassifier:
        print("Training not top crime classifier")
        notTopCrime_classificator = Pipeline([
            ('min/max scaler', MinMaxScaler(feature_range=(-1.0, 1.0))),
            ('neural network', Classifier(
                    layers=[
                        Layer("Sigmoid", units=30),
                        Layer("Softmax")
                    ],
                    learning_rate=0.001,
                    n_iter=80
            ))
        ])
        newXTrain, newCategories = filterByCategory(x_train=trainXData, categories=trainCategories, possibleCategory=notTopCrimes)
        notTopCrime_classificator.fit(newXTrain, newCategories)
        pickle.dump(notTopCrime_classificator, open('nottopcrime_decider.pkl', 'wb'))

def predict():
    print("Start predicting!")

    topCrimeDecider = pickle.load(open('category_decider.pkl', 'rb'))
    nearestNeighbor_classificator = pickle.load(open('nearestneighbor_top.pkl', 'rb'))
    topCrime_classificator = pickle.load(open('topcrime_decider.pkl', 'rb'))
    notTopCrime_classificator = pickle.load(open('nottopcrime_decider.pkl', 'rb'))

    result = []

    for i in range(0, len(testCategories)):
        if i % 1000 == 0:
            print i, len(testCategories)
        row = [testXData[i]]

        dist = nearestNeighbor_classificator.kneighbors(row, 1)
        # Average 17084926.4395

        newRow = [0]*len(classes)
        if dist[0][0] < 5000:
            # use nearest neighbor
            newRow[nearestNeighbor_classificator.predict(row)[0]] = 1
        else:
            # use neural network
            if topCrimeDecider.predict(row)[0]:
                # predict using top 10
                tmpResult = topCrime_classificator.predict_proba(row)[0]
                currentCrimes = topCrimes
            else:
                tmpResult = notTopCrime_classificator.predict_proba(row)[0]
                currentCrimes = notTopCrimes

            for index, res in enumerate(tmpResult):
                newRow[currentCrimes[index]] = res

        result.append(newRow)

    print "Dump result"
    result = np.array(result)

    indices = np.array([[x] for x in range(0, result.shape[0])])
    result = np.append(indices, result, axis=1)

    with open('new_result.csv', 'wb') as f:
        f.write('Id,ARSON,ASSAULT,BAD CHECKS,BRIBERY,BURGLARY,DISORDERLY CONDUCT,DRIVING UNDER THE INFLUENCE,DRUG/NARCOTIC,DRUNKENNESS,EMBEZZLEMENT,EXTORTION,FAMILY OFFENSES,FORGERY/COUNTERFEITING,FRAUD,GAMBLING,KIDNAPPING,LARCENY/THEFT,LIQUOR LAWS,LOITERING,MISSING PERSON,NON-CRIMINAL,OTHER OFFENSES,PORNOGRAPHY/OBSCENE MAT,PROSTITUTION,RECOVERED VEHICLE,ROBBERY,RUNAWAY,SECONDARY CODES,SEX OFFENSES FORCIBLE,SEX OFFENSES NON FORCIBLE,STOLEN PROPERTY,SUICIDE,SUSPICIOUS OCC,TREA,TRESPASS,VANDALISM,VEHICLE THEFT,WARRANTS,WEAPON LAWS\n')
        np.savetxt(f, result, delimiter=",", fmt='%i,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f')


learn()
predict()
