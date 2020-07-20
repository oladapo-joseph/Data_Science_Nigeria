import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
from collections import Counter
import math
import random


style.use('fivethirtyeight')

"""
    This script is a continuation of previously built KNeighbor Classifier
    with the introduction of a larger dataset and also a test of the accuracy
    of the Classifier
    Two methods to fetch the data

"""
#getting the data Method 1
def fetch_data():
    data = []
    dict_data = {'a': [],'b': []}
    n,m = 0,0
    with open ('datasets/breast-cancer-wisconsin.data','r') as ff:
        headers = ff.readline().split(',')
        for i in ff:
            data.append(i.split(','))

        for idea in data:
            if '?' not in idea:
                if idea[-1] == '2\n':
                    dict_data['a'].append(list(idea[1:-1]))
                if idea[-1] == '4\n':
                    dict_data['b'].append(list(idea[1:-1]))
#returns the data in a dictionary
    return(dict_data)




def get_data():
    df = pd.read_csv('datasets/breast-cancer-wisconsin.data')
    df.replace('?', -99999, inplace=True)
    df.drop(['id_no'], 1, inplace=True)

    full_data = df.astype(int).values.tolist()
    random.shuffle(full_data)

    test_size = 0.2
    train_data = full_data[:-int(test_size*len(full_data))]
    test_data = full_data[-int(test_size*len(full_data)):]

    train_set = {2:[], 4:[]}
    test_set = {2:[], 4:[]}

    for val in train_data:
        train_set[val[-1]].append(val[:-1])

    for val in test_data:
        test_set[val[-1]].append(val[:-1])
#returns the train set and test set
    return(train_set, test_set)


def KNN(data, predict, k=3):
        '''
            This function utilises the basic property of the KNeighborsClassifier
            which is the shortest ecludean distance between the test data and the
            already classified train data to classify the test data.
        Args

            data : dataset containing the train data ,datatype "dictionary"

            predict: the test data we want to classify datatype 'One
            dimensional array'

            k : number of nearest neighbors
        '''
        distance = []
        for group in data:
            for i in data[group]:
                mini = [(predict[a]-int(i[a]))**2 for a in range(len(i))]
                ecludean = np.linalg.norm(math.sqrt(sum(mini)))
                distance.append((ecludean,group))

                review = [ i for i in sorted(distance)[:k]]

                result = Counter(review).most_common(1)[0][0][1]
                return(result)



def testrun():
    '''
        this method works like the clssifier.score(feature_test,label_test)
        use to test the accuracy of the algorithm

    '''
    train, test = get_data()
    total = 0
    correct = 0
    for i in test:
        for x in test[i]:
            predict = KNN(train,x ,k =15)
            if predict == i:
                correct+=1
            total+=1
    accuracy = correct/total
    return (accuracy)

# testing the built algorithms

breastdata = fetch_data()
prediction = KNN(breastdata, [8,2,1,5,5,7,6,7,4],k=7)
print(prediction)

print(testrun())
