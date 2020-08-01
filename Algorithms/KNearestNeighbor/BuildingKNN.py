import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import warnings
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
from matplotlib import style

style.use('ggplot')

#sample data
data = {'k': [[2,3],[1,2],[3,1],[1,1]],
        'r':[[5,5],[6,7],[7,7],[7,8]],
         'b': [[2,7],[3,8],[1,8],[2,9]]}
d  = [3,5]
[[plt.scatter(ii[0],ii[1], s=100 , color =i) for ii in data[i]] for i in data]

def KNN(data, predict, k=3):

    distance = []
    for group in data:
        for i in data[group]:
            ecludean = np.linalg.norm(math.sqrt((predict[0]-i[0])**2 + (predict[1]- i[1])**2))
            distance.append((ecludean,group))

    review = list(sorted(distance)[:k])
    print(review)
    result = Counter(review).most_common(1)[0][0][1]
    return(result)

print(KNN(data, d))


plt.scatter(d[0],d[1], s =100, color= 'g')
plt.show()
