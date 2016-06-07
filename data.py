import random
import numpy as np

def ml_1m():
    # load data
    data = []
    for line in open("ml-1m/ratings.dat"):
        t = line.strip().split("::")
        userId = int(t[0]) - 1
        itemId = int(t[1]) - 1
        rating = float(t[2])
        data.append([userId, itemId, rating])
    
    # shuffle data
    data = np.array(data)
    np.random.seed(123456789)
    np.random.shuffle(data)

    # split data
    ratingCount = data.shape[0]
    splitPoint = int(ratingCount * 0.9)
    trainSet = data[:splitPoint,:]
    testSet = data[splitPoint:,:]

    # return data
    userCount = 6040
    itemCount = 3952
    return userCount, itemCount, trainSet, testSet
