import random
import numpy as np

def ml_1m():
    # id to index
    userIdToUserIndex = {}
    basicUserIndex = 0
    itemIdToItemIndex = {}
    basicItemIndex = 0
    
    # load data
    data = []
    for line in open("ml-1m/ratings.dat"):
        userId, itemId, rating, timestamp = line.strip().split("::")
        data.append([userId, itemId, rating])
    
    # shuffle data
    random.seed(123456789)
    random.shuffle(data)

    # split data
    ratingCount = len(data)
    
    # train data
    trainSet = []
    for i in range(int(ratingCount * 0.9)):
        userId, itemId, rating = data[i]
        if userId not in userIdToUserIndex:
            userIdToUserIndex[userId] = basicUserIndex
            basicUserIndex += 1
        if itemId not in itemIdToItemIndex:
            itemIdToItemIndex[itemId] = basicItemIndex
            basicItemIndex += 1
        userIndex = userIdToUserIndex[userId]
        itemIndex = itemIdToItemIndex[itemId]
        trainSet.append([userIndex, itemIndex, float(rating)])
    
    # test data
    testSet = []
    for i in range(int(ratingCount * 0.9), ratingCount):
        userId, itemId, rating = data[i]
        if userId in userIdToUserIndex and itemId in itemIdToItemIndex:
            userIndex = userIdToUserIndex[userId]
            itemIndex = itemIdToItemIndex[itemId]
            testSet.append([userIndex, itemIndex, float(rating)])
    
    # return data
    return len(userIdToUserIndex), len(itemIdToItemIndex), np.array(trainSet), np.array(testSet)

