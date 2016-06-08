import numpy as np
from collections import defaultdict
import math

# load data
import data
userCount, itemCount, trainSet, testSet = data.ml_1m()
globalMean = trainSet[:,2:3].mean()

userRating = defaultdict(list)
itemRating = defaultdict(list)
for t in trainSet:
    userId = int(t[0])
    itemId = int(t[1])
    rating = float(t[2])
    userRating[userId].append(rating)
    itemRating[itemId].append(rating)

userMean = {}
for userId in userRating:
    userMean[userId] = np.array(userRating[userId]).mean()
itemMean = {}
for itemId in itemRating:
    itemMean[itemId] = np.array(itemRating[itemId]).mean()

cnt = 0
global_rmse = 0.0
user_rmse = 0.0
item_rmse = 0.0
user_item_rmse = 0.0
for t in testSet:
    userId = int(t[0])
    itemId = int(t[1])
    rating = float(t[2])

    cnt += 1
    
    global_rmse += (rating - globalMean) ** 2

    if userId in userMean:
        user_rmse += (rating - userMean[userId]) ** 2
    else:
        user_rmse += (rating - globalMean) ** 2

    if itemId in itemMean:
        item_rmse += (rating - itemMean[itemId]) ** 2
    else:
        item_rmse += (rating - globalMean) ** 2

    if userId in userMean:
        if itemId in itemMean:
            user_item_rmse += (rating - (userMean[userId] + itemMean[itemId])/2) ** 2
        else:
            user_item_rmse += (rating - (userMean[userId] + globalMean)/2) ** 2
    else:
        if itemId in itemMean:
            user_item_rmse += (rating - (globalMean + itemMean[itemId])/2) ** 2
        else:
            user_item_rmse += (rating - globalMean) ** 2

global_rmse = math.sqrt(global_rmse / cnt)
user_rmse = math.sqrt(user_rmse / cnt)
item_rmse = math.sqrt(item_rmse / cnt)
user_item_rmse = math.sqrt(user_item_rmse / cnt)
print("global mean: %.4f"%global_rmse)
print("user mean: %.4f"%user_rmse)
print("item mean: %.4f"%item_rmse)
print("user mean and item mean: %.4f"%user_item_rmse)



