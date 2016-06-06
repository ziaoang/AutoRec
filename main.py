import tensorflow as tf
import numpy as np
import random

random.seed(123456789)
np.random.seed(123456789)
tf.set_random_seed(123456789)

# load data
userCount = 6040
itemCount = 3952

trainX = {}
trainMask = {}
testX = {}
testMask = {}

lines = open("ml-1m/ratings.dat").readlines()
random.shuffle(lines)

for i in range(len(lines)):
    t = lines[i].strip().split("::")
    userId = int(t[0]) - 1
    itemId = int(t[1]) - 1
    rating = float(t[2])

    if i < 0.9 * len(lines):
        if itemId not in trainX:
            trainX[itemId] = [0] * userCount
            trainMask[itemId] = [0] * userCount
        trainX[itemId][userId] = rating
        trainMask[itemId][userId] = 1.0
    else:
        if itemId not in testX:
            testX[itemId] = [0] * userCount
            testMask[itemId] = [0] * userCount
        testX[itemId][userId] = rating
        testMask[itemId][userId] = 1.0

print(len(trainX))
print(len(testX))

# auto encoder
k = 500

x = tf.placeholder(tf.float32, [None, userCount])
mask = tf.placeholder(tf.float32, [None, userCount])

W1 = tf.Variable(tf.random_uniform([userCount, k], -1.0, 1.0))
b1 = tf.Variable(tf.random_uniform([k], -1.0, 1.0))
yMid = tf.nn.softmax(tf.matmul(x, W1) + b1)
W2 = tf.Variable(tf.random_uniform([k, userCount], -1.0, 1.0))
b2 = tf.Variable(tf.random_uniform([userCount], -1.0, 1.0))
y = tf.matmul(yMid, W2) + b2

# loss function
regularization = 0.001 # 0.001, 0.01, 0.1, 1, 100, 1000
learnRate = 0.01

loss = tf.reduce_sum(tf.square(tf.mul(tf.sub(x, y), mask))) + regularization / 2 * (tf.reduce_sum(tf.square(W1)) + tf.reduce_sum(tf.square(W2)))
optimizer = tf.train.GradientDescentOptimizer(learnRate)
trainStep = optimizer.minimize(loss)

# evaluation
x_ = tf.placeholder(tf.float32, [None, userCount])
mask_ = tf.placeholder(tf.float32, [None, userCount])
rmse = tf.sqrt(tf.div(tf.reduce_sum(tf.square(tf.mul(tf.sub(x_, y), mask_))), tf.reduce_sum(mask_)))

# training
epochCount = 50
batchSize = 16

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

for epoch in range(epochCount):

    print("epoch: %d/%d"%(epoch+1, epochCount))

    itemIdList = trainX.keys()
    random.shuffle(itemIdList)
    
    # train
    for batchId in range( len(itemIdList) / batchSize ):
        start = batchId * batchSize
        end = start + batchSize

        batchX = []
        batchMask = []
        for i in range(start, end):
            itemId = itemIdList[i]
            batchX.append(trainX[itemId])
            batchMask.append(trainMask[itemId])

        batchX = np.array(batchX)
        batchMask = np.array(batchMask)
        trainStep.run(feed_dict={x: batchX, mask: batchMask})

    # predict
    totalX = []
    totalMask = []
    totalX_ = []
    totalMask_ = []
    for itemId in itemIdList:
        if itemId in testX:
            totalX.append(trainX[itemId])
            totalMask.append(trainMask[itemId])
            totalX_.append(testX[itemId])
            totalMask_.append(testMask[itemId])

    totalX = np.array(totalX)
    totalMask = np.array(totalMask)
    totalX_ = np.array(totalX_)
    totalMask_ = np.array(totalMask_)

    result = rmse.eval(feed_dict={x: totalX, x_: totalX_, mask_: totalMask_})
    print("rmse: %.4f"%result)
