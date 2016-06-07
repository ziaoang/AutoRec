import tensorflow as tf
import numpy as np
import random
import sys

try:
    learnRate = float(sys.argv[1])
    batchSize = int(sys.argv[2])
    regular   = float(sys.argv[3])
except:
    print("learnRate batchSize regular")
    exit()

#learnRate = 0.1
#batchSize = 64
#regular = 0.1

epochCount = 100
k = 10
globalMean = 3.5811

random.seed(123456789)
np.random.seed(123456789)
tf.set_random_seed(123456789)

# load data
userCount = 6040
itemCount = 3952

trainSet = []
testSet = []
lines = open("ml-1m/ratings.dat").readlines()
random.shuffle(lines)
for i in range(len(lines)):
    t = lines[i].strip().split("::")
    userId = int(t[0]) - 1
    itemId = int(t[1]) - 1
    rating = float(t[2])

    if i < 0.9 * len(lines):
        trainSet.append([userId, itemId, rating])
    else:
        testSet.append([userId, itemId, rating])

trainSet = np.array(trainSet)
testSet = np.array(testSet)

# matrix factorization
u = tf.placeholder(tf.int32,   [None, 1])
v = tf.placeholder(tf.int32,   [None, 1])
r = tf.placeholder(tf.float32, [None, 1])

userFactorEmbeddingMatrix = tf.Variable(tf.random_uniform([userCount, k], 0.0, 1.0))
itemFactorEmbeddingMatrix = tf.Variable(tf.random_uniform([itemCount, k], 0.0, 1.0))
userBiasEmbeddingMatrix   = tf.Variable(tf.random_uniform([userCount, 1], 0.0, 1.0))
itemBiasEmbeddingMatrix   = tf.Variable(tf.random_uniform([itemCount, 1], 0.0, 1.0))

userFactorEmbedding = tf.reshape(tf.nn.embedding_lookup(userFactorEmbeddingMatrix, u), [-1, k])
itemFactorEmbedding = tf.reshape(tf.nn.embedding_lookup(itemFactorEmbeddingMatrix, v), [-1, k])
userBiasEmbedding   = tf.reshape(tf.nn.embedding_lookup(userBiasEmbeddingMatrix, u),   [-1, 1])
itemBiasEmbedding   = tf.reshape(tf.nn.embedding_lookup(itemBiasEmbeddingMatrix, v),   [-1, 1])

y = tf.reduce_sum(tf.mul(userFactorEmbedding, itemFactorEmbedding), 1, keep_dims=True)
userRegular = tf.reduce_sum(tf.square(userFactorEmbedding), 1, keep_dims=True)
itemRegular = tf.reduce_sum(tf.square(itemFactorEmbedding), 1, keep_dims=True)

loss = tf.reduce_mean( tf.square(r - y) + regular * (userRegular + itemRegular) )
optimizer = tf.train.GradientDescentOptimizer(learnRate)
trainStep = optimizer.minimize(loss)

rmse = tf.sqrt(tf.reduce_mean(tf.square(r-y)))

# training
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

for epoch in range(epochCount):
    np.random.shuffle(trainSet)
 
    # train
    for batchId in range( trainSet.shape[0] / batchSize ):
        start = batchId * batchSize
        end = start + batchSize

        batch_u = trainSet[start:end, 0:1]
        batch_v = trainSet[start:end, 1:2]
        batch_r = trainSet[start:end, 2:3]
        
        trainStep.run(feed_dict={u:batch_u, v:batch_v, r:batch_r})

    # predict
    test_u = testSet[:, 0:1]
    test_v = testSet[:, 1:2]
    test_r = testSet[:, 2:3]

    #predict_r = y.eval(feed_dict={u:test_u, v:test_v, r:test_r})    
    #print(test_r[0][0], predict_r[0][0])

    result = rmse.eval(feed_dict={u:test_u, v:test_v, r:test_r})
    print("%d/%d\t%.4f"%(epoch+1, epochCount, result))





