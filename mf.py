import tensorflow as tf
import numpy as np

tf.set_random_seed(123456789)
np.random.seed(123456789)

import sys
try:
    useBias   = int(sys.argv[1])
    batchSize = int(sys.argv[2])
    learnRate = float(sys.argv[3])
    reLambda  = float(sys.argv[4])
except:
    print("useBias(0|1) batchSize learnRate reLambda")
    exit()

#useBias   = 1   # 1 for use and 0 for not use
#batchSize = 64
#learnRate = 0.1
#reLambda  = 0.1

# log
print("parameter list:")
print("use bias:\t%d"%useBias)
print("batch size:\t%d"%batchSize)
print("learn rate:\t%f"%learnRate)
print("regular lambda:\t%f"%reLambda)
print("="*20)

# hyper parameter
k = 10
epochCount = 100

# load data
import data
userCount, itemCount, trainSet, testSet = data.ml_1m()
globalMean = trainSet[:,2:3].mean()

# matrix factorization
u = tf.placeholder(tf.int32,   [None, 1])
v = tf.placeholder(tf.int32,   [None, 1])
r = tf.placeholder(tf.float32, [None, 1])

U     = tf.Variable(tf.random_uniform([userCount, k], 0.0, 1.0))
V     = tf.Variable(tf.random_uniform([itemCount, k], 0.0, 1.0))
biasU = tf.Variable(tf.random_uniform([userCount, 1], 0.0, 1.0))
biasV = tf.Variable(tf.random_uniform([itemCount, 1], 0.0, 1.0))

uFactor = tf.reshape(tf.nn.embedding_lookup(U, u), [-1, k])
vFactor = tf.reshape(tf.nn.embedding_lookup(V, v), [-1, k])
uBias   = tf.reshape(tf.nn.embedding_lookup(biasU, u), [-1, 1])
vBias   = tf.reshape(tf.nn.embedding_lookup(biasV, v), [-1, 1])

uFactorRegular = tf.reduce_sum(tf.square(uFactor), 1, keep_dims=True)
vFactorRegular = tf.reduce_sum(tf.square(vFactor), 1, keep_dims=True)
uBiasRegular   = tf.square(uBias)
vBiasRegular   = tf.square(vBias)

interAction = tf.reduce_sum(tf.mul(uFactor, vFactor), 1, keep_dims=True)
y = interAction + useBias * (globalMean + uBias + vBias)

# loss function
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

loss = tf.reduce_mean(tf.square(r - y) + reLambda * (uFactorRegular + vFactorRegular + useBias * (uBiasRegular + vBiasRegular)))
trainStep = tf.train.GradientDescentOptimizer(learnRate).minimize(loss)

rmse = tf.sqrt(tf.reduce_mean(tf.square(r - y)))
mae  = tf.reduce_mean(tf.abs(r - y))

# iterator
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

    # predict_r = y.eval(feed_dict={u:test_u, v:test_v, r:test_r})
    # print(test_r[0][0], predict_r[0][0])

    result = rmse.eval(feed_dict={u:test_u, v:test_v, r:test_r})
    print("%d/%d\t%.4f"%(epoch+1, epochCount, result))




