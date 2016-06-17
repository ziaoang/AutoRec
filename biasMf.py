import tensorflow as tf
import numpy as np

tf.set_random_seed(123456789)
np.random.seed(123456789)

import sys
try:
    batchSize = int(sys.argv[1])
    learnRate = float(sys.argv[2])
    reLambda  = float(sys.argv[3])
except:
    print("batchSize learnRate reLambda")
    exit()

print("parameter info:")
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

print("dataset info:")
print("user count:\t%d"%(userCount))
print("item count:\t%d"%(itemCount))
print("train count:\t%d"%(trainSet.shape[0]))
print("test count:\t%d"%(testSet.shape[0]))
print("global mean:\t%.4f"%(globalMean))
print("="*20)

# matrix factorization
u = tf.placeholder(tf.int32,   [None, 1])
v = tf.placeholder(tf.int32,   [None, 1])
r = tf.placeholder(tf.float32, [None, 1])

U = tf.Variable(tf.random_uniform([userCount, k], -0.05, 0.05))
V = tf.Variable(tf.random_uniform([itemCount, k], -0.05, 0.05))
biasU = tf.Variable(tf.random_uniform([userCount, 1], -0.05, 0.05))
biasV = tf.Variable(tf.random_uniform([itemCount, 1], -0.05, 0.05))

uFactor = tf.reshape(tf.nn.embedding_lookup(U, u), [-1, k])
vFactor = tf.reshape(tf.nn.embedding_lookup(V, v), [-1, k])
uBias   = tf.reshape(tf.nn.embedding_lookup(biasU, u), [-1, 1])
vBias   = tf.reshape(tf.nn.embedding_lookup(biasV, v), [-1, 1])

y = tf.reduce_sum(tf.mul(uFactor, vFactor), 1, keep_dims=True) + uBias + vBias + globalMean

rmse = tf.sqrt(tf.reduce_mean(tf.square(r - y)))
mae  = tf.reduce_mean(tf.abs(r - y))

# loss function
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

uFactorRegular = tf.reduce_sum(tf.square(uFactor), 1, keep_dims=True)
vFactorRegular = tf.reduce_sum(tf.square(vFactor), 1, keep_dims=True)
uBiasRegular   = tf.square(uBias)
vBiasRegular   = tf.square(vBias)
loss = tf.reduce_mean(tf.square(r - y) + reLambda * (uFactorRegular + vFactorRegular + uBiasRegular + vBiasRegular))
trainStep = tf.train.GradientDescentOptimizer(learnRate).minimize(loss)

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

    rmse_score = rmse.eval(feed_dict={u:test_u, v:test_v, r:test_r})
    mae_score = mae.eval(feed_dict={u:test_u, v:test_v, r:test_r})
    print("%d/%d\t%.4f\t%.4f"%(epoch+1, epochCount, rmse_score, mae_score))



