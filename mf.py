import tensorflow as tf
import numpy as np
import random
import sys

try:
    learnRate = float(sys.argv[1])
    batchSize = int(sys.argv[2])
except:
    print("learnRate batchSize")
    exit()

learnRate = 0.1
batchSize = 64
epochCount = 50
regular = 0.1
k = 10

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
print(trainSet.shape)
print(testSet.shape)

# matrix factorization
U = tf.Variable(tf.random_uniform([userCount, k], 0.0, 1.0))
V = tf.Variable(tf.random_uniform([itemCount, k], 0.0, 1.0))

u = tf.placeholder(tf.int32, [None, 1])
v = tf.placeholder(tf.int32, [None, 1])

userEmbedding = tf.reshape(tf.nn.embedding_lookup(U, u), [-1, k])
itemEmbedding = tf.reshape(tf.nn.embedding_lookup(V, v), [-1, k])

predict_r = tf.reduce_sum(tf.mul(userEmbedding, itemEmbedding), 1, keep_dims=True)

r = tf.placeholder(tf.float32, [None, 1])

loss = tf.reduce_mean(tf.square(tf.sub(predict_r, r)))
loss_r = tf.reduce_mean(tf.square(tf.sub(predict_r, r))) + regular * (tf.reduce_sum(tf.square(U)) + tf.reduce_sum(tf.square(V)))
optimizer = tf.train.GradientDescentOptimizer(learnRate)
trainStep = optimizer.minimize(loss)

rmse = tf.sqrt(loss)

#print(U.get_shape())
#print(V.get_shape())
#print(u.get_shape())
#print(v.get_shape())
#print(userEmbedding.get_shape())
#print(itemEmbedding.get_shape())
#print(predict_r.get_shape())
#print(r.get_shape())
#print(loss.get_shape())
#print(rmse.get_shape())

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
        #print(batch_u.shape)
        #print(batch_v.shape)
        #print(batch_r.shape)
        
        trainStep.run(feed_dict={u:batch_u, v:batch_v, r:batch_r})

    # predict
    test_u = testSet[:, 0:1]
    test_v = testSet[:, 1:2]
    test_r = testSet[:, 2:3]
    #print(test_u.shape)
    #print(test_v.shape)
    #print(test_r.shape)

    pre_r = predict_r.eval(feed_dict={u:test_u, v:test_v, r:test_r})    
    print(test_r[0][0], pre_r[0][0])

    result = rmse.eval(feed_dict={u:test_u, v:test_v, r:test_r})
    print("epoch %d/%d\trmse: %.4f"%(epoch+1, epochCount, result))


