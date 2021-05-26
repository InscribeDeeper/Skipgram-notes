#!/usr/bin/env python


import random
import numpy as np
from utils.treebank import StanfordSentiment
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import time

from knn import knn
from word2vec import *
from sgd import *

# Check Python Version # 必须是3.5以上的版本
import sys
assert sys.version_info[0] == 3
assert sys.version_info[1] >= 5

# Reset the random seed to make sure that everyone gets the same results
random.seed(314)
dataset = StanfordSentiment()
tokens = dataset.tokens()
nWords = len(tokens)

# We are going to train 10-dimensional vectors for this assignment
###### 如果要增加 distribution representation 的准确性, 可以把这个调高, 可以调成200. 但是速度会慢很多, 需要GPU
dimVectors = 10

# Context size
###### 决定了Skip-gram的windows大小, training samples 的数量
C = 5

# Reset the random seed to make sure that everyone gets the same results
random.seed(31415)
np.random.seed(9265)



startTime=time.time()

#### WV Initialization: including centerWordVectors-W and outsideVectors-W, which are vstack
    # input-W initialized with uniform between -0.5~0.5/dimV.
    # WV dimension = 19539 x dimVectors (embedding 维度)
    # Different initialization may have no big influence
wordVectors = np.concatenate(((np.random.rand(nWords, dimVectors) - 0.5) / dimVectors
                              , np.zeros((nWords, dimVectors))), axis=0) ## (39078, dimVectors)


#### Fitting # 这里的方法可以切换成别的;
# negative sampling 已经算快了, 可以预处理部分再优化
# 这里输入的 x0 是一个 超高纬度的矩阵, 这个wrapper能够这样吗?
wordVectors = sgd(f=lambda vec: word2vec_sgd_wrapper(skipgram, tokens, vec, dataset, C, negSamplingLossAndGradient)
                  ,x0=wordVectors, step=0.3, iterations=40000, postprocessing=None, useSaved=True, PRINT_EVERY=20)

### 通过lambda表达式定义函数, vec 为自变量x, 后面样本点构成的loss function的参数空间. 只不过这个x是一个 参数矩阵. 维度: (nWords*2, dimVectors)
# lambda vec: word2vec_sgd_wrapper(skipgram, tokens, vec, dataset, C, negSamplingLossAndGradient)
# Step 可以调小一点 -> 准确性, 收敛
# 负采样数量可以小一点 -> 提速


# Note that normalization is not called here. This is not a bug,
# normalizing during training loses the notion of length.

print("sanity check: cost at convergence should be around or below 10")
print("training took %d seconds" % (time.time() - startTime))

# concatenate the input and output word vectors

########################  Read from saved params
# params_file = "saved_params_40000.npy"
# # state_file = "saved_state_10000.pickle"
# # params = np.load(params_file) # W 参数 (nWords*2, Vectdim)
# # # with open(state_file, "rb") as f:
# # #     state = pickle.load(f)
# wordVectors = params
# print(wordVectors.shape)

########################



wordVectors = np.concatenate((wordVectors[:nWords,:], wordVectors[nWords:,:]),axis=0)

visualizeWords = [
    "great", "cool", "brilliant", "wonderful", "well", "amazing",
    "worth", "sweet", "enjoyable", "boring", "bad", "dumb",
    "annoying", "female", "male", "queen", "king", "man", "woman", "rain", "snow",
    "hail", "coffee", "tea"]



######################## 数据降维 可视化
visualizeIdx = [tokens[word] for word in visualizeWords] # 提取index
visualizeVecs = wordVectors[visualizeIdx, :] # 可视化目标的 Word Vectors
# covariance = np.cov(visualizeVecs,rowvar=False) # 协方差矩阵

temp = (visualizeVecs - np.mean(visualizeVecs, axis=0))
covariance = 1.0 / len(visualizeIdx) * temp.T.dot(temp)
# np.max(np.cov(visualizeVecs, rowvar=False) - covariance) # 与直接调用函数的差值小于10e-3

# 这里是对协方差 方阵 求SVD, 协方差矩阵的SVD的左矩阵U? 意义是啥?
# 答: 因为对称, 所以分解出来的U 和 V.T 是一样的, test: np.max(U - V.T) <10e-15
U,S,V = np.linalg.svd(covariance)
coord = temp.dot(U[:,0:2]) # 只取最大的2个奇异值作为类似于PCA的投影维度上



# U,S,V=np.linalg.svd(visualizeVecs,full_matrices=True,hermitian=False)
# # S本来应该是方阵, 但是np简化输出为一个array, 且为0时, 自动舍弃.hermitian 矩阵分解提高计算效率, 只对方阵有效
# coord = temp.dot(V[:,0:2]) # 只取最大的2个奇异值作为类似于PCA的投影维度上


for i in range(len(visualizeWords)):
    plt.text(coord[i,0], coord[i,1], visualizeWords[i], bbox=dict(facecolor='green', alpha=0.1))

plt.xlim((np.min(coord[:,0]), np.max(coord[:,0])))
plt.ylim((np.min(coord[:,1]), np.max(coord[:,1])))

plt.savefig('word_vectors'+str(0)+'.png')


######################## KNN
mtx = wordVectors[0:len(tokens),:]
# mtx = wordVectors[len(tokens):,:] # 对于每一个center word外, outside word的W, 可以用来generation;
# 但可以看到有很多stopwords, 需要去优化 pre-processing

idx2word = {v : k for k, v in tokens.items()}
for i, v in enumerate(visualizeVecs):
    res = knn(v, mtx, 6) # 最相似的7个词
    print(visualizeWords[i].ljust(11),end=": ") # print(*objects, sep=' ', end='\n', file=sys.stdout, flush=False)
    for j in res[1::]:  # 不包括自己
        print(idx2word[j].ljust(16), end="|")
    print("")




################################################
###### 3D plot
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# matplotlib.use( 'tkagg' )
# fig = plt.figure()
# ax = fig.gca(projection='3d')
#
# for i in range(len(visualizeWords)):
#     ax.text(coord[i,0], coord[i,1],coord[i,2], visualizeWords[i], bbox=dict(facecolor='green', alpha=0.1))
#
# xs=coord[:,0]
# ys=coord[:,1]
# zs=coord[:,2]
# ax.scatter(xs, ys, zs)
#
# plt.show()



