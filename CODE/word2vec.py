#!/usr/bin/env python

import numpy as np
import random

from utils.gradcheck import gradcheck_naive
from utils.utils import normalizeRows, softmax


def sigmoid(x):
    """
    Compute the sigmoid function for the input here.
    Arguments:
    x -- A scalar or numpy array.
    Return:
    s -- sigmoid(x)
    """

    ### YOUR CODE HERE
    s = 1. / (1. + np.exp(-x))
    ### END YOUR CODE

    return s


def naiveSoftmaxLossAndGradient(centerWordVec, outsideWordIdx, outsideVectors, dataset):
    """ Naive Softmax loss & gradient function for word2vec models

    Implement the naive softmax loss and gradients between a center word's
    embedding and an outside word's embedding. This will be the building block
    for our word2vec models.

    Arguments:
    centerWordVec -- numpy ndarray, center word's embedding
                    (v_c in the pdf handout)
    outsideWordIdx -- integer, the index of the outside word
                    (o of u_o in the pdf handout)
    outsideVectors -- outside vectors (rows of matrix) for all words in vocab
                      (U in the pdf handout)
    dataset -- needed for negative sampling, unused here.

    Return:
    loss -- naive softmax loss
    gradCenterVec -- the gradient with respect to the center word vector
                     (dJ / dv_c in the pdf handout)
    gradOutsideVecs -- the gradient with respect to all the outside word vectors
                    (dJ / dU)
    """

    ### YOUR CODE HERE
    ### This is the basic version

    # 在HW中, outsideVectors 是 U, 每一列就是一个 word 的vector
    # 在代码中, 没必要 centerWordVec = centerWordVec.reshape(-1,1) # 初始化为列形式 = hw公式表示的初始形式
    # 直接放在dot的右边, 计算时, 就以列向量的形式进行broadcasting
    # 在代码中, outsideVectors 每一 row就是一个 word 的vector
    # softmax 的input应该是一个行向量. 如果是列向量, 就无法计算了.
    y_pred = softmax(np.dot(outsideVectors, centerWordVec))  # input 为一个1-array, numpy默认为行向量
    # print("#"*30, "\n y_pred shape:", y_pred.shape,"\n",y_pred)

    ### Based on the provement (c i), the simplified CE is Negative Summation of (y_true*log(y_predict)),
    ### since y is one-hot vector, only the kth y is 1 and the other are 0.
    ### the loss should be -1*log(softmax(input));
    ### Therefore, we can just treat the outsideWordIdx of y_pred as y_hat
    loss = -np.log(y_pred[outsideWordIdx])  # 如果这里只有一个outsideword的话, 那么这样没问题. 如果不止一个, 则外面需要求和

    ### According to provement (c i), the gradient of loss related to center word is (y_pred - y_true)*U.T (如果都是vector,那么dot不影响)
    y_pred[outsideWordIdx] -= 1  # For grad_Vc of the outside words, the first part is y_pred - 1; 因为只有他们存在梯度,所有只有改变他们的必要
    gradCenterVec = np.dot(y_pred, outsideVectors)  # 1x5 * 5x3  = 1x3 ~ gradCenterVec

    ### According to (c ii), the gradient of loss related to the one outside word is (y_pred - y_true)*v_c.T
    # gradOutsideVecs = np.dot(y_pred.reshape(y_pred.shape[0],1), centerWordVec.reshape(1, centerWordVec.shape[0]) )
    gradOutsideVecs = np.outer(y_pred, centerWordVec)  # 5x1 * 1x3  = 5x3 ~ gradOutsideVecs
    # print("#"*30,"\n", gradOutsideVecs)
    ##########################################################

    ### END YOUR CODE

    return loss, gradCenterVec, gradOutsideVecs


def getNegativeSamples(outsideWordIdx, dataset, K):
    """ Samples K indexes which are not the outsideWordIdx """

    negSampleWordIndices = [None] * K
    for k in range(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == outsideWordIdx:
            newidx = dataset.sampleTokenIdx()
        negSampleWordIndices[k] = newidx
    return negSampleWordIndices


def negSamplingLossAndGradient(centerWordVec, outsideWordIdx, outsideVectors, dataset, K=10):
    """ Negative sampling loss function for word2vec models

    Implement the negative sampling loss and gradients for a centerWordVec
    and a outsideWordIdx word vector as a building block for word2vec
    models. K is the number of negative samples to take.

    Note: The same word may be negatively sampled multiple times. For
    example if an outside word is sampled twice, you shall have to
    double count the gradient with respect to this word. Thrice if
    it was sampled three times, and so forth.

    Arguments/Return Specifications: same as naiveSoftmaxLossAndGradient

    outsideVectors 包括了所有除了 center 以外的词
    print(centerWordVec.shape) # (3,)
    print(gradOutsideVecs.shape) # 5X3
    print(outsideVectors[negSampleWordIndices].shape) # 10 x 3  # 因为10个neg index
    outsideVectors[negSampleWordIndices].shape #  # 1x3 因为training sample 是 1对1的 outsideWordIdx一次只有一个??
    """
    # Negative sampling of words is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    negSampleWordIndices = getNegativeSamples(outsideWordIdx, dataset, K)
    indices = [outsideWordIdx] + negSampleWordIndices  # 这里可以保留所有的index, 需要计算的梯度 index 全在这里

    ### YOUR CODE HERE
    # 因为skipgram 每次只丢 一个 windows 中的pair. 所以这里的outsideWordIdx 只是一个词的idx, 然后附带一个中心词, 然后附带 多个 负样本词的 index
    # sigmoid就是计算这里有多相似, 如果越相似, 他们的sigmoid输出值越小 (因为取log后小于0, 所以loss function 需要添加负号)
    # 如果这个周围的词 与中心词越相似, 他们的 loss 越低
    # 然后这里只更新 一个正样本和 一个中心词的梯度. 但是计算的时候, 需要用到 所有 neg samples 的词的向量作为更新依据
    z_o = sigmoid(np.dot(outsideVectors[outsideWordIdx], centerWordVec)) 
    z_n = sigmoid(np.dot(-1 * outsideVectors[negSampleWordIndices], centerWordVec))

    ####### gradOutsideVecs #######
    # outsideVectors[negSampleWordIndices] 将重复采样部分也加入多次计算
    # print((sigmoid(-1 * np.dot(outsideVectors[negSampleWordIndices], centerWordVec)) - 1).shape)
    sum_neg_loss = np.log(z_n)  # 10x3  x  (3,) = (10,)
    loss = -1 * np.log(z_o) - np.sum(sum_neg_loss)

    ####### gradCenterVec #######
    # print(sigmoid(1 - np.dot(outsideVectors[negSampleWordIndices], centerWordVec)).shape)
    # (10,)  x  (10,3) = (3,) # 前者被看成行向量 (1,10), 所以最终结果为 (3,)
    sum_neg_grad = np.dot((z_n - 1), outsideVectors[negSampleWordIndices])  # 公式 1c iii 第8行.

    ####方案1 When outsideWordIdx is single number
    #     gradCenterVec = (z_o - 1) * outsideVectors[outsideWordIdx] - sum_neg_grad
    ####方案2 When outsideWordIdx is list of number
    gradCenterVec = np.dot((z_o - 1), outsideVectors[outsideWordIdx]) - sum_neg_grad  # 公式 1c iii 第11行.

    ####### gradOutsideVecs #######
    # Part1 - for outside words
    gradOutsideVecs = np.zeros(outsideVectors.shape)
    gradOutsideVecs[outsideWordIdx] = np.outer((z_o - 1), centerWordVec)  # 公式 1c iii 第18行.

    # Part2 - for negative sampling words
    ####方案1  # (10,) x (3,), 这里存在重叠, 这样可以只计算一次
    acc = {k: negSampleWordIndices.count(k) * (gradOutsideVecs[k] + -(sigmoid(-1 * np.dot(outsideVectors[k], centerWordVec)) - 1) * centerWordVec) for k in set(negSampleWordIndices)}
    for i in acc.keys():
        gradOutsideVecs[i] += acc[i]
    #  print(gradOutsideVecs,"\n")

    #     ####方案2  # 逐个更新慢, 因为要重复计算
    #     for k in negSampleWordIndices:
    #         gradOutsideVecs[k] += -(sigmoid(-1*np.dot(outsideVectors[k], centerWordVec))-1)*centerWordVec # 公式 1c iii 第25行.
    #         print(gradOutsideVecs,"\n")

    ##########################################################

    ### END YOUR CODE
    return loss, gradCenterVec, gradOutsideVecs


def skipgram(currentCenterWord, windowSize, outsideWords, word2Ind, centerWordVectors, outsideVectors, dataset, word2vecLossAndGradient=naiveSoftmaxLossAndGradient):
    """ Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    currentCenterWord -- a string of the current center word
    windowSize -- integer, context window size
    outsideWords -- list of no more than 2*windowSize strings, the outside words
    word2Ind -- a dictionary that maps words to their indices in
              the word vector list
    centerWordVectors -- center word vectors (as rows) for all words in vocab
                        (V in pdf handout)
    outsideVectors -- outside word vectors (as rows) for all words in vocab
                    (U in pdf handout)
    word2vecLossAndGradient -- the loss and gradient function for
                               a prediction vector given the outsideWordIdx
                               word vectors, could be one of the two
                               loss functions you implemented above.

    Return:
    loss -- the loss function value for the skip-gram model
            (J in the pdf handout)
    gradCenterVecs -- the gradient with respect to the center word vectors
            (dJ / dV in the pdf handout)
    gradOutsideVectors -- the gradient with respect to the outside word vectors
                        (dJ / dU in the pdf handout)
    """

    loss = 0.0
    gradCenterVecs = np.zeros(centerWordVectors.shape)
    gradOutsideVectors = np.zeros(outsideVectors.shape)

    ### YOUR CODE HERE

    center = word2Ind[currentCenterWord]  # word2Ind is a index table
    current = centerWordVectors[center]  # get the current v_c

    # For each window formed sample, accumulate Loss and calculate Grad for these words in the target index place
    for i in outsideWords:
        outside = word2Ind[i]
        Loss, GradCenterVecs, GradOutsideVectors = word2vecLossAndGradient(current, outside, outsideVectors, dataset)
        loss += Loss
        gradCenterVecs[center] += GradCenterVecs  # 对应位置保存 更新的grad
        gradOutsideVectors += GradOutsideVectors  ## 这里返回来的本身就是矩阵 所有outsidewords的vectors grad 矩阵
    # print("#"*30,"GradOutsideVectors.shape", GradOutsideVectors.shape)
    ##########################################################

    ### END YOUR CODE

    return loss, gradCenterVecs, gradOutsideVectors


#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################


def word2vec_sgd_wrapper(word2vecModel, word2Ind, wordVectors, dataset, windowSize, word2vecLossAndGradient=naiveSoftmaxLossAndGradient):
    # 这里把每一个 wordVectors 看成x, 但是需要用这样的一个warpper去包装, 告诉sgd内部应该怎么处理这个 matrix
    # 这里将整个matrix作为x输入 f(x), 得到导数grad, 然后在 sgd 中根据导数更新x
    # 这个wrapper的核心目的是 通过 vectors 得到更新的梯度 f(x) = f_grad => 用来更新x
    # warpper 是一个function f(x), 输入是 X matrix, 输出是skipgram算出来的 grad. 这是warpper的作用.
    # warpper需要调用写好的model, 因为model 部分才知道怎么计算 gradient, 然后在这里返回出来
    
    
    batchsize = 50
    loss = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    centerWordVectors = wordVectors[:int(N / 2), :]
    outsideVectors = wordVectors[int(N / 2):, :]
    for i in range(batchsize):
        windowSize1 = random.randint(1, windowSize)
        centerWord, context = dataset.getRandomContext(windowSize1)

        c, gin, gout = word2vecModel(centerWord, windowSize1, context, word2Ind, centerWordVectors, outsideVectors, dataset, word2vecLossAndGradient)
        loss += c / batchsize  # 计算cost = loss
        grad[:int(N / 2), :] += gin / batchsize  # W-input = 前一半累加in, 累加整个batch的samples算出来的 平均更新梯度
        grad[int(N / 2):, :] += gout / batchsize  # W-output = 后一半累加out, 累加整个batch的samples算出来的 平均更新梯度

    return loss, grad  # 这样来模拟一个function


def test_word2vec():
    """ Test the two word2vec implementations, before running on Stanford Sentiment Treebank """
    dataset = type('dummy', (), {})()

    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], \
            [tokens[random.randint(0,4)] for i in range(2*C)]

    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10, 3))
    dummy_tokens = dict([("a", 0), ("b", 1), ("c", 2), ("d", 3), ("e", 4)])

    print("==== Gradient check for skip-gram with naiveSoftmaxLossAndGradient ====")
    # 因为寻找的就是这个vec, 这个vec的维度可能很高也没事儿. 然后每次训练一个vec? 还是整个matrix?  
    # 这里的代码是直接输入的一个 vector. 
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5, naiveSoftmaxLossAndGradient), dummy_vectors, "naiveSoftmaxLossAndGradient Gradient")

    print("==== Gradient check for skip-gram with negSamplingLossAndGradient ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5, negSamplingLossAndGradient), dummy_vectors, "negSamplingLossAndGradient Gradient")  # 只需要这样代入loss function即可, 最重要的参数只有:loss, gradCenterVec, gradOutsideVecs

    print("\n=== Results ===")
    print("Skip-Gram with naiveSoftmaxLossAndGradient")

    print("Your Result:")
    print("Loss: {}\nGradient wrt Center Vectors (dJ/dV):\n {}\nGradient wrt Outside Vectors (dJ/dU):\n {}\n".format(*skipgram("c", 3, ["a", "b", "e", "d", "b", "c"], dummy_tokens, dummy_vectors[:5, :], dummy_vectors[5:, :], dataset)))

    print("Expected Result: Value should approximate these:")
    print("""Loss: 11.16610900153398
Gradient wrt Center Vectors (dJ/dV):
 [[ 0.          0.          0.        ]
 [ 0.          0.          0.        ]
 [-1.26947339 -1.36873189  2.45158957]
 [ 0.          0.          0.        ]
 [ 0.          0.          0.        ]]
Gradient wrt Outside Vectors (dJ/dU):
 [[-0.41045956  0.18834851  1.43272264]
 [ 0.38202831 -0.17530219 -1.33348241]
 [ 0.07009355 -0.03216399 -0.24466386]
 [ 0.09472154 -0.04346509 -0.33062865]
 [-0.13638384  0.06258276  0.47605228]]
    """)

    print("Skip-Gram with negSamplingLossAndGradient")
    print("Your Result:")
    print("Loss: {}\nGradient wrt Center Vectors (dJ/dV):\n {}\n Gradient wrt Outside Vectors (dJ/dU):\n {}\n".format(*skipgram("c", 1, ["a", "b"], dummy_tokens, dummy_vectors[:5, :], dummy_vectors[5:, :], dataset, negSamplingLossAndGradient)))
    print("Expected Result: Value should approximate these:")
    print("""Loss: 16.15119285363322
Gradient wrt Center Vectors (dJ/dV):
 [[ 0.          0.          0.        ]
 [ 0.          0.          0.        ]
 [-4.54650789 -1.85942252  0.76397441]
 [ 0.          0.          0.        ]
 [ 0.          0.          0.        ]]
 Gradient wrt Outside Vectors (dJ/dU):
 [[-0.69148188  0.31730185  2.41364029]
 [-0.22716495  0.10423969  0.79292674]
 [-0.45528438  0.20891737  1.58918512]
 [-0.31602611  0.14501561  1.10309954]
 [-0.80620296  0.36994417  2.81407799]]
    """)


if __name__ == "__main__":
    test_word2vec()
