# encoding=utf-8
# 

'''
Author: WeiY
Email: hey.weiyang@gmail.com

Date: 3/2/2020 12:32 AM
Desc:

'''


# def slower_negSamplingLossAndGradient(
#         centerWordVec,
#         outsideWordIdx,
#         outsideVectors,
#         dataset,
#         K=10
# ):
#     """ Negative sampling loss function for word2vec models
#
#     Implement the negative sampling loss and gradients for a centerWordVec
#     and a outsideWordIdx word vector as a building block for word2vec
#     models. K is the number of negative samples to take.
#
#     Note: The same word may be negatively sampled multiple times. For
#     example if an outside word is sampled twice, you shall have to
#     double count the gradient with respect to this word. Thrice if
#     it was sampled three times, and so forth.
#
#     Arguments/Return Specifications: same as naiveSoftmaxLossAndGradient
#
#     outsideVectors 包括了所有除了 center 以外的词
#     print(centerWordVec.shape) # (3,)
#     print(gradOutsideVecs.shape) # 5X3
#     print(outsideVectors[negSampleWordIndices].shape) # 10 x 3  # 因为10个neg index
#     outsideVectors[negSampleWordIndices].shape #  # 1x3 因为training sample 是 1对1的 outsideWordIdx一次只有一个??
#     """
#     # Negative sampling of words is done for you. Do not modify this if you
#     # wish to match the autograder and receive points!
#     negSampleWordIndices = getNegativeSamples(outsideWordIdx, dataset, K)
#     indices = [outsideWordIdx] + negSampleWordIndices  # 这里可以保留所有的index, 需要计算的梯度 index 全在这里
#
#     ### YOUR CODE HERE
#
#     ## 终极优化 - 一次性保存所有的 indices 的矩阵乘积结果
#
#     ####### gradOutsideVecs #######
#     # outsideVectors[negSampleWordIndices] 将重复采样部分也加入多次计算
#     # print((sigmoid(-1 * np.dot(outsideVectors[negSampleWordIndices], centerWordVec)) - 1).shape)
#     sum_neg_loss = np.log(
#         sigmoid(-1 * np.dot(outsideVectors[negSampleWordIndices], centerWordVec)))  # 10x3  x  (3,) = (10,)
#     loss = -1 * np.log(sigmoid(np.dot(outsideVectors[outsideWordIdx], centerWordVec))) - np.sum(sum_neg_loss)
#
#     ####### gradCenterVec #######
#     # print(sigmoid(1 - np.dot(outsideVectors[negSampleWordIndices], centerWordVec)).shape)
#     # (10,)  x  (10,3) = (3,) # 前者被看成行向量 (1,10), 所以最终结果为 (3,)
#     sum_neg_grad = np.dot((sigmoid(np.dot(-1 * outsideVectors[negSampleWordIndices], centerWordVec)) - 1),
#                           outsideVectors[negSampleWordIndices])  # 公式 1c iii 第8行.
#
#     ####方案1 When outsideWordIdx is single number
#     #     gradCenterVec = (sigmoid(np.dot(outsideVectors[outsideWordIdx], centerWordVec)) - 1) * outsideVectors[outsideWordIdx] - sum_neg_grad
#     ####方案2 When outsideWordIdx is list of number
#     gradCenterVec = np.dot((sigmoid(np.dot(outsideVectors[outsideWordIdx], centerWordVec)) - 1),
#                            outsideVectors[outsideWordIdx]) - sum_neg_grad  # 公式 1c iii 第11行.
#
#     ####### gradOutsideVecs #######
#     # Part1 - for outside words
#     gradOutsideVecs = np.zeros(outsideVectors.shape)
#     gradOutsideVecs[outsideWordIdx] = np.outer((sigmoid(np.dot(outsideVectors[outsideWordIdx], centerWordVec)) - 1),
#                                                centerWordVec)  # 公式 1c iii 第18行.
#     #     print(gradOutsideVecs,"\n")
#
#     # Part2 - for negative sampling words
#     ####方案1  # (10,) x (3,), 这里存在重叠, 这样可以只计算一次
#     acc = {k: negSampleWordIndices.count(k) * (
#                 gradOutsideVecs[k] + -(sigmoid(-1 * np.dot(outsideVectors[k], centerWordVec)) - 1) * centerWordVec) for
#            k in set(negSampleWordIndices)}
#     for i in acc.keys():
#         gradOutsideVecs[i] += acc[i]
#     #  print(gradOutsideVecs,"\n")
#
#     #     ####方案2  # 逐个更新慢, 因为要重复计算
#     #     for k in negSampleWordIndices:
#     #         gradOutsideVecs[k] += -(sigmoid(-1*np.dot(outsideVectors[k], centerWordVec))-1)*centerWordVec # 公式 1c iii 第25行.
#     #         print(gradOutsideVecs,"\n")
#
#     ##########################################################
#
#     ### END YOUR CODE
#     return loss, gradCenterVec, gradOutsideVecs