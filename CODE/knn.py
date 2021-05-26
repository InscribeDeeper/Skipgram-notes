import numpy as np


def knn(vect, matrx, k):
    '''
    The algorithm receives a vector, a matrix and an integer k,
    and returns k indices of the matrix’s rows that are closest to the vector.
    Use the cosine similarity as a distance metric

    :return: the most similar vectors' indices of rows in matrx
    '''
    cos_s = [np.dot(vect, vector2) / (np.linalg.norm(vector2, ord=2) * (np.linalg.norm(vect, ord=2))) for vector2 in matrx]  # 默认取行
    res = np.argsort(cos_s)[::-1][0:k] # descending order and select k biggest vector's index
    return res


if __name__ == "__main__":
    vectr = np.random.rand(10)
    matrx = np.random.rand(5000, 10)

    ans = knn(vectr, matrx, 10)
    print(ans)

