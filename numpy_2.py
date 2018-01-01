import numpy as np


A = np.arange(3,15).reshape(3,4)
print(A)
print(A[2])
print(A[2][1])
print(A[:,1])
print(A[1,1:2])
print(A.T)
for row in A:
    print(row)
for column in A.T:
    print(column)
print(A.flatten())
for item in A.flat:
    print(item)


A = np.array([1,1,1])
B = np.array([2,2,2])


C = np.vstack((A,B)) # vertical stack
D = np.hstack((A,B)) # horizontal stack
print(C)
print(A.shape,D.shape)

A = np.array([1,1,1])[:,np.newaxis]
B = np.array([2,2,2])[:,np.newaxis]

C = np.concatenate((A,B,B,A),axis=1)
print(C)
