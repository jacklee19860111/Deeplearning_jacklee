import numpy as np


a = np.empty((3,4))
print("a.shape:",a.shape)
print("np.empty((3,4)):", a)
print("np.linspace:")
b = np.linspace(1,10,20)
print(b)
a = np.array([10,20,30,40])
a = np.array([[1,1],
              [0,1]])
b = np.arange(4).reshape((2,2))
c = a * b
print("c:",c)
print("a:",a)
print("b:",b)
print(b==3)
a = np.random.random((2,4))

print("a:",a)
print("np.sum(a):",np.sum(a,axis=1))
print("np.min(a)",np.min(a,axis=0))
print("np.max(a):",np.max(a,axis=1))

A = np.arange(14,2,-1).reshape((3,4))

print(A)
print(np.sum(A))
print(np.argmin(A))
print(np.argmax(A))
print(np.mean(A))
print(np.average(A))
print(np.cumsum(A))
print(np.diff(A))
print(np.nonzero(A))
print(np.transpose(A))
print(np.clip(A,4,10))
print(np.mean(A,axis=0))
print(np.mean(A,axis=1))
