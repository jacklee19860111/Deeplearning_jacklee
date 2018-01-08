from sklearn.neural_network import BernoulliRBM
X = [[0,0],[1,1]]
y = [0,1]
clf = BernoulliRBM().fit(X,y)
print clf
