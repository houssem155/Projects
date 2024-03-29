

from sklearn import datasets 
irisData = datasets.load_iris() 
print (irisData.data) 
print (irisData.target)

print(irisData.data.shape)
print(irisData.target.shape)

print(irisData.target_names)

print(irisData.feature_names)

from collections import Counter
c = Counter(irisData.target)
print (c)

for i in [0,1,2]:
  print("classe : {}, nb exemplaires : {}".format(i,len(irisData.target[irisData.target==i])))

print (irisData.data[31])

print (irisData.target[31])

print (irisData.target[:])

import matplotlib
import pylab as pl 
import itertools as it

def plot_2D(data, target, target_names): 
  colors = it.cycle('rgbcmykw') # cycle de couleurs 
  target_ids = range(len(target_names)) 
  pl.figure() 
  for i, c, label in zip(target_ids, colors, target_names): 
    pl.scatter(data[target == i, 2], data[target == i, 3], c=c, label=label) 
  pl.xlabel(irisData.feature_names[2])
  pl.ylabel(irisData.feature_names[3])
  pl.plot([2.5,2.5],[0,3],color='y',linestyle='--')
  pl.plot([0.75,7],[0.75,0.75],'b--')
  pl.legend() 
  pl.show()
  
plot_2D(irisData.data , irisData.target , irisData.target_names)

 from sklearn import naive_bayes 
 nb = naive_bayes.MultinomialNB(fit_prior=True)# un algo d'apprentissage >>> irisData = datasets.load_iris() 
 nb.fit(irisData.data[: -1], irisData.target[: -1]) 
 p31 = nb.predict([irisData.data[31]])
 print (p31) 
 plast = nb.predict([irisData.data[-1]]) 
 print (plast)
 p = nb.predict(irisData.data[:]) 
 print (p)

 from sklearn import naive_bayes 
 nb = naive_bayes.MultinomialNB(fit_prior=True) 
 nb.fit(irisData.data[:99], irisData.target[:99]) 
 nb.predict(irisData.data[100:149])

from sklearn import naive_bayes 
nb = naive_bayes.MultinomialNB(fit_prior=True) 
nb.fit(irisData.data[0:150], irisData.target[0:150]) 
P=nb.predict(irisData.data[0:150])
Y=irisData.target
ea = 0 
for i in range(len(irisData.data)): 
    if (P[i] != Y[i]):
        ea = ea + 1 
print(ea)
print (ea/len(irisData.data))

import numpy as np
np.count_nonzero(P-Y)/len(irisData.data)

import random
def split(S):
  n= len(S.data)
  X=random.sample(list(range(0,n)),int(9*n/10))
  traindatas1=[S.data[i].tolist() for i in X]
  traintargets1 = [S.target[i].tolist() for i in X]
  Y=[i for i in list(range(0,n)) if i not in X]
  testdatas2=[S.data[i].tolist()for i in Y]
  testtargets2= [S.target[i].tolist()for  i in Y]
  return [traindatas1 , traintargets1 , testdatas2 , testtargets2]

traindatas1 , traintargets1 , testdatas2 , testtargets2 = split(irisData)
print("la longeur de l'ensemble d'apprentissage est egal a :",len(traindatas1))
print("la longeur de l'ensemble de test est egal a :",len(testdatas2))

def test(S,clf):
  [traindatas1, traintargets1, testdatas2 ,testtargets2]=split(S)
  clf.fit(traindatas1,traintargets1)
  P=clf.predict(testdatas2)
  Y=testtargets2
  e=(P-Y !=0).sum()
  return(e/len(testdatas2))

test(irisData,nb)

print("go1")
for t in [10,50,100,200,500,1000]:
  sum=0
  for j in range (0,t):
    sum =sum +test(irisData,nb)
  moy= sum/t
  print("for t={},la moy={}".format(t,moy))

print("go2")
for t in [10,50,100,200,500,1000,1000]:
  for rep in range(0,20):

    sum=0
    for j in range (0,t):
      sum =sum +test(irisData,nb)
    moy= sum/t
    print("for t={},la moy={}".format(t,moy))

from sklearn.model_selection import train_test_split
def testsplit(S,clf,p):
  [datas1,datas2,targets1,targets2]=train_test_split(S.data , S.target ,test_size=p)
  clf.fit(datas1,targets1)
  return(1-(clf.score(datas2,targets2)))

testsplit(irisData,nb,0.2)

for t in [10,50,100,200,500,1000]:
  for rep in range(0,20):

    sum=0
    for j in range (0,t):
      sum =sum +test(irisData,nb)
    moy= sum/t
  print("for t={},la moy={}".format(t,moy))

from sklearn.model_selection import cross_val_score
scores=cross_val_score(nb,irisData.data,irisData.target,cv=12)
s=scores.sum()/len(scores)
print(scores)
print(1-s)

from sklearn.model_selection import cross_val_score
scores=cross_val_score(nb,irisData.data,irisData.target,cv=5)
s=scores.sum()/len(scores)
print(scores)
print(1-s)

from sklearn.model_selection import cross_val_score
scores=cross_val_score(nb,irisData.data,irisData.target,cv=3)
s=scores.sum()/len(scores)
print(scores)
print(1-s)

from sklearn.model_selection import cross_val_score
scores=cross_val_score(nb,irisData.data,irisData.target,cv=16)
s=scores.sum()/len(scores)
print(scores)
print(1-s)
