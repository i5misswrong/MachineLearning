import math
import numpy as np
x_s=[1,2,3]
y_s=[1,2,3]

coo=[]

for i in x_s:
    for j in y_s:
        te=[i,j]
        coo.append(te)
coo.remove([2,2])

def parDer_X(x,y):
    x_p=-2*x*math.exp(-(x**2+y**2))
    return x_p
def parDer_Y(x,y):
    y_p=-2*y*math.exp(-(x**2+y**2))
    return y_p
def countAB(A,B):
    m=B[0]-A[0]
    n=B[1]-A[1]
    return [m,n]
def countABSAB(mn):
    return math.sqrt(mn[0]**2+mn[1]**2)
def countL(mn,AAB):
    L1=mn[0]/AAB
    L2=mn[1]/AAB
    return [L1,L2]

A=[2,2]
u=[]
for i in coo:
    AB_=countAB(A,i)
    absAB=countABSAB(AB_)
    L=countL(AB_,absAB)
    x_p=parDer_X(A[0],A[1])
    y_p=parDer_Y(A[0],A[1])
    u_l=x_p*L[0]+y_p*L[1]
    u.append(u_l)
    print(i,'***',u_l)
# print(u)