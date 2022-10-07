
from scipy.stats import norm
import numpy as np
import math

def findThre(meanCorrect, stdWrong, meanWrong):
    maxprob=-1
    bestThre = 0.5
    for i in range(1,20):
        thre = i*0.1
        p = probFunc(thre, meanCorrect, stdWrong, meanWrong)
        print(p)
        if p>maxprob:
            maxprob = p
            bestThre = thre
    print(maxprob)
    return bestThre
            



def probFunc(thre, meanCorrect, stdWrong, meanWrong):
    rate_param= 1/meanCorrect
    f = 0.6*(1 - math.exp(-1*rate_param*thre)) -0.4* ( norm.cdf(thre,loc=meanWrong,scale=stdWrong))
    #f = (1- norm.cdf(thre,loc=meanWrong,scale=stdWrong))/(math.exp(-1*rate_param*thre)) 
    return f

#best = findThre(0.902,0.221,0.638) #confidence of easy
#best = findThre(0.584,0.623,1.283) #model a. hard entropy
#best = findThre(0.588,0.666,1.308) #resnet32 block2 entropy :best=1.1
best = findThre(0.4366,0.674,1.276) #resnet32. entropy : best=1.0
print(best)
