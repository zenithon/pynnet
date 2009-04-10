from nlins import *
from simplenet import *
from errors import *
from trainers import *
import numpy as N
import copy
import time

NINPUT=120
NOUTPUT=2
NHIDDEN=45
STEP=10

def splitMat(mat,targetnum):
    return (mat[:,:-targetnum],mat[:,-targetnum:])

def doEpoch(nnet,step):
    nnet.train_loop(trainer,epochs=step)
    
def initNNet():
    nnet=SimpleNet(NINPUT,NHIDDEN,NOUTPUT,error=mse,onlin=sigmoid)
    return nnet

def earlyStopping(nnet,train,valid,thresh=10):
    bestError=1
    currError=1
    stage=0
    count=0
    history=[]
    bestNNet=copy.deepcopy(nnet)
    while count<thresh:
        doEpoch(nnet,STEP)
        stage+=STEP
        #trainError=getError(nnet,train)
        #currError=getError(nnet,valid)
	trainIn,trainTar=splitMat(train,2)
	validIn,validTar=splitMat(valid,2)
        trainError=nnet.test(trainIn,trainTar)
        currError=nnet.test(validIn,validTar)
	trainCError=class_error(nnet.eval(trainIn),trainTar)
	validCError=class_error(nnet.eval(validIn),validTar)
        history.append([trainError,currError,trainCError,validCError])
        print '%iN\n%iE'%(NHIDDEN,stage)
        print 'Train Error : %f\nValid Error : %f'%(trainError, currError)
	print 'Train Class Error : %f\nValid Class Error : %f'%(trainCError, validCError)
        if currError<bestError:
            count=0
            bestError=currError
            bestNNet=copy.deepcopy(nnet)
        else :
            count+=1
    return bestNNet,history
    
def runEarlyStopping(prefix='early_stop'):
    timeA=time.time()
    fileName='%s_%iN'%(prefix,NHIDDEN)
    nnet=initNNet()
    nnet,hist=earlyStopping(nnet,trainMatrix,testMatrix)
    nnet.save('%s.nnet'%fileName)
    N.savetxt('%s_history.mat'%fileName,history)
    timeB=time.time()
    print 'Total time : %f seconds'%(timeB-timeA) 
    return nnet, history


trainMatrix=N.loadtxt('piano_guitar_train.mat')
testMatrix=N.loadtxt('piano_guitar_test.mat')
(inputs,targets)=splitMat(trainMatrix,2)
trainer = bprop(inputs, targets)
#nnet=SimpleNet(120,45,2,onlin=sigmoid)


