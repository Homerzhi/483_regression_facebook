import mlp
import numpy as np
import pylab as pl
import sys

import numpy as np

data =np.loadtxt('facebook.csv', delimiter = ';')

data = data-data.mean(axis=0)

imax = np.concatenate((data.max(axis=0)*np.ones((1,19)),np.abs(data.min(axis=0)*np.ones((1,19)))),axis=0).max(axis=0)    #find max of each dimension
data = data/imax 



#set target


train = data[::2,:]
traintargets = data[::2,18].reshape((np.shape(train)[0]),1)
    
valid = data[1::4,:]
validtarget = data[1::4,18].reshape((np.shape(valid)[0]),1)

test = data[3::4,:]
testtargets = data[3::4,18].reshape((np.shape(test)[0]),1)

net = mlp.mlp(train, traintargets, 30, outtype = 'linear')
iteration=500
net.earlystopping(train, traintargets, valid, validtarget, 0.4, iteration)
#net.confmat(test,testtargets)



test = np.concatenate((test,-np.ones((np.shape(test)[0],1))),axis=1)
testout = net.mlpfwd(test)

pl.figure()
pl.plot(np.arange(np.shape(test)[0]),testout,'.')
pl.plot(np.arange(np.shape(test)[0]),testtargets,'x')
pl.legend(('Predictions','Targets'))
print 0.5*np.sum((testtargets-testout)**2)
pl.show()





