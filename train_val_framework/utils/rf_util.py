'''
This source code was developed under the DARPA Radio Frequency Machine 
Learning Systems (RFMLS) program contract N00164-18-R-WQ80. All the code 
released here is unclassified and the Government has unlimited rights 
to the code.
'''

from keras.callbacks import Callback
import pickle
from keras import backend as K
from sysmonitor import SysMonitor

class HistoryCheckPoint(Callback):
    def __init__(self, name, **kargs):
        super(HistoryCheckPoint,self).__init__(**kargs)
        self.name = name
        self.monitor = SysMonitor()
        self.monitor.start()
        self.history_log = {'acc':[], 'loss':[], 'val_loss':[], 'val_acc':[]}

    def on_epoch_end(self, epoch, logs={}):
        # things done on end of the epoch
        for key, val in logs.items():
            self.history_log[key].append(val)
        self.monitor.plot(str(epoch))
        with open('./logs/history_log/' + self.name, 'wb') as handle: # saving the history of the model
            pickle.dump(self.history_log, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            
class SGDLearningRateTracker(Callback):
    
    def on_epoch_end(self, epoch, logs={}):
        optimizer = self.model.optimizer
        lr = K.eval(optimizer.lr * (1. / (1. + optimizer.decay * optimizer.iterations)))
        print('\nLR: {:.6f}\n'.format(lr))
