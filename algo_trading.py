# -*- coding: utf-8 -*-
"""
Trading Algorithms

by: Rodrigo Hernández Mota

This script aims to provided objects to facilitate 

"""
# %% Libraries 
# download neural_net library 
def download_package():
    import os
    if not os.path.exists('neural_net') == os.path.exists('forex_analysis'):
        import subprocess
        subprocess.call("download.sh", shell = True)

from neural_net.neural_net import *
from forex_analysis.forex_data import *
from ggplot import *

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# %% First Object 

class cluster_trading_algo:
    '''
    Cluster Trading Algorithm
    This algorithm is based on competitive neural net to...
    '''
    desc = 'Cluster Trading Algorithm'
    download_package()
    
    def __init__(self, x_data = None, y_data = None, competitive_neuron = None):
        # dave dataset (pandas DataFrame)
        self.x_data = x_data
        self.y_data = y_data
        if type(y_data) != type(None):
            self.y_data = pd.DataFrame({'Category':y_data.values})
        # Trained competitive neuron model 
        self.competitive_neuron = competitive_neuron
        
    def train_competitive_neuron(self, x_data = None):
        
        # Verify if a model exists
        if type(self.competitive_neuron) != type(None):
            print('A competitive neural network model has been provided.')
            return 0
        
        # Verify if a dataset was provided
        if type(x_data) == type(None):
            if type(self.x_data) == type(None):
                print('A dataset (pandas DataFrame) must be provided.')
                return 0
            else:
                x_data = self.x_data 
        else:
            x_data = x_data
        
        print('Incomplete method.')
        return 0
    
    def train(self, x_data = None, pr_nt = True):
        
        if type(self.x_data) == type(None) and type(self.competitive_neuron.x_data) != type(None):
            self.x_data = self.competitive_neuron.x_data
        else: 
            self.train_competitive_neuron(x_data = x_data)
            
        # TODO: add training to compet neuron
        
        # Save some variables to easy acces
        self.clusters = self.competitive_neuron.clusters
        self.clusters_size = self.competitive_neuron.clusters_size
        self.elements = self.competitive_neuron.clusters_elements
        self.elements_index = self.competitive_neuron.clusters_elements_index
        self.w = self.competitive_neuron.w
        
        # Determine probabilty of "one"
        cluster_one_probability = []
        trading_sign = []
        for i in self.elements_index:
            total_elements = self.y_data.iloc[i].count()
            one = self.y_data.iloc[i].sum()
            cluster_one_probability.append((one / total_elements).values[0])
            if cluster_one_probability[-1] > 0.5:
                trading_sign.append(1)
            else:
                trading_sign.append(0)
            
        self.clusters_probability = cluster_one_probability
        self.trading_sign = trading_sign
        if pr_nt:
            for i, j, k in zip(self.clusters, self.clusters_size,cluster_one_probability):
                string = "Neuron {} with cluster's size of {} :::: probability = {}".format(i,j,k)
                print(string)
        
    def evaluate(self, decision_boundary = 0.05, x_data = None, ret_rn = False, argret_rn = '[]'):
        
        save = False
        if type(x_data) == type(None):
            x_data = self.x_data
            save = True
        if type(x_data) == type(np.array([])) or type(x_data) == type([]) or type(x_data) == type(pd.Series([])):
            x_data = pd.DataFrame(x_data)
            x_data = x_data.T
            
        w = self.w
        
        aux_x = lambda x: np.argmin(w.apply(lambda y: np.linalg.norm(np.asarray(y) - np.asarray(x))))
        sign_num = pd.DataFrame(x_data.apply(aux_x, 1))
        
        prob = self.clusters_probability
        def map_prob(x):
            x = np.asscalar(x)
            signal = 0
            if prob[x] > 0.5:
                signal = 1
            return signal
        
        def map_prob_spef(x):
            x = np.asscalar(x)
            signal = '---'
            if prob[x] > 0.5 + decision_boundary:
                signal = 'BUY'
            if prob[x] < 0.5 - decision_boundary:
                signal = 'SELL'
            return signal
            
        ts = sign_num.apply(map_prob, 1)
        ss = sign_num.apply(map_prob_spef, 1)
        if save:
            self.trade_signal = ts
            self.specific_signal = ss
        print('Next move: {}'.format(ss.iloc[-1]))
        
        if ret_rn:
            return eval(argret_rn)
   
    def score(self, ret_rn = False):
        
        total_valid = 0
        wrong = 0
        correct = 0
        for i in range(len(self.y_data)):
            prediction = self.specific_signal.iloc[i]
            real = np.asscalar(self.y_data.iloc[i])
            if prediction == '---':
                continue
            if prediction == 'BUY':
                total_valid += 1
                if real == 1.:
                    correct += 1
                else:
                    wrong += 1
            if prediction == 'SELL':
                total_valid += 1
                if real == 0.:
                    correct += 1
                else:
                    wrong += 1
        def div(a,b):
            if b == 0 or b == 0.:
                return 'division by zero'
            return a/b
        
        df_inside = {'Correct':div(correct,total_valid), 'Wrong': div(wrong,total_valid), '_TotalValid':total_valid, '_ratio':div(correct,wrong)}
        score_table = pd.DataFrame(df_inside, index = np.array([0]))
        self.score_table = score_table
        if ret_rn:
            return score_table
      
        
'''
    def plot(self):
        # dataset
        x_data = self.x_data
        y_data = self.y_data
        y_data.columns = ['Y']
        dataset = pd.concat(x_data, self.y_data)
        deteset.columns([str(i) for i in range(len(dataset.columns))])
        x_data = self.x_data
        x_data.columns = ['X', 'Y']
        y_data = self.y_data
        y_data.columns = ['Z']
        # separate positives and negatives
        x_data_positive = x_data[y_data]
        for i in compet_gen.w.columns:
            temp = compet_gen.w[i]
            if i in compet_gen.clusters:
                plt.plot(temp.values[0], temp.values[1],'gx',ms = 10, mew = 3)
        plt.plot(ada_positive.x_data, ada_positive.y_data,'b.', alpha = 0.7)        
        plt.plot(ada_negative.x_data, ada_negative.y_data,'r.', alpha = 0.7)
        plt.show()
'''


# %% Example 
# compet_gen is the competitive neural net
'''
ex = cluster_trading_algo(competitive_neuron = compet_gen, y_data = dataset.Z)
ex.train()
ex.evaluate()
ex.score()

ex.score_table
# %% Test Algorithm 

tg = len(ex.y_data)
sc = []
tv = []
search_ = []
for i in np.arange(0,0.3,0.01):
    ex = cluster_trading_algo(competitive_neuron = compet_gen, y_data = dataset.Z)
    ex.train(pr_nt = False)
    ex.evaluate(decision_boundary = i)
    ex.score()
    if ex.score_table._TotalValid[0] == 0:
        break
    search_.append(i)
    sc.append(ex.score_table.Correct[0])
    tv.append(ex.score_table._TotalValid[0] / tg)
# %%
    
fig = plt.figure()
plt.title('Training data ')
fig.add_subplot(211)
plt.plot(search_, sc, 'b')
plt.ylabel('% of correct signals')
fig.add_subplot(212)
plt.plot(search_, tv, 'r')
plt.ylabel('number of signals \nper total of datapoints')
plt.xlabel('decision_boundary')
plt.show()
'''