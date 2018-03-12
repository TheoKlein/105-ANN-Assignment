# -*- coding: utf-8 -*-
from abc import abstractmethod
import numpy as np
import time
import sys

class Neuron(object):
    def __init__(self):
        self.printInit()

    def symhardlim(self, n):
        for i in range(n.shape[0]):
            for j in range(n.shape[1]):
                if n[i][j] >= 0:
                    n[i][j] = 1.
                else:
                    n[i][j] = -1.
        return np.array(n, dtype='f')

    '''Print information'''
    def logger(self, s):
        print(time.strftime("[%H:%M:%S]") + s)

    '''Print matrix'''
    def printMatrix(self, n):
        if n.size == 0:
            print('  No match!')
            return
        for r in n:
            sys.stdout.write('  ')
            for c in r:
                sys.stdout.write(str(c) + ' ')
            print("")

    '''Print initial value'''
    def printInit(self):
        self.logger("[Info] 初始資料")
        self.logger("[Data] 最大迭代次數: %s 次" % str(self.maxIteration))
        self.logger("[Data] 容忍值: " + str(self.tolerate))
        self.logger("[Data] Learning rate: " + str(self.learningRate))
        print("Initial weight = ")
        self.printMatrix(self.weight)

        print("Initial bias = ")
        self.printMatrix(self.bias)
        print("")

        self.training()

    '''Print training result'''
    def printTrainingResult(self):
        self.logger("[Info] 訓練完成")
        self.logger("[Data] 迭代次數: " + str(self.iteration) + " 次")
        self.logger("[Info] 訓練結果")
        print("Final weight = ")
        self.printMatrix(self.weight)

        print("Final bias = ")
        self.printMatrix(self.bias)
        print("")

        self.testing()

    '''Training neuron netwok'''
    @abstractmethod
    def training(self):
        pass

    '''Use given data to test training result'''
    @abstractmethod
    def testing(self):
        pass