# -*- coding: utf-8 -*-
from __future__ import division
from adaline.neuron import Neuron
from matplotlib import pyplot as plt
import numpy as np
import time

class TwoNeuron(Neuron):
    np.set_printoptions(precision = 4)
    def __init__(self, option, rate, maxCount, tolerate):
        print("==============Two-neuron perceptron==============")
        self.weight = np.array([[1, 0], [0, 1]], dtype='f')
        self.bias = np.array([[1], [1]], dtype='f')
        self.iteration = 0
        self.maxIteration = maxCount
        self.learningRate = rate
        self.tolerate = tolerate
        self.noChange = False
        self.option = 0
        if option == 'problem':
            self.option = 1
        elif option == 'extra':
            self.option = 2
        self.inputData = []
        self.trainingData = []
        super(TwoNeuron, self).__init__()

    '''Transform target to specific matrix'''
    def target2Matrix(self, type):
        matrix = {
            'A': np.array([[1], [1]], dtype='f'),
            'B': np.array([[1], [-1]], dtype='f'),
            'D': np.array([[-1], [-1]], dtype='f'),
            'C': np.array([[-1], [1]], dtype='f')
        }
        return matrix[type]

    '''Transform matrix to specific target'''
    def matrix2Target(self,matrix):
        if np.array_equal(matrix, np.array([[1], [1]], dtype='f')):
            return 'A'
        elif np.array_equal(matrix, np.array([[1], [-1]], dtype='f')):
            return 'B'
        elif np.array_equal(matrix, np.array([[-1], [-1]], dtype='f')):
            return 'D'
        elif np.array_equal(matrix, np.array([[-1], [1]], dtype='f')):
            return 'C'

    '''Main training function'''
    def training(self):
        self.logger("[Info] 開始訓練")
        e_old = np.array([[0], [0]], dtype='f')
        if self.option == 1:
            self.logger("[Info] 使用題目訓練資料（8筆）")
            while not self.noChange and self.iteration < self.maxIteration:
                self.noChange = True
                e_new = np.array([[0], [0]], dtype='f')

                with open('data/training_data1.txt') as f:
                    for line in f:
                        cont = line.split()
                        p = np.array([[float(cont[0])], [float(cont[1])]], dtype='f')
                        t = np.array([[float(cont[2])], [float(cont[3])]], dtype='f')

                        a = np.add(np.dot(self.weight, p), self.bias)
                        e = t - a

                        self.weight = np.add(self.weight, 2 * self.learningRate * np.dot(e, np.transpose(p)))
                        self.bias = np.add(self.bias, 2 * self.learningRate * e)

                        e_new = np.add(e_new, e)

                e_new = e_new / 8
                if not np.all(np.less(np.absolute(e_new - e_old), self.tolerate)):
                    self.noChange = False
                e_old = e_new
                self.iteration += 1
        elif self.option == 2:
            self.logger("[Info] 使用額外訓練資料（1000筆）")
            while not self.noChange and self.iteration < self.maxIteration:
                self.noChange = True
                e_new = np.array([[0], [0]], dtype='f')

                with open('data/training_data2.txt') as f:
                    for line in f:
                        cont = line.split()
                        self.trainingData.append([float(cont[0]), float(cont[1]), cont[2]])
                        p = np.array([[float(cont[0])], [float(cont[1])]], dtype='f')
                        t = self.target2Matrix(str(cont[2]))

                        a = np.add(np.dot(self.weight, p), self.bias)
                        e = t - a

                        self.weight = np.add(self.weight, 2 * self.learningRate * np.dot(e, np.transpose(p)))
                        self.bias = np.add(self.bias, 2 * self.learningRate * e)


                        print(a)
                        print(e)
                        print(self.weight)
                        print(self.bias)
                        print("--------------")
                        time.sleep(1)

                        e_new = np.add(e_new, e)

                e_new = e_new / 1000
                if not np.all(np.less(np.absolute(e_new - e_old), self.tolerate)):
                    self.noChange = False

                e_old = e_new
                self.iteration += 1
        self.printTrainingResult()

    '''Main test function'''
    def testing(self):
        count = 1
        if self.option == 1:
            self.logger("[Info] 使用題目測試資料（4筆）")
            with open('data/testing_data1.txt') as f:
                for line in f:
                    cont = line.split(' ')
                    p = np.array([[float(cont[0])], [float(cont[1])]], dtype='f')
                    a = self.symhardlim(np.dot(self.weight, p) + self.bias)
                    print("Test data %2s = %s" % (str(count), self.matrix2Target(a)))
                    count += 1
        elif self.option == 2:
            self.logger("[Info] 使用額外測試資料（10筆）")
            with open('data/testing_data2.txt') as f:
                for line in f:
                    cont = line.split()
                    p = np.array([[float(cont[0])], [float(cont[1])]], dtype='f')
                    a = self.symhardlim(np.dot(self.weight, p) + self.bias)
                    self.inputData.append([float(cont[0]), float(cont[1]), self.matrix2Target(a)])
                    print("Test data %02s: %s" % (str(count), self.matrix2Target(a)))
                    count += 1
            #self.graph()

    def graph(self):
        self.logger('[Info] 製作圖檔中')
        x = np.arange(-20, 30)

        y1 = (-self.bias[0][0] - (self.weight[0][0] * x)) / self.weight[0][1]
        y2 = (-self.bias[1][0] - (self.weight[1][0] * x)) / self.weight[1][1]

        plt.xlim(-10, 20)
        plt.ylim(-15, 15)

        plt.xlabel('x-axis')
        plt.ylabel('y-axis')

        for t in self.inputData:
            if t[2] == 'A':
                plt.scatter(t[0], t[1], marker='o', label='A', color='r')
            elif t[2] == 'B':
                plt.scatter(t[0], t[1], marker='x', label='B', color='b')
            elif t[2] == 'C':
                plt.scatter(t[0], t[1], marker='^', label='C', color='m')
            else:
                plt.scatter(t[0], t[1], marker='v', label='D', color='g')

        plt.legend(['A', 'B', 'C', 'D'], loc='upper left')

        plt.plot(x, y1, color='black')
        plt.plot(x, y2, color='black')

        for t in self.trainingData:
            plt.scatter(t[0], t[1], marker=',', color='black')


        file = time.strftime("%Y%m%d%H%M%S-")
        file += '額外資料-TwoNeuronResult.png'
        plt.savefig(file)
        plt.clf()
        plt.cla()
        plt.close()
        self.logger('[Info] 圖檔製作完成！')
        self.logger('[Info] 檔案名稱為「%s」' % file)