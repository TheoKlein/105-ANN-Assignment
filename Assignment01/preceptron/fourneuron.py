# -*- coding: utf-8 -*-
from __future__ import division
from preceptron.neuron import Neuron
from matplotlib import pyplot as plt
import numpy as np
import time
import sys

class FourNeuron(Neuron):
    np.set_printoptions(precision = 4)
    def __init__(self, option, rate, maxCount):
        print("\n==============Four-neuron perceptron=============")
        self.logger("[Info] 隨機生成weight、bios")
        self.weight = np.random.rand(4, 2)
        self.bios = np.random.rand(4, 1)
        self.iteration = 0
        self.maxIteration = maxCount
        self.learningRate = rate
        self.noChange = False
        self.option = 0
        if option == 'problem':
            self.option = 1
        elif option == 'extra':
            self.option = 2
        self.inputData = []
        self.trainingData = []
        super(FourNeuron, self).__init__()

    '''Transform 2x1 matrix to 4x1 matrix'''
    def two2four(self, matrix):
        if np.array_equal(matrix, np.array([[0], [0]], dtype='f')):
            return np.array([[1], [0], [0] ,[0]], dtype='f')
        elif np.array_equal(matrix, np.array([[0], [1]], dtype='f')):
            return np.array([[0], [1], [0] ,[0]], dtype='f')
        elif np.array_equal(matrix, np.array([[1], [0]], dtype='f')):
            return np.array([[0], [0], [1] ,[0]], dtype='f')
        elif np.array_equal(matrix, np.array([[1], [1]], dtype='f')):
            return np.array([[0], [0], [0] ,[1]], dtype='f')

    '''Transform 4x1 matrix to 2x1 matrix'''
    def four2two(self, matrix):
        if np.array_equal(matrix, np.array([[1], [0], [0] ,[0]], dtype='f')):
            return np.array([[0], [0]], dtype='f')
        elif np.array_equal(matrix, np.array([[0], [1], [0] ,[0]], dtype='f')):
            return np.array([[0], [1]], dtype='f')
        elif np.array_equal(matrix, np.array([[0], [0], [1] ,[0]], dtype='f')):
            return np.array([[1], [0]], dtype='f')
        elif np.array_equal(matrix, np.array([[0], [0], [0] ,[1]], dtype='f')):
            return np.array([[1], [1]], dtype='f')
        else:
            return 'none'

    '''Transform target to specific matrix'''
    def target2Matrix(self, type):
        matrix = {
            'A': np.array([[1], [0], [0], [0]], dtype='f'),
            'B': np.array([[0], [1], [0], [0]], dtype='f'),
            'C': np.array([[0], [0], [1], [0]], dtype='f'),
            'D': np.array([[0], [0], [0], [1]], dtype='f')
        }
        return matrix[type]

    '''Transform matrix to specific target'''
    def matrix2Target(self, matrix):
        if np.array_equal(matrix, np.array([[1], [0], [0], [0]], dtype='f')):
            return 'A'
        elif np.array_equal(matrix, np.array([[0], [1], [0], [0]], dtype='f')):
            return 'B'
        elif np.array_equal(matrix, np.array([[0], [0], [1], [0]], dtype='f')):
            return 'C'
        elif np.array_equal(matrix, np.array([[0], [0], [0], [1]], dtype='f')):
            return 'D'
        else:
            return 'Not belongs to A, B, C nor D.'

    '''Main training function'''
    def training(self):
        self.logger("[Info] 開始訓練")
        if self.option == 1:
            self.logger("[Info] 使用題目訓練資料（8筆）")
            while not self.noChange and self.iteration < self.maxIteration:
                self.noChange = True

                with open('data/training_data1.txt') as f:
                    for line in f:
                        cont = line.split()
                        p = np.array([[float(cont[0])], [float(cont[1])]], dtype='f')
                        t = self.two2four(np.array([[float(cont[2])], [float(cont[3])]], dtype='f'))

                        a = self.hardlim(np.add(np.dot(self.weight, p), self.bios))

                        if not np.array_equal(t, a):
                            self.noChange = False
                            e = t - a
                            self.weight = np.add(self.weight, self.learningRate * np.dot(e, np.transpose(p)))
                            self.bios = np.add(self.bios, self.learningRate * e)
                self.iteration += 1
        elif self.option == 2:
            self.logger("[Info] 使用額外訓練資料（1000筆）")
            while not self.noChange and self.iteration < self.maxIteration:
                self.noChange = True

                with open('data/training_data2.txt') as f:
                    for line in f:
                        cont = line.split()
                        self.trainingData.append([float(cont[0]), float(cont[1]), cont[2]])
                        p = np.array([[float(cont[0])], [float(cont[1])]], dtype='f')
                        t = self.target2Matrix(cont[2])

                        a = self.hardlim(np.add(np.dot(self.weight, p), self.bios))
                        self.trainingData.append([float(cont[0]), float(cont[1]), cont[2]])

                        if not np.array_equal(t, a):
                            self.noChange = False
                            e = t - a
                            self.weight = np.add(self.weight, np.dot(e, np.transpose(p)))
                            self.bios = np.add(self.bios, e)
                self.iteration += 1
        self.printTrainingResult()

    '''Main test function'''
    def testing(self):
        if self.option == 1:
            count = 1
            self.logger("[Info] 使用題目測試資料（4筆）")
            with open('data/testing_data1.txt') as f:
                for line in f:
                    cont = line.split()
                    p = np.array([[float(cont[0])], [float(cont[1])]], dtype='f')
                    a = self.hardlim(np.dot(self.weight, p) + self.bios)
                    print("Test data %2s = %s" % (str(count), self.matrix2Target(a)))
                    count += 1
        elif self.option == 2:
            count = 1
            self.logger("[Info] 使用額外測試資料（10筆）")
            with open('data/testing_data2.txt') as f:
                for line in f:
                    cont = line.split()
                    p = np.array([[float(cont[0])], [float(cont[1])]], dtype='f')
                    a = self.hardlim(np.dot(self.weight, p) + self.bios)
                    self.inputData.append([float(cont[0]), float(cont[1]), self.matrix2Target(a)])
                    print("Test data %2s: %s" % (str(count), self.matrix2Target(a)))
                    count += 1
            self.graph()

    def graph(self):
        self.logger('[Info] 製作圖檔中')
        x = np.arange(-20, 30)

        y1 = (-self.bios[0][0] - (self.weight[0][0] * x)) / self.weight[0][1]
        y2 = (-self.bios[1][0] - (self.weight[1][0] * x)) / self.weight[1][1]
        y3 = (-self.bios[2][0] - (self.weight[2][0] * x)) / self.weight[2][1]
        y4 = (-self.bios[3][0] - (self.weight[3][0] * x)) / self.weight[3][1]

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
            elif t[2] == 'D':
                plt.scatter(t[0], t[1], marker='v', label='D', color='g')
            else:
                plt.scatter(t[0], t[1], marker='*', label='None', color='c')

        plt.legend(['A', 'B', 'C', 'D', 'Others'], loc='upper left')

        plt.plot(x, y1, color='black')
        plt.plot(x, y2, color='black')
        plt.plot(x, y3, color='black')
        plt.plot(x, y4, color='black')

        for t in self.trainingData:
            plt.scatter(t[0], t[1], marker=',', color='black')

        file = time.strftime("%Y%m%d%H%M%S-")
        file += '額外資料-FourNeuronResult.png'
        plt.savefig(file)
        plt.clf()
        plt.cla()
        plt.close()
        self.logger('[Info] 圖檔製作完成！')
        self.logger('[Info] 檔案名稱為「%s」' % file)