# -*- coding: utf-8 -*-
import sys
from adaline import twoneuron
from adaline import fourneuron

def main(option, rate, maxCount, tolerate):
    _2neuron = twoneuron.TwoNeuron(option, rate, maxCount, tolerate)

    _4neuron = fourneuron.FourNeuron(option, rate, maxCount, tolerate)


if __name__ == "__main__":
    if len(sys.argv) < 8:
        print ("輸入參數有誤！\n")
        print ("命令格式:\n  python main.py [資料型態] {learning rate} {最大迭代數} {容忍值}\n")
        print ("資料型態：")
        print ("  problem - 使用題目資料（8 + 4）")
        print ("  extra   - 使用額外資料（1000 + 10）\n")
        print ("參數：")
        print ("  -l <number> 設定learning rate")
        print ("  -m <number> 設定最大迭代次數，以免永不收斂")
        print ("  -t <number> 設定差異容忍值\n")
        print ("範例:\n  python main.py problem -l 0.1 -m 1000 -t 0.001\n    " +
               "- 使用題目資料，指定learning rate = 0.1、最大迭代數 = 1000、容忍值 = 0.001\n")

    else:
        main(sys.argv[1], float(sys.argv[3]), int(sys.argv[5]), float(sys.argv[7]))