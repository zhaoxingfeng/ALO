# coding:utf-8
"""
作者：zhaoxingfeng	日期：2016.12.02
功能：蚁狮优化算法，Ant Lion Optimizer（ALO）
版本：2.0
参考文献:
[1]Seyedali Mirjalili.The Ant Lion Optimizer[J].ADVANCES IN ENGINEERING SOFTWARE, 2015, 83, 80-98.
[2]Verma, S. Mukherjee, V.Optimal real power rescheduling of generators for congestion management
   using a novel ant lion optimiser[J].ENGINEERING, ELECTRICAL & ELECTRONIC,2016,7,2548-2561.
[3]崔东文, 王宗斌. 基于ALO-ENN算法的洪灾评估模型及应［J].人民珠江, 2016, 37(5): 44-50.
说明：
2015年被人提出来的一种仿生优化算法，Ant Lion Optimizer即蚁狮优化算法，具有全局优化、调节参数少、收敛精度高、鲁棒性
好的优点，已被应用到SVM、Elman神经网络、GM(1,1)以及螺旋桨页面曲线参数寻优等场合。
"""
from __future__ import division
import numpy as np
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
import time


class ALO(object):
    def __init__(self, N, Max_iter, lb, ub, dim, Fobj):
        """
        N：蚂蚁和蚁狮规模，蚂蚁和蚁狮数量相等
        Max_iter：最大迭代次数
        lb, ub ：搜索范围 -> 变量取值范围
        dim：解的维度
        Fobj：价值函数
        """
        self.N = N
        self.Max_iter = Max_iter
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.dim = dim
        self.Fobj = Fobj

    # 初始化 ant 和 antlion 位置
    def Initialization(self):
        x = [[0 for col in range(self.dim)] for row in range(self.N)]
        for i in xrange(self.N):
            for j in xrange(self.dim):
                x[i][j] = random.random() * (self.ub[j]-self.lb[j]) + self.lb[j]
        return x

    # 轮盘赌
    def RouletteWheelSelection(self, weights):
        accumulation = [0 for col in range(self.N)]
        for i in xrange(self.N):
            accumulation[-1] = 0
            accumulation[i] += accumulation[i-1] + weights[i]
        p = random.random() * accumulation[-1]
        for j in xrange(self.N):
            if accumulation[j] > p:
                index = j
                break
        return index

    # 随机游走
    def Random_walk_around_antlion(self, antlion, current_iter):
        if current_iter >= self.Max_iter * 0.95:
            I = 1 + 10**6 * (current_iter/self.Max_iter)
        elif current_iter >= self.Max_iter * 0.9:
            I = 1 + 10**5 * (current_iter/self.Max_iter)
        elif current_iter >= self.Max_iter * 3/4:
            I = 1 + 10**4 * (current_iter/self.Max_iter)
        elif current_iter >= self.Max_iter * 0.5:
            I = 1 + 10**3 * (current_iter/self.Max_iter)
        else:
            I = 1 + 10**2 * (current_iter/self.Max_iter)
        # 公式 (2.10)、(2.11)
        lb, ub = self.lb/I, self.ub/I
        # 公式 (2.8)
        if random.random() < 0.5:
            lb = lb + antlion
        else:
            lb = -lb + antlion
        # 公式 (2.9)
        if random.random() >= 0.5:
            ub = ub + antlion
        else:
            ub = -ub + antlion
        # create n random walks and normalize accroding to lb and ub
        RWs = [[0 for col in range(self.dim)] for row in range(self.Max_iter + 1)]
        for dim in xrange(self.dim):
            # 公式 (2.2)
            X1 = [0]
            for i in xrange(self.Max_iter):
                X1.append(1) if random.random() > 0.5 else X1.append(-1)
            # X：公式 (2.1)
            X = [0 for col in range(self.Max_iter + 1)]
            for j in xrange(self.Max_iter + 1):
                if j == 0:
                    pass
                else:
                    X[j] = X[j-1] + X1[j]
            a, b = min(X), max(X)
            c, d = lb[dim], ub[dim]
            aa = [a for ii in xrange(self.Max_iter + 1)]
            # 公式 (2.7)
            X_norm = [(x-y) * (d-c)/(b-a) + c for x, y in zip(X, aa)]
            for t in xrange(len(X_norm)):
                RWs[t][dim] = X_norm[t]
        return RWs

    # 绘制迭代-误差图
    def Ploterro(self, Current_iter, Convergence_curve):
        mpl.rcParams['font.sans-serif'] = ['Courier New']
        mpl.rcParams['axes.unicode_minus'] = False
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        x = [i for i in range(Current_iter)]
        plt.plot(x, Convergence_curve, 'r-', linewidth=1.5, markersize=5)
        ax.set_xlabel(u'Iter', fontsize=18)
        ax.set_ylabel(u'Best score', fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlim(0, )
        plt.grid(True)
        plt.title("Func = (x[0]-1) ** 2 + (x[1] + 1) ** 2 + x[2] ** 2 + x[3] ** 2")
        plt.show()

    def Run(self):
        # 初始化 ant 和 antlion 位置
        antlion_position = self.Initialization()
        ant_position = self.Initialization()

        Sorted_antlions = [[0 for col in range(self.dim)] for row in range(self.N)]
        # 精英蚁狮位置，精英蚁狮适应度，每次迭代后的最佳适应度，N只蚁狮各自的适应度，蚂蚁的适应度
        Elite_antlion_position = [0 for col in range(self.dim)]
        Elite_antlion_fitness = float('inf')
        Convergence_curve = [0 for col in range(self.Max_iter)]
        antlions_fitness = [0 for col in range(self.N)]
        ants_fitness = [0 for col in range(self.N)]

        # 按照适应度（越小越优）对N只蚁狮排序，取排序后的第一个蚁狮为精英蚁狮
        for i in xrange(self.N):
            antlions_fitness[i] = self.Fobj(antlion_position[i])
        sorted_antlion_fitness = sorted(antlions_fitness)
        sorted_indexes = np.argsort(antlions_fitness)
        for newindex in xrange(self.N):
            Sorted_antlions[newindex] = antlion_position[sorted_indexes[newindex]]
        Elite_antlion_position = Sorted_antlions[0]
        Elite_antlion_fitness = sorted_antlion_fitness[0]

        # 主循环
        for Current_iter in xrange(self.Max_iter):
            print("Iter = " + str(Current_iter))
            for i in xrange(self.N):
                Rolette_index = self.RouletteWheelSelection([1./item for item in sorted_antlion_fitness])
                if Rolette_index == -1:
                    Rolette_index = 1
                RA = self.Random_walk_around_antlion(Sorted_antlions[Rolette_index], Current_iter)
                RE = self.Random_walk_around_antlion(Elite_antlion_position, Current_iter)
                # 公式 (2.13)
                ant_position[i] = [(x + y)/2 for x, y in zip(RA[Current_iter], RE[Current_iter])]
            for j in xrange(self.N):
                for k in xrange(self.dim):
                    if ant_position[j][k] > self.ub[k]:
                        ant_position[j][k] = self.ub[k]
                    elif ant_position[j][k] < self.lb[k]:
                        ant_position[j][k] = self.lb[k]
                    else:
                        pass
                ants_fitness[j] = self.Fobj(ant_position[j])
            double_population = []
            double_population.extend(Sorted_antlions)
            double_population.extend(ant_position)
            double_fitness = []
            double_fitness.extend(sorted_antlion_fitness)
            double_fitness.extend(ants_fitness)

            double_fitness_sorted = sorted(double_fitness)
            I = np.argsort(double_fitness)
            double_sorted_population = []
            for index in I:
                double_sorted_population.append(double_population[index])
            antlions_fitness = double_fitness_sorted[0:self.N]
            Sorted_antlions = double_sorted_population[0:self.N]
            if antlions_fitness[0] <= Elite_antlion_fitness:
                Elite_antlion_fitness = antlions_fitness[0]
                Elite_antlion_position = Sorted_antlions[0]
            Sorted_antlions[0] = Elite_antlion_position
            antlions_fitness[0] = Elite_antlion_fitness
            Convergence_curve[Current_iter] = Elite_antlion_fitness
            Current_iter += 1

            if Elite_antlion_fitness <= 0.001:
                break
        print("Best_score = " + str(Elite_antlion_fitness))
        print("Best_pos = " + str(Elite_antlion_position))
        self.Ploterro(Current_iter, Convergence_curve[0:Current_iter])
        return Elite_antlion_fitness, Elite_antlion_position, Convergence_curve[0:Current_iter]

if __name__ == "__main__":
    # 价值函数
    def Fobj(x):
        cost = (x[0] - 1) ** 2 + (x[1] + 1) ** 2 + x[2] ** 2 + x[3] ** 2
        return cost
    starttime = time.time()
    a = ALO(10, 80, [-1, -1, -1, -1], [1, 1, 1, 1], 4, Fobj)
    Best_score, Best_pos, Cg_curve = a.Run()
    endtime = time.time()
    print("Runtime = " + str(endtime - starttime))
