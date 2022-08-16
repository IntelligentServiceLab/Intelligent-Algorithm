import random
import numpy as np
import copy
import math
from AbstractCompositeService import AbstractCompositeService

"""
    TLBO algorithm for Industrial Internet
"""


class TLBO():
    """教学优化算法"""

    def __init__(self, abstract_service, task_number, population_size, iteration_number, bounds):
        self.abstract_service = abstract_service  # 抽象服务组合链
        self.task_number = task_number  # 子任务数
        self.population_size = population_size  # 种群规模
        self.iteration_number = iteration_number  # 迭代次数
        self.bounds = bounds  # 候选服务集的上下界

    def initialization(self):
        """初始化阶段:根据种群规模，生成相应个数的个体（服务组合解）;
        通过为每个任务随机挑选候选服务来初始化一个组合服务"""
        population = []  # 种群
        for i in range(0, self.population_size):
            temp = []
            for j in range(0, self.task_number):
                high = self.abstract_service.candidate_service_number
                r = random.randint(0, high - 1)
                # print(r)
                my_temp = copy.deepcopy(self.abstract_service.Time_candidates[j][r])  # 从第j个任务的时间候选服务集选择
                my_temp.append(self.abstract_service.Cost_candidates[j][r])  # 添加对应的成本属性到列表的第4个位置
                temp.append(my_temp)
            population.append(temp)
        return population

    def teacher_phase(self, population, teacher):
        """教师阶段:所有个体通过老师和个体平均值的差值像老师;
        学习参数是 种群列表 和 候选服务集的上下界列表"""

        Mean = self.get_Mean(population)  # 每个任务的平均值列表
        old_population = copy.deepcopy(population)  # 保存算法开始前的种群
        # old_population_fitness = self.fitness_evaluation(old_population)  # 保存旧种群的适应值

        # X_new = []  # 更新后的X
        # X_old = []  # 更新前的X
        # difference = []  # X_teacher(老师)和种群平均值的差值
        # r = random.random()  # 学习步长 ri=rand(0,1)
        # TF = 0  # teach factor教学因素 = round[1 + rand(0, 1)]
        # Mean = 0  # Xi的和 / P

        # 这个循环遍历每个个体
        for i in range(0, self.population_size):
            # TF = round(1 + random.random())  # 教学因素 = round[1 + rand(0, 1)]
            # r = random.random()  # ri=rand(0,1), 学习步长
            # 这个循环与第一个循环一起用来更新每个个体的第j个任务
            for j in range(0, self.task_number):
                TF = round(1 + random.random())  # 教学因素 = round[1 + rand(0, 1)]
                r = random.random()  # ri=rand(0,1), 学习步长
                # 更新第i个解的第j个任务的响应时间
                difference_Res = r * (teacher[j][0] - TF * Mean[j][0])
                population[i][j][0] += difference_Res
                # 更新第i个解的第j个任务的运输时间
                difference_Del = r * (teacher[j][1] - TF * Mean[j][1])
                population[i][j][1] += difference_Del
                # 更新第i个解的第j个任务的执行时间
                difference_Exe = r * (teacher[j][2] - TF * Mean[j][2])
                population[i][j][2] += difference_Exe
                # 更新第i个解的第j个任务的成本
                difference_Cost = r * (teacher[j][3] - TF * Mean[j][3])
                population[i][j][3] += difference_Cost

        # 在教师阶段方法内直接调用refine方法
        # new_population = copy.deepcopy(self.refine(population, self.bounds))

        # 匹配运算结束以后的服务真实值——这个操作导致算法的时间复杂度随着候选服务集的数量增加而增加
        new_population = copy.deepcopy(self.FuzzingMatch(population))

        # 在教师阶段方法内直接调用update方法
        new_population = copy.deepcopy(self.update(old_population, new_population))

        return new_population

    def student_phase(self, population):
        """学生阶段"""
        old_population = copy.deepcopy(population)  # 保存算法开始前的旧种群
        new_population = []  # 初始化新种群
        for i in range(0, self.population_size):
            num_list = self.get_list()  # 获得一个种群大小的数字列表
            num_list.remove(i)
            index = random.choice(num_list)  # 这两步获得一个除了自身以外的随机索引

            # print("第"+str(i)+"个选择了"+"第"+str(index)+"个")
            X = copy.deepcopy(population[i])
            Y = copy.deepcopy(population[index])  # 被选中与X交叉的个体
            # 如果X支配Y, X比Y好
            if ((self.TimeFitness(X) < self.TimeFitness(Y)) and (self.CostFitness(X) < self.CostFitness(Y))) or (
                    (round(self.TimeFitness(X)) == round(self.TimeFitness(Y))) and (
                    self.CostFitness(X) < self.CostFitness(Y))) or ((self.TimeFitness(X) < self.TimeFitness(Y)) and (
                    round(self.CostFitness(X)) == round(self.CostFitness(Y)))):
                r = random.random()  # 学习步长ri=rand(0,1)
                for j in range(0, self.task_number):
                    # 更新第Y的第j个任务的响应时间
                    X[j][0] += r * (X[j][0] - Y[j][0])

                    # 更新第Y的第j个任务的运输时间
                    X[j][1] += r * (X[j][1] - Y[j][1])

                    # 更新第Y的第j个任务的执行时间
                    X[j][2] += r * (X[j][2] - Y[j][2])

                    # 更新第Y的第j个任务的成本
                    X[j][3] += r * (X[j][3] - Y[j][3])

            # 如果Y支配X, Y比X好
            elif ((self.TimeFitness(X) > self.TimeFitness(Y)) and (self.CostFitness(X) > self.CostFitness(Y))) or (
                    (round(self.TimeFitness(X)) == round(self.TimeFitness(Y))) and (
                    self.CostFitness(X) > self.CostFitness(Y))) or ((self.TimeFitness(X) > self.TimeFitness(Y)) and (
                    round(self.CostFitness(X)) == round(self.CostFitness(Y)))):
                r = random.random()  # 学习步长ri=rand(0,1)
                for j in range(0, self.task_number):
                    # 更新第X的第j个任务的响应时间
                    X[j][0] += r * (Y[j][0] - X[j][0])

                    # 更新第X的第j个任务的运输时间
                    X[j][1] += r * (Y[j][1] - X[j][1])

                    # 更新第X的第j个任务的执行时间
                    X[j][2] += r * (Y[j][2] - X[j][2])

                    # 更新第X的第j个任务的成本
                    X[j][3] += r * (Y[j][3] - X[j][3])

            # 若互相不支配，则两个目标函数分别学习
            else:
                # 若X的时间目标强于Y，成本目标弱于Y
                if (self.TimeFitness(X) < self.TimeFitness(Y)) & (self.CostFitness(X) > self.CostFitness(Y)):
                    r = random.random()  # 学习步长ri=rand(0,1)
                    for j in range(0, self.task_number):
                        # 更新第Y的第j个任务的响应时间
                        X[j][0] += r * (X[j][0] - Y[j][0])

                        # 更新第Y的第j个任务的运输时间
                        X[j][1] += r * (X[j][1] - Y[j][1])

                        # 更新第Y的第j个任务的执行时间
                        X[j][2] += r * (X[j][2] - Y[j][2])

                        # 更新第Y的第j个任务的成本
                        X[j][3] += r * (Y[j][3] - X[j][3])

                # 若X的时间目标弱于Y，成本目标强于Y
                else:
                    r = random.random()  # 学习步长ri=rand(0,1)
                    for j in range(0, self.task_number):
                        # 更新第X的第j个任务的响应时间
                        X[j][0] += r * (Y[j][0] - X[j][0])

                        # 更新第X的第j个任务的运输时间
                        X[j][1] += r * (Y[j][1] - X[j][1])

                        # 更新第X的第j个任务的执行时间
                        X[j][2] += r * (Y[j][2] - X[j][2])

                        # 更新第X的第j个任务的成本
                        X[j][3] += r * (X[j][3] - Y[j][3])

            new_population.append(X)

        # 在教师阶段方法内直接调用refine方法
        # new_population = copy.deepcopy(self.refine(population, self.bounds))

        # 匹配运算结束以后的服务真实值——这个操作导致算法的时间复杂度随着候选服务集的数量增加而增加
        new_population = copy.deepcopy(self.FuzzingMatch(new_population))

        # 在教师阶段方法内直接调用update方法
        new_population = copy.deepcopy(self.update(old_population, new_population))

        return new_population

    def find_teacher(self, population):
        """找到种群中的老师(Pareto解集)"""
        teacher = []
        ParetoSet = copy.deepcopy(self.ParetoSearch(population))
        # 若pareto解集里只有一个解
        if len(ParetoSet) == 1:
            teacher = copy.deepcopy(ParetoSet[0])
        # 若pareto解集里有多个解
        else:
            r = np.random.randint(0, len(ParetoSet) - 1)
            teacher = copy.deepcopy(ParetoSet[r])

        return teacher

    def refine(self, population, bounds):
        """refine操作符:防止越界"""
        # 第一步
        # for i in range(0, self.population_size):
        #     for j in range(0, self.task_number):
        #         population[i][j] = round(population[i][j])

        # 第二步
        for i in range(0, self.population_size):
            for j in range(0, self.task_number):
                # 如果超过上界,等于上界
                if population[i][j][0] > bounds[j][0][0]:
                    population[i][j][0] = bounds[j][0][0]

                if population[i][j][1] > bounds[j][0][1]:
                    population[i][j][1] = bounds[j][0][1]

                if population[i][j][2] > bounds[j][0][2]:
                    population[i][j][2] = bounds[j][0][2]
                # 成本上界
                if population[i][j][3] > bounds[j][0][3]:
                    population[i][j][3] = bounds[j][0][3]

                # 如果小于下界，等于下界
                if population[i][j][0] < bounds[j][1][0]:
                    population[i][j][0] = bounds[j][1][0]

                if population[i][j][1] < bounds[j][1][1]:
                    population[i][j][1] = bounds[j][1][1]

                if population[i][j][2] < bounds[j][1][2]:
                    population[i][j][2] = bounds[j][1][2]
                # 成本上界
                if population[i][j][3] < bounds[j][1][3]:
                    population[i][j][3] = bounds[j][1][3]

        return population

    def ParetoSearch(self, population):
        """
            Pareto前沿面搜索
            功能：找出种群中的非支配解集
            参数：种群
            返回值：Pareto解集
        """
        # Pareto非支配解集
        ParetoSet = []

        # 种群的适应值列表
        Fitness_List = []
        # 计算出所有解的两个适应值
        for i in range(0, self.population_size):
            temp = []
            # 添加种群中第i个个体的时间适应值
            TimeFit = copy.deepcopy(self.TimeFitness(population[i]))
            # 添加种群中第i个个体的成本适应值
            CostFit = copy.deepcopy(self.CostFitness(population[i]))
            temp.append(TimeFit)
            temp.append(CostFit)
            Fitness_List.append(temp)

        # 将适应值列表的第三位视作判断是否为pareto解的依据
        for i in range(0, self.population_size):
            Fitness_List[i].append(0)

        # 寻找Pareto解集
        for i in range(0, self.population_size):
            for j in range(0, self.population_size):
                if i != k:
                    if (Fitness_List[i][0] > Fitness_List[j][0] and Fitness_List[i][1] > Fitness_List[j][1]) \
                            or (Fitness_List[i][0] > Fitness_List[j][0] and round(Fitness_List[i][1]) == round(
                        Fitness_List[j][1])) \
                            or (round(Fitness_List[i][0]) == round(Fitness_List[j][0]) and Fitness_List[i][1] >
                                Fitness_List[j][1]):
                        Fitness_List[i][2] = 1
                    else:
                        Fitness_List[i][2] = 0
                else:
                    continue
            if Fitness_List[i][2] == 0:
                ParetoSet.append(population[i])

        # 若不存在非支配解则选出两个适应值相加最小的作为唯一的非支配解
        if len(ParetoSet) == 0:
            pareto = copy.deepcopy(population[0])  # 初始化pareto解为第一个
            pareto_fit = sum(Fitness_List[0])  # 初始化pareto解的适应值

            for i in range(1, self.population_size):
                my_fit = sum(Fitness_List[i])
                if my_fit < pareto_fit:
                    pareto_fit = my_fit
                    pareto = copy.deepcopy(population[i])
            ParetoSet.append(pareto)

        # 将适应值列表的第三位移除
        for i in range(0, self.population_size):
            Fitness_List[i].pop(2)

        return ParetoSet

    def TimeFitness(self, solution):
        """
            时间消耗适应函数
        """
        Time_Fit = 0  # 解的总体时间消耗
        T_1 = 0  # 第一个任务/服务的时间消耗
        T_Exe = 0  # 除第一个任务外其他任务的执行时间消耗
        T_Wait = 0  # 除第一个任务外所有任务的等待时间消耗

        # 计算第一个任务的时间消耗
        T_1 = solution[0][0] + solution[0][1] + solution[0][2]
        # 计算除第一个任务外所有任务的执行时间消耗
        for i in range(1, self.task_number):
            T_Exe += solution[i][2]

        # 计算除第一个任务外所有任务的等待时间消耗——通过与前继任务比较
        for i in range(1, self.task_number):
            a = 0  # 决策变量，当两个相邻任务之间存在时间等待浪费时为1，否则为0
            # 如果后一个服务在前一个服务完成后才到达，则有等待时间
            if (solution[i][0] + solution[i][1]) > (solution[i - 1][0] + solution[i - 1][1] + solution[i - 1][2]):
                a = 1
            T_Wait += a * ((solution[i][0] + solution[i][1]) - (
                    solution[i - 1][0] + solution[i - 1][1] + solution[i - 1][2]))

        Time_Fit = T_1 + T_Exe + T_Wait
        return Time_Fit

    def CostFitness(self, solution):
        """
            成本消耗适应函数
        """
        Cost_Fit = 0  # 解的总体成本消耗
        C_exe = 0  # 服务组合中所有单个任务本身的成本消耗——调用成本和执行成本（目前作为一个来考虑）
        for i in range(0, self.task_number):
            C_exe += solution[i][3]

        C_ware = 0  # 服务组合中的仓储成本warehousing cost
        w = 20  # 仓储成本，待定10万元/day

        # 计算服务组合中仓储成本——通过与前继任务比较
        for i in range(1, self.task_number):
            if (solution[i - 1][0] + solution[i - 1][1] + solution[i - 1][2]) > (solution[i][0] + solution[i][1]):
                C_ware += w * ((solution[i - 1][0] + solution[i - 1][1] + solution[i - 1][2]) - (
                        solution[i][0] + solution[i][1]))

        Cost_Fit = C_exe + C_ware
        return Cost_Fit

    def update(self, old_group, new_group):
        """这个函数用来更新种群:若新解支配旧解，则替换;否则保留"""

        updated_group = []
        for i in range(self.population_size):
            # 如果新解支配旧解
            if ((self.TimeFitness(new_group[i]) < self.TimeFitness(old_group[i])) and (
                    self.CostFitness(new_group[i]) < self.CostFitness(old_group[i]))) or (
                    (round(self.TimeFitness(new_group[i])) == round(self.TimeFitness(old_group[i]))) and (
                    self.CostFitness(new_group[i]) < self.CostFitness(old_group[i]))) or (
                    (self.TimeFitness(new_group[i]) < self.TimeFitness(old_group[i])) and (
                    round(self.CostFitness(new_group[i])) == round(self.CostFitness(old_group[i])))):
                updated_group.append(new_group[i])
            else:
                updated_group.append(old_group[i])

        return updated_group

    def get_list(self):
        """"为了学生阶段获得一个种群大小的数字列表"""
        nums_list = []
        for i in range(0, self.population_size):
            nums_list.append(i)
        return nums_list

    def get_Mean(self, population):
        """获得种群中 每个任务 的平均值;
           参数为种群;
           返回值为每个任务平均值的列表
        """
        Mean = []
        for i in range(0, self.task_number):
            # 第i个任务的响应时间之和
            Sum_Res = 0
            # 第i个任务的运输时间之和
            Sum_Del = 0
            # 第i个任务的执行时间之和
            Sum_Exe = 0
            # 第i个任务的成本之和
            Sum_Cost = 0
            # 第i个任务的平均值列表
            Mean_i = []
            for j in range(0, self.population_size):
                # 响应时间
                Sum_Res += population[j][i][0]
                # 运输时间
                Sum_Del += population[j][i][1]
                # 执行时间
                Sum_Exe += population[j][i][2]
                # 成本
                Sum_Cost += population[j][i][3]
            Mean_i.append(Sum_Res / self.population_size)
            Mean_i.append(Sum_Del / self.population_size)
            Mean_i.append(Sum_Exe / self.population_size)
            Mean_i.append(Sum_Cost / self.population_size)
            Mean.append(Mean_i)
        return Mean

    def FuzzingMatch(self, population):
        """
            用最小欧氏距离，在候选服务集中寻找种群中与每个个体的单个服务值最接近的真实值
            参数：种群
        """
        new_population = []  # 初始化新种群
        # 对于种群中的每个个体
        for i in range(0, self.population_size):
            temp_list = []

            # 对于每个个体的每个任务
            for j in range(0, self.task_number):
                map_service = [0, 0, 0, 0]  # 初始化匹配的服务
                # 对于每个任务的候选服务集
                difference = 1000000
                E_distance = 0  # 初始化欧氏距离列表
                # 第j个任务的候选服务集
                Time_candidates_j = copy.deepcopy(self.abstract_service.Time_candidates[j])
                Cost_candidates_j = copy.deepcopy(self.abstract_service.Cost_candidates[j])
                # 欧氏距离中被替换个体的参数
                s_i_res = population[i][j][0]
                s_i_del = population[i][j][1]
                s_i_exe = population[i][j][2]
                s_i_cost = population[i][j][3]
                for k in range(0, self.abstract_service.candidate_service_number):
                    # 欧氏距离中替换个体的参数
                    s_j_res = Time_candidates_j[k][0]
                    s_j_del = Time_candidates_j[k][1]
                    s_j_exe = Time_candidates_j[k][2]
                    s_j_cost = Cost_candidates_j[k]
                    # 计算种群第i个个体任务j的服务与第j个任务候选服务集中第k个候选服务的 欧氏距离
                    E_distance = abs(s_i_res - s_j_res) + abs(s_i_del - s_j_del) + abs(s_i_exe - s_j_exe) + abs(
                        s_i_cost - s_j_cost)

                    # 为种群第i个个体任务j的服务 匹配欧氏距离最小的真实服务
                    if E_distance < difference:
                        map_service[0] = copy.deepcopy(Time_candidates_j[k][0])
                        map_service[1] = copy.deepcopy(Time_candidates_j[k][1])
                        map_service[2] = copy.deepcopy(Time_candidates_j[k][2])
                        map_service[3] = copy.deepcopy(Cost_candidates_j[k])
                        difference = E_distance

                # 将第i个个体第j个任务的真实服务添加进来
                temp_list.append(map_service)
            # 将第i个个体所有任务的真实服务添加进来
            new_population.append(temp_list)

        return new_population
