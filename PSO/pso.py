import random
import numpy as np
import dataset
import copy

"""
    PSO for Industrial Internet
"""


class PSO():
    """粒子群算法"""

    def __init__(self, abstract_service, task_number, population_size, iteration_number, bounds):

        self.w = 0.8  # w为惯性因子
        self.c1 = 2
        self.c2 = 2  # c1, c2为学习因子，一般取2
        self.bounds = bounds  # 位置的边界
        self.abstract_service = abstract_service  # 抽象服务组合链
        self.task_number = task_number  # 任务数
        self.population_size = population_size  # 种群规模(粒子数量)
        self.iteration_number = iteration_number  # 迭代次数

    def initialization_X(self):
        """初始化阶段:根据种群规模，生成相应个数的个体（服务组合解）;
        通过为每个任务随机挑选候选服务来初始化一个组合服务
           初始化解的 位置
        """
        # 初始化位置：
        population_X = []  # 种群
        for i in range(0, self.population_size):
            temp = []
            for j in range(0, self.task_number):
                high = self.abstract_service.candidate_service_number
                r = random.randint(0, high - 1)
                # print(r)
                my_temp = copy.deepcopy(self.abstract_service.Time_candidates[j][r])  # 从第j个任务的时间候选服务集选择
                my_temp.append(self.abstract_service.Cost_candidates[j][r])  # 添加对应的成本属性到列表的第4个位置
                temp.append(my_temp)
            population_X.append(temp)

        return population_X

    def initialization_V(self, Vmin, Vmax):
        """
            初始化解的 速度
        """
        population_V = []  # 速度
        for i in range(0, self.population_size):
            temp = [0, 0, 0, 0]
            i_task = []
            for j in range(0, self.task_number):
                temp[0] = random.uniform(Vmin[j][0], Vmax[j][0])
                temp[1] = random.uniform(Vmin[j][1], Vmax[j][1])
                temp[2] = random.uniform(Vmin[j][2], Vmax[j][2])
                temp[3] = random.uniform(Vmin[j][3], Vmax[j][3])
                i_task.append(temp)
            population_V.append(i_task)

        return population_V

    def get_Vmax(self, bounds):
        """获取速度的上下界"""
        Vmax = []  # 每个任务的速度上界
        # 速度的上界
        for i in range(self.task_number):
            temp = [0, 0, 0, 0]
            temp[0] = 0.3 * (bounds[i][0][0])
            temp[1] = 0.3 * (bounds[i][0][1])
            temp[2] = 0.3 * (bounds[i][0][2])
            temp[3] = 0.3 * (bounds[i][0][3])

            Vmax.append(temp)
        return Vmax

    def get_Vmin(self, bounds):
        """获取速度的上下界"""
        Vmin = []  # 每个任务的速度下界
        for i in range(self.task_number):
            temp = [0, 0, 0, 0]
            temp[0] = (-0.3) * (bounds[i][0][0])
            temp[1] = (-0.3) * (bounds[i][0][1])
            temp[2] = (-0.3) * (bounds[i][0][2])
            temp[3] = (-0.3) * (bounds[i][0][3])

            Vmin.append(temp)
        return Vmin

    def update_X(self, pop_X, pop_V):
        """更新位置"""
        new_pop_X = []  # 种群更新后的位置
        for i in range(0, self.population_size):
            temp = []
            new_X = [0, 0, 0, 0]
            for j in range(0, self.task_number):
                new_X[0] = pop_X[i][j][0] + pop_V[i][j][0]
                new_X[1] = pop_X[i][j][1] + pop_V[i][j][1]
                new_X[2] = pop_X[i][j][2] + pop_V[i][j][2]
                new_X[3] = pop_X[i][j][3] + pop_V[i][j][3]

                # 判断是否越上界
                if new_X[0] > self.bounds[j][0][0]:
                    new_X[0] = self.bounds[j][0][0]
                if new_X[1] > self.bounds[j][0][1]:
                    new_X[1] = self.bounds[j][0][1]
                if new_X[2] > self.bounds[j][0][2]:
                    new_X[2] = self.bounds[j][0][2]
                if new_X[3] > self.bounds[j][0][3]:
                    new_X[3] = self.bounds[j][0][3]

                # 判断是否越下界
                if new_X[0] < self.bounds[j][1][0]:
                    new_X[0] = self.bounds[j][1][0]
                if new_X[1] < self.bounds[j][1][1]:
                    new_X[1] = self.bounds[j][1][1]
                if new_X[2] < self.bounds[j][1][2]:
                    new_X[2] = self.bounds[j][1][2]
                if new_X[3] < self.bounds[j][1][3]:
                    new_X[3] = self.bounds[j][1][3]
                temp.append(new_X)
            new_pop_X.append(temp)
        return new_pop_X

    def update_V(self, pop_X, pop_V, pbest, gbest, Vmin, Vmax):
        """更新速度"""
        new_pop_V = []  # 种群更新后的速度
        for i in range(0, self.population_size):
            temp = []
            speed = [0, 0, 0, 0]
            for j in range(0, self.task_number):
                r1 = random.random()
                r2 = random.random()
                speed[0] = self.w * pop_V[i][j][0] + self.c1 * r1 * (pbest[i][j][0] - pop_X[i][j][0]) + self.c2 * r2 * (
                        gbest[j][0] - pop_X[i][j][0])
                speed[1] = self.w * pop_V[i][j][1] + self.c1 * r1 * (pbest[i][j][1] - pop_X[i][j][1]) + self.c2 * r2 * (
                        gbest[j][1] - pop_X[i][j][1])
                speed[2] = self.w * pop_V[i][j][2] + self.c1 * r1 * (pbest[i][j][2] - pop_X[i][j][2]) + self.c2 * r2 * (
                        gbest[j][2] - pop_X[i][j][2])
                speed[3] = self.w * pop_V[i][j][3] + self.c1 * r1 * (pbest[i][j][3] - pop_X[i][j][3]) + self.c2 * r2 * (
                        gbest[j][3] - pop_X[i][j][3])

                # 判断是否越上界
                if speed[0] > Vmax[j][0]:
                    speed[0] = Vmax[j][0]

                if speed[1] > Vmax[j][1]:
                    speed[1] = Vmax[j][1]

                if speed[2] > Vmax[j][2]:
                    speed[2] = Vmax[j][2]

                if speed[3] > Vmax[j][3]:
                    speed[3] = Vmax[j][3]

                # 判断是否越下界
                if speed[0] < Vmin[j][0]:
                    speed[0] = Vmin[j][0]
                if speed[1] < Vmin[j][1]:
                    speed[1] = Vmin[j][1]
                if speed[2] < Vmin[j][2]:
                    speed[2] = Vmin[j][2]
                if speed[3] < Vmin[j][3]:
                    speed[3] = Vmin[j][3]

                temp.append(speed)
            new_pop_V.append(temp)

        return new_pop_V

    def save_pbest(self, pbest, pop_X):
        """更新个体历史最优"""
        updated_pbest = []
        for i in range(self.population_size):
            # 如果新解支配旧解
            if ((self.TimeFitness(pop_X[i]) < self.TimeFitness(pbest[i])) and (
                    self.CostFitness(pop_X[i]) < self.CostFitness(pbest[i]))) or (
                    (round(self.TimeFitness(pop_X[i])) == round(self.TimeFitness(pbest[i]))) and (
                    self.CostFitness(pop_X[i]) < self.CostFitness(pbest[i]))) or (
                    (self.TimeFitness(pop_X[i]) < self.TimeFitness(pbest[i])) and (
                    round(self.CostFitness(pop_X[i])) == round(self.CostFitness(pbest[i])))):
                updated_pbest.append(pop_X[i])
            else:
                updated_pbest.append(pbest[i])

        return updated_pbest

    def save_gbest(self, gbest, Pop_X):
        """更新种群历史最优"""

        Pareto = copy.deepcopy(self.ParetoSearch(Pop_X))

        for i in range(0, len(Pareto)):
            if ((self.TimeFitness(Pareto[i]) < self.TimeFitness(gbest)) and (
                    self.CostFitness(Pareto[i]) < self.CostFitness(gbest))) or (
                    (round(self.TimeFitness(Pareto[i])) == round(self.TimeFitness(gbest))) and (
                    self.CostFitness(Pareto[i]) < self.CostFitness(gbest))) or (
                    (self.TimeFitness(Pareto[i]) < self.TimeFitness(gbest)) and (
                    round(self.CostFitness(Pareto[i])) == round(self.CostFitness(gbest)))):
                gbest = copy.deepcopy(Pareto[i])

        return gbest

    def find_teacher(self, population):
        """找到种群中的老师(Pareto解集)"""
        teacher = []
        ParetoSet = copy.deepcopy(self.ParetoSearch(population))
        # 若pareto解集里只有一个解
        if len(ParetoSet) == 1:
            teacher = copy.deepcopy(ParetoSet[0])
        # 若pareto解集里有多个解
        else:
            r = np.random.randint(0, len(ParetoSet)-1)
            teacher = copy.deepcopy(ParetoSet[r])

        return teacher

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
                if i!=j:
                    # 若i被j支配
                    if (Fitness_List[i][0] > Fitness_List[j][0] and Fitness_List[i][1] > Fitness_List[j][1]) \
                            or (Fitness_List[i][0] > Fitness_List[j][0] and round(Fitness_List[i][1]) == round(Fitness_List[j][1])) \
                            or (round(Fitness_List[i][0]) == round(Fitness_List[j][0]) and Fitness_List[i][1] > Fitness_List[j][1]):
                        Fitness_List[i][2] = 1
                    else:
                        Fitness_List[i][2] = 0
                        break
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
        w = 20  # 仓储成本，待定500yuan/day

        # 计算服务组合中仓储成本——通过与前继任务比较
        for i in range(1, self.task_number):
            if (solution[i - 1][0] + solution[i - 1][1] + solution[i - 1][2]) > (solution[i][0] + solution[i][1]):
                C_ware += w * ((solution[i - 1][0] + solution[i - 1][1] + solution[i - 1][2]) - (
                        solution[i][0] + solution[i][1]))

        Cost_Fit = C_exe + C_ware
        return Cost_Fit

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

    def update(self, old_group, new_group):
        """这个函数用来更新种群:若新解支配旧解，则替换;否则保留"""

        updated_group = []
        for i in range(self.population_size):
            # 如果新解支配旧解
            if ((self.TimeFitness(new_group[i]) < self.TimeFitness(old_group[i])) and (
                    self.CostFitness(new_group[i]) < self.CostFitness(old_group[i]))) or (
                    (self.TimeFitness(new_group[i]) == self.TimeFitness(old_group[i])) and (
                    self.CostFitness(new_group[i]) < self.CostFitness(old_group[i]))) or (
                    (self.TimeFitness(new_group[i]) < self.TimeFitness(old_group[i])) and (
                    self.CostFitness(new_group[i]) == self.CostFitness(old_group[i]))):
                updated_group.append(new_group[i])
            else:
                updated_group.append(old_group[i])

        return updated_group

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
