import random
import numpy as np
import dataset8_10_100
import copy

"""
    Genetic algorithm for Industrial Internet
"""


class GA():
    """遗传算法"""

    def __init__(self, abstract_service, task_number, population_size, iteration_number, bounds):
        self.crossover_probability = 0.88  # 交叉率
        self.mutation_probability = 0.2  # 变异率
        self.abstract_service = abstract_service  # 抽象服务组合链
        self.task_number = task_number  # 任务数
        self.population_size = population_size  # 种群规模
        self.iteration_number = iteration_number  # 迭代次数
        self.bounds = bounds  # 候选服务集的上下界列表

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

    def Selection(self, population):
        """选择操作：采用锦标赛选择算法（ps：由于本场景下，个体的适应值越小表示越好，故不宜使用轮盘赌选择算法）"""
        new_population = []
        tournament_size = 2  # 锦标赛规模

        # 锦标赛
        for i in range(0, self.population_size):
            temp = copy.deepcopy(population)  # 临时列表，供锦标赛抽取
            competitor_a = copy.deepcopy(random.choice(temp))  # 随机抽取选手a
            temp.remove(competitor_a)
            competitor_b = copy.deepcopy(random.choice(temp))  # 随机抽取选手b
            temp.remove(competitor_b)

            # 若a支配b
            if ((self.TimeFitness(competitor_a) < self.TimeFitness(competitor_b)) and (
                    self.CostFitness(competitor_a) < self.CostFitness(competitor_b))) or (
                    (round(self.TimeFitness(competitor_a)) == round(self.TimeFitness(competitor_b))) and (
                    self.CostFitness(competitor_a) < self.CostFitness(competitor_b))) or (
                    (self.TimeFitness(competitor_a) < self.TimeFitness(competitor_b)) and (
                    round(self.CostFitness(competitor_a)) == round(self.CostFitness(competitor_b)))):
                new_population.append(competitor_a)
            # 若b支配a
            elif ((self.TimeFitness(competitor_b) < self.TimeFitness(competitor_a)) and (
                    self.CostFitness(competitor_b) < self.CostFitness(competitor_a))) or (
                    (round(self.TimeFitness(competitor_b)) == round(self.TimeFitness(competitor_a))) and (
                    self.CostFitness(competitor_b) < self.CostFitness(competitor_a))) or (
                    (self.TimeFitness(competitor_b) < self.TimeFitness(competitor_a)) and (
                    round(self.CostFitness(competitor_b)) == round(self.CostFitness(competitor_a)))):
                new_population.append(competitor_b)
            # 若互相不支配
            else:
                Fiteness_a = self.TimeFitness(competitor_a) + self.CostFitness(competitor_a)  # a的适应值之和
                Fiteness_b = self.TimeFitness(competitor_b) + self.CostFitness(competitor_b)  # b的适应值之和
                if Fiteness_a < Fiteness_b:
                    new_population.append(competitor_a)
                else:
                    new_population.append(competitor_b)
        return new_population

    def Crossover(self, population):
        """交叉操作"""

        cp = self.crossover_probability  # 交叉概率
        new_population = []  # 初始化交叉完毕的种群
        crossover_population = []  # 初始化需要交叉的种群
        # 根据交叉概率选出需要交叉的个体
        for c in population:
            r = random.random()
            if r <= cp:
                crossover_population.append(c)
            else:
                new_population.append(c)

        # 需保证交叉的个体是偶数,若不是偶数，则删掉需交叉列表的最后一个元素
        if len(crossover_population) % 2 != 0:
            new_population.append(crossover_population[len(crossover_population) - 1])
            del crossover_population[len(crossover_population) - 1]

        # crossover——单点交叉
        for i in range(0, len(crossover_population), 2):
            i_solution = crossover_population[i]
            j_solution = crossover_population[i + 1]
            crossover_position = random.randint(1, self.task_number - 2)  # 随机生成一个交叉位
            left_i = copy.deepcopy(i_solution[0:crossover_position])
            right_i = copy.deepcopy(i_solution[crossover_position:self.task_number])
            left_j = copy.deepcopy(j_solution[0:crossover_position])
            right_j = copy.deepcopy(j_solution[crossover_position:self.task_number])
            # 生成新个体
            new_i = copy.deepcopy(left_i + right_j)
            new_j = copy.deepcopy(left_j + right_i)
            new_population.append(new_i)
            new_population.append(new_j)

            if (i + 1) == (len(crossover_population) - 1):
                break

        return new_population

    def Mutation(self, population):
        """变异操作"""
        mp = self.mutation_probability  # 变异率
        new_population = []  # 初始化变异后的种群

        for c in population:
            r = random.random()
            if r <= mp:
                # mutation——随机选择某个体的一个任务（位置），从对应候选服务集中随机选择某服务替换
                mutation_position = random.randint(0, self.task_number - 1)  # 变异位置
                replaced_time_service = dataset8_10_100.Time_candidates[mutation_position][
                    random.randint(0, len(dataset8_10_100.Time_candidates) - 1)]
                replaced_cost_service = dataset8_10_100.Cost_candidates[mutation_position][
                    random.randint(0, len(dataset8_10_100.Cost_candidates) - 1)]
                replaced_time_service.append(replaced_cost_service)
                replaced_service = copy.deepcopy(replaced_time_service)
                c[mutation_position] = copy.deepcopy(replaced_service)
                new_population.append(c)
            else:
                new_population.append(c)

        # # 调用refine方法，确保不越界
        # new_population = self.refine(new_population, bounds)
        #
        # # 匹配对应任务的候选服务集中的真实服务
        # new_population = self.map_real_service(new_population, candidate_service_list)

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
                if i != j:
                    # 若i被j支配，则淘汰
                    if (Fitness_List[i][0] < Fitness_List[j][0] and Fitness_List[i][1] < Fitness_List[j][1]) \
                            or (Fitness_List[i][0] < Fitness_List[j][0] and Fitness_List[i][1] == Fitness_List[j][1]) \
                            or (Fitness_List[i][0] == Fitness_List[j][0] & Fitness_List[i][1] < Fitness_List[j][1]):
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
