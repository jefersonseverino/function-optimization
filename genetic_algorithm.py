import random
from functions.benchmark_functions import ackley, rastrigin, schwefel, rosenbrock
from utils.functions_utils import ACKLEY_A,ACKLEY_B, ACKLEY_C, DIMENSIONS, ACKLEY_BOUND, RASTRIGIN_BOUND, SCHWEFEL_BOUND, ROSENBROCK_BOUND
import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
class GA():

    def __init__(self, function_name):
        ## TODO: define better parameters

        self.mutation_prob = 0.05
        self.max_iterations = 10000
        self.crossover_prob = 0.9
        self.population_size = 1000 # Aumentei --> fitness converge mais rápido
        self.selected_parents = 2
        self.number_of_crossovers = self.population_size // 2
        self.function_name = function_name
        self.genoma_size = DIMENSIONS
        self.best_fitness = 0
        self.best_fit_count = 0
        self.it_best_fitness_list = []
        self.it_fitness_mean_list = []
        self.total_run_times = 3

    def get_boundaries(self):

        if self.function_name == "ackley":
            boundary = ACKLEY_BOUND
        elif self.function_name == "rastrigin":
            boundary = RASTRIGIN_BOUND
        elif self.function_name == "schwefel":
            boundary = SCHWEFEL_BOUND
        elif self.function_name == "rosenbrock":
            boundary = ROSENBROCK_BOUND
        
        return boundary

    def generate_initial_population(self):
        population = []
        boundary = self.get_boundaries()

        for i in range(self.population_size):
            individual = [random.uniform(-boundary, boundary) for _ in range(100)]
            population.append(individual)
        
        return population
    
    def fitness(self, individual):
        fitness = 0

        if self.function_name == "ackley":
            fitness = ackley(ACKLEY_A, ACKLEY_B, ACKLEY_C, DIMENSIONS, individual)
        elif self.function_name == "rastrigin":
            fitness = rastrigin(DIMENSIONS, individual)
        elif self.function_name == "schwefel":
            fitness = schwefel(DIMENSIONS, individual)
        elif self.function_name == "rosenbrock":
            fitness = rosenbrock(DIMENSIONS, individual)

        return 1 / (1 + fitness)
    
    # def parent_selection(self, population):
    #     # Roulette
    #     fitness_list = [self.fitness(individual) for individual in population]
    #     total_fitness = sum(fitness_list)
    #     probabilities = [(ind_fitness / total_fitness) for ind_fitness in fitness_list]
    #     selected_individuals = random.choices(population, probabilities, k=self.selected_parents)
    #     return selected_individuals

    def parent_selection(self, population):
        # Tournament (mais rápido que roulette)
        selected_individuals = []
        for _ in range(self.selected_parents*3):
            selected_individuals.append(random.choice(population))
        selected_individuals.sort(key=lambda individual : self.fitness(individual), reverse=True)
        return selected_individuals[:self.selected_parents]

    def crossover(self, parents):
        ### TODO : implement other types crossovers
        # Cruzamentos testados:
        # - Cut and crossfill 
        # - Aritmético simples (mistura de 1 gene)
        # - Aritmético com escolha de ponto aleatório (mistura genes a partir de um ponto aleatório)
        # - Aritmético completo (mistura todos os genes)
        # - Todas as formas de mistura de genes foram testadas com alpha = 0.5 e alpha aleatório
        # Todos geraram resultados semelhantes (antes de eu aumentar a população)

        [parent1, parent2] = parents

        child1 = parent1.copy()
        child2 = parent2.copy()

        idx = random.randint(0, self.genoma_size - 1)

        alpha = random.uniform(0,1)
        child1[idx] = (alpha * parent1[idx] + (1 - alpha) * parent2[idx])
        child2[idx] = (alpha * parent2[idx] + (1 - alpha) * parent1[idx])

        return [child1, child2]

    def mutation(self, population):
        ### TODO : implement other types of mutation

        boundary = self.get_boundaries()
        for individual in population:
            for i in range(self.genoma_size):
                if random.random() < self.mutation_prob:
                    individual[i] = random.uniform(-boundary, boundary)
        
        return population

    def survival_selection(self, population):
        population.sort(key=lambda individual : self.fitness(individual), reverse=True)
        return population[0:self.population_size]

    def check_stopping_criteria(self):

        epsilon = 0.0001
        if abs(self.it_best_fitness_list[-1] - self.best_fitness) < epsilon:
            self.best_fit_count += 1
        else:
            self.best_fit_count = 0
            self.best_fitness = self.it_best_fitness_list[0]

        if self.best_fit_count >= 100:
            return True
        
        return False

    def update_data(self, population):
        fitness_list = [self.fitness(individual) for individual in population]
        self.it_fitness_mean_list.append(sum(fitness_list) / len(fitness_list))
        self.it_best_fitness_list.append(self.fitness(population[0]))

    def find_solution(self):
        initial_time = time.time()
        ### TODO
        self.best_fitness = 0
        self.best_fit_count = 0
        self.it_best_fitness_list = []

        population = self.generate_initial_population()
        
        for i in range(self.max_iterations):

            for _ in range(self.number_of_crossovers): 
                parents = self.parent_selection(population)
                children = parents
                if random.random() < self.crossover_prob:
                    children = self.crossover(parents)
                children = self.mutation(children)
                population.extend(children)

            population = self.survival_selection(population)

            print(f"Iteration : {i}, Fitness : {self.fitness(population[0])}" )

            self.update_data(population)
            
            if self.check_stopping_criteria():
                break


        total_time = time.time() - initial_time

        print(f"Best fitness is: {self.it_best_fitness_list[-1]}")
        print(f"Time elapsed: {total_time}")

        return population, i

    def execute(self):
        num_converged_executions = 0
        exec_fit_mean = []
        exec_best_individual = []
        exec_num_iterations = []
        exec_best_fit = []

        for _ in range(self.total_run_times):
            population, num_iterations = self.find_solution()

            exec_best_individual.append(population[0])
            exec_num_iterations.append(num_iterations)
            exec_best_fit.append(self.it_best_fitness_list[-1])
            exec_fit_mean.append(self.it_fitness_mean_list[-1])
            
            num_converged_executions += 1 if (1-self.it_best_fitness_list[-1] < 0.0001) else 0

        print('Número de execuções convergidas: ', num_converged_executions)

        ## Informações sobre todas as execuções
        # Melhor indivíduo por execução
        self.plot_graph(exec_best_fit, 'Melhor indivíduo por execução', 'Execução', 'Fitness', 'bar')

        # Iterações por execução
        self.plot_graph(exec_num_iterations, 'Número de iterações por execução', 'Execução', 'Número de iterações', 'bar')

        # Média do fitness por execução
        self.plot_graph(exec_fit_mean, 'Fitness médio por execução', 'Execução', 'Fitness médio', 'bar')
        
        ## Informações sobre a última execução
        # Melhor indivíduo da última execução
        print('Melhor indivíduo da última execução: ', exec_best_individual[-1])

        # Melhor indivíduo por iteração
        self.plot_graph(self.it_best_fitness_list, 'Melhor indivíduo por iteração', 'Iteração', 'Fitness')

        # Fitness médio por iteração
        self.plot_graph(self.it_fitness_mean_list, 'Fitness médio por iteração', 'Iteração', 'Fitness médio')
        

    def plot_graph(self, y, title, x_label, y_label, type='line'):
        mean = np.mean(y)
        std = np.std(y)

        print(title)
        print('Média: ', mean)
        print('Desvio padrão: ', std)

        sns.set_style("darkgrid")
        plt.figure(figsize=(12, 4))
        if type == 'bar':
            plt.bar(range(1,len(y)+1), y, color='#6141ac')
        else:
            plt.plot(y, color='#6141ac', linewidth=2)
        plt.axhline(y=mean, color='#0097b2', linestyle='--')
        plt.axhline(y=mean + std, color='#0097b2', linestyle='--')
        plt.axhline(y=mean - std, color='#0097b2', linestyle='--')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        
        plt.show()
