from utils.functions_utils import ACKLEY_A,ACKLEY_B, ACKLEY_C, DIMENSIONS, ACKLEY_BOUND, RASTRIGIN_BOUND, SCHWEFEL_BOUND, ROSENBROCK_BOUND
from functions.benchmark_functions import ackley, rastrigin, schwefel, rosenbrock
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class EA(ABC):
    def __init__(self, function_name, max_iterations, population_size, selected_parents, crossover_prob, mutation_prob, number_of_crossovers, genoma_size):
        self.function_name = function_name
        self.max_iterations = max_iterations
        self.population_size = population_size
        self.selected_parents = selected_parents
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.number_of_crossovers = number_of_crossovers
        self.genoma_size = genoma_size

        self.best_fitness = 0
        self.it_best_fitness_list = []
        self.it_fitness_mean_list = []
        self.total_run_times = 1
        
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
    
    def get_answer(self):
        return 0
    
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
    
    def survival_selection(self, population):
        population.sort(key=lambda individual : self.fitness(individual), reverse=True)
        return population[0:self.population_size]

    def update_data(self, population):
        fitness_list = [self.fitness(individual) for individual in population]
        self.it_fitness_mean_list.append(sum(fitness_list) / len(fitness_list))
        self.it_best_fitness_list.append(self.fitness(population[0]))

    def check_stopping_criteria(self, iteration, epsilon=0.005):
        if abs(self.it_best_fitness_list[-1] - self.best_fitness) < epsilon:
            self.best_fit_count += 1
        else:
            self.best_fit_count = 0
            self.best_fitness = self.it_best_fitness_list[-1]

        if self.best_fit_count >= 100:
            return True
        
        return False

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
        

    @abstractmethod
    def generate_initial_population(self):
        pass

    @abstractmethod
    def parent_selection(self, population):
        pass

    @abstractmethod
    def mutation(self, population):
        pass

    @abstractmethod
    def crossover(self, population):
        pass

    @abstractmethod
    def execute(self):
        pass

