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

    # @abstractmethod
    # def find_solution(self):
    #     pass

    @abstractmethod
    def execute(self):
        pass
