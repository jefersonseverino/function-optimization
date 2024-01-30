import random
import numpy as np
from functions.benchmark_functions import ackley, rastrigin, schwefel, rosenbrock
from utils.functions_utils import GLOBAL_TAU, LOCAL_TAU, ACKLEY_A,ACKLEY_B, ACKLEY_C, DIMENSIONS, ACKLEY_BOUND, RASTRIGIN_BOUND, SCHWEFEL_BOUND, ROSENBROCK_BOUND
import matplotlib.pyplot as plt
import seaborn as sns

class ES():

    def __init__(self, function_name, max_iterations = 1000, population_size=100, selected_parents = 2, number_of_crossovers = 100,
                 crossover_probability = 0.9, mutation_probability = 0.03, genoma_size = 100):
        self.function_name = function_name
        self.max_iterations = max_iterations
        self.population_size = population_size
        self.selected_parents = selected_parents
        self.number_of_crossovers = number_of_crossovers
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.genoma_size = genoma_size

    def generate_initial_population(self):
        population = []
        boundarie = self.get_boundaries()
        sigma = 1 # maybe change

        for _ in range(self.population_size):
            individual = [random.uniform(-boundarie, boundarie) for _ in range(self.genoma_size)]
            individual.append(sigma)
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

    def parent_selection(self, population):
        # Tournament (mais rápido que roulette)
        parents = []
        for k in range(self.selected_parents):
            parent_1 = random.choice(population)
            parent_2 = random.choice(population)

            parents.append(parent_1)
            parents.append(parent_2)
        
        return parents

    def crossover(self, parent1, parent2):
        # intermediate recombination
        prob = random.uniform(0, 1)
        if prob < self.crossover_probability:
            child1 = []
            child2 = []

            dim = len(parent1) - 1
            for idx in range(dim):
                ratio = random.random()
                child1.append(parent1[idx] + ratio * (parent2[idx] - parent1[idx]))
                ratio = random.random()
                child2.append(parent2[idx] + ratio * (parent1[idx] - parent2[idx]))

            if len(parent1) == dim + 1:
                child1.append(parent1[dim])
                child2.append(parent2[dim])

            children = np.array([child1, child2])
            return children

        return [parent1, parent2]

    def survival_selection(self, population):
        population.sort(key=lambda individual : self.fitness(individual), reverse=True)
        return population[0:self.population_size]

    def population_fitness(self, population):
        fitness = [self.fitness(individual) for individual in population]
        return fitness
    
    def get_best_fitness_and_best_individual(self, population):
        fitness = -1
        best_individual = []
        for individual in population:
            if self.fitness(individual) > fitness:
                fitness = self.fitness(individual)
                best_individual = individual
        
        return fitness, best_individual

    def mutationES(self, individual):
        dim = len(individual) - 1
        normal = np.random.normal(0, 1)
        for i in range(dim):
            r = random.uniform(0, 1)
            individual[dim] = np.exp(GLOBAL_TAU*normal + LOCAL_TAU*np.random.normal(0, 1))
            if r < self.mutation_probability:
                individual[i] = individual[dim] * np.random.normal(0, 1)
        
        return individual
    
    def generate_children(self, parents, population):
        for i in range(0, len(parents), 2):
            children = self.crossover(parents[i], parents[i + 1])
            for child in children:
                mutated_child = self.mutationES(child)
                population.append(mutated_child)
        population = self.survival_selection(population)        
        fitness = [self.fitness(individual) for individual in population]

        return population, fitness

    def evaluate_sigma(self, population, mutation_sucess_rate):
        UPDATE = 0.2
        for individual in population:
            sigma_idx = len(individual) - 1
            change = np.random.uniform(0.8, 1)

            if mutation_sucess_rate > UPDATE:
                individual[sigma_idx] /= change # Exploration
            else:
                individual[sigma_idx] *= change # Explotation

    def mutation_sucess(self, fitness_before, fitness_after, mutations):
        sucess = 0
        for i in range(mutations):
            if fitness_after[i] > fitness_before[i]:
                sucess += 1
        return sucess

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
        population = self.generate_initial_population()
        exec_fitness = []

        for i in range(self.max_iterations):
            print(f"Iteration: {i}")
            parents = []
            for _ in range(self.number_of_crossovers):
                parents.extend(self.parent_selection(population)) # Change to generate more parents

            fitness_before_generation = self.population_fitness(population)
            new_population, new_population_fitness = self.generate_children(parents, population)
            num_of_mutations = len(new_population)
            num_mutation_sucess = self.mutation_sucess(fitness_before_generation, new_population_fitness, num_of_mutations)
            population = new_population

            current_best_fitness, _ = self.get_best_fitness_and_best_individual(population)
            exec_fitness.append(current_best_fitness)

            if i % 5 == 0:
                sucess_rate = num_mutation_sucess / num_of_mutations
                self.evaluate_sigma(population, sucess_rate)

        self.plot_graph(exec_fitness, "Fitness over time", "iteration", "fitness")

        fitness, best_individual = self.get_best_fitness_and_best_individual(population)
        print(f"The best individual found is: {best_individual}")
        print(f"Best result : {fitness}")
        

