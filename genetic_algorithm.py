import random
from utils.functions_utils import DIMENSIONS
import time
from evolutionary_algorithm import EA
class GA(EA):

    def __init__(self, function_name, mutation_prob = 0.05, max_iterations = 10000, crossover_prob = 0.9, population_size = 1000,
                 selected_parents = 2):
        super().__init__(function_name, max_iterations, population_size, selected_parents, crossover_prob, mutation_prob, population_size//2, DIMENSIONS)
        
        self.best_fitness = 0
        self.best_fit_count = 0
        self.it_best_fitness_list = []
        self.it_fitness_mean_list = []
        self.total_run_times = 3


    def generate_initial_population(self):
        population = []
        boundary = self.get_boundaries()

        for i in range(self.population_size):
            individual = [random.uniform(-boundary, boundary) for _ in range(self.genoma_size)]
            population.append(individual)
        
        return population

    def parent_selection(self, population):
        # Tournament (mais rápido que roulette)
        selected_individuals = []
        for _ in range(self.selected_parents*3):
            selected_individuals.append(random.choice(population))
        selected_individuals.sort(key=lambda individual : self.fitness(individual), reverse=True)
        return selected_individuals[:self.selected_parents]

    def crossover(self, parents):
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
        
