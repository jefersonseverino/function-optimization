import random
from functions.benchmark_functions import ackley, rastrigin, schwefel, rosenbrock
from utils.functions_utils import ACKLEY_A,ACKLEY_B, ACKLEY_C, DIMENSIONS, ACKLEY_BOUND, RASTRIGIN_BOUND, SCHWEFEL_BOUND, ROSENBROCK_BOUND

class GA():

    def __init__(self, function_name):
        ## TODO: define better parameters

        self.mutation_prob = 0.05
        self.max_iterations = 10000
        self.crossover_prob = 0.9
        self.population_size = 100
        self.selected_parents = 2
        self.number_of_crossovers = 20
        self.function_name = function_name
        self.genoma_size = DIMENSIONS
    
    def get_boundaries(self):

        if self.function_name == "ackley":
            boundarie = ACKLEY_BOUND
        elif self.function_name == "rastrigin":
            boundarie = RASTRIGIN_BOUND
        elif self.function_name == "schwefel":
            boundarie = SCHWEFEL_BOUND
        elif self.function_name == "rosenbrock":
            boundarie = ROSENBROCK_BOUND
        
        return boundarie

    def generate_initial_population(self):
        population = []
        boundarie = self.get_boundaries()

        for i in range(self.population_size):
            individual = [random.uniform(-boundarie, boundarie) for _ in range(100)]
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
    
    def parent_selection(self, population):
        # Roulette
        fitness_list = [self.fitness(individual) for individual in population]
        total_fitness = sum(fitness_list)
        probabilities = [(ind_fitness / total_fitness) for ind_fitness in fitness_list]
        selected_individuals = random.choices(population, probabilities, k=self.selected_parents)
        return selected_individuals

    def crossover(self, parents):
        ### TODO : implement other types crossovers

        children = []
        [parent1, parent2] = parents
        crossIndex = random.randint(0, self.genoma_size - 1)
        children.append(parent1[:crossIndex] + parent2[crossIndex:])
        children.append(parent2[:crossIndex] + parent1[crossIndex:])
        return children

    def mutation(self, population):
        ### TODO : implement other types of mutation

        boundarie = self.get_boundaries()
        for individual in population:
            for i in range(self.genoma_size):
                if random.random() < self.mutation_prob:
                    individual[i] = random.uniform(-boundarie, boundarie)
        
        return population

    def survival_selection(self, population):
        population.sort(key=lambda individual : self.fitness(individual), reverse=True)
        return population[0:self.population_size]

    def execute(self):
        ### TODO
        population = self.generate_initial_population()

        for i in range(self.max_iterations):
            print(f"Iteration : {i}" )

            for _ in range(self.number_of_crossovers): 
                parents = self.parent_selection(population)
                children = parents
                if random.random() < self.crossover_prob:
                    children = self.crossover(parents)
                children = self.mutation(children)
                population.extend(children)

            population = self.survival_selection(population)


        final_fitness = [self.fitness(individual) for individual in population]
        final_fitness.sort(reverse=True)

        print(f"Best fitness is: {final_fitness[0]}")
