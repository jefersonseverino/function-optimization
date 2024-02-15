import random
from utils.functions_utils import DIMENSIONS
from evolutionary_algorithm import EA

'''
    a. Representação das soluções (indivíduos): 
        Lista de números reais de tamanho igual ao número de dimensões
    b. Função de Fitness: 1 / (1 + f(x))
    c. População (tamanho, inicialização): 
        2000, inicialização aleatória
    d. Processo de seleção de pais: 
        Seleção dos 2 melhores indivíduos de 6 aleatórios
    e. Operadores Genéticos (Recombinação e Mutação): 
        Recombinação aritmética e mutação gaussiana
    f. Processo de seleção por sobrevivência: 
        Seleção dos 200 melhores indivíduos
    g. Condições de término do Algoritmo Evolucionário :
        100 iterações sem melhora ou 5000 iterações
'''
class GA(EA):

    def __init__(self, function_name, mutation_prob = 0.05, max_iterations = 5000, crossover_prob = 0.9, population_size = 250,
                 selected_parents = 2):
        super().__init__(function_name, max_iterations, population_size, selected_parents, crossover_prob, mutation_prob, population_size//2, DIMENSIONS)


    def generate_initial_population(self):
        population = []
        boundary = self.get_boundaries()

        for i in range(self.population_size):
            individual = [random.uniform(-boundary, boundary) for _ in range(self.genoma_size)]
            population.append(individual)
        
        return population

    def parent_selection(self, population):
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
        boundary = self.get_boundaries()
        for individual in population:
            for i in range(self.genoma_size):
                if random.random() < self.mutation_prob:
                    individual[i] = random.gauss(0,1)
        
        return population

    def find_solution(self):
        self.best_fitness = 0
        self.best_fit_count = 0
        self.it_best_fitness_list = []
        self.it_fitness_mean_list = []

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

        return population, i
