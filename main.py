from genetic_algorithm import GA
from evolutionary_strategy import ES

def main():
    # GA("ackley").execute()
    # GA("rastrigin").execute()
    # GA("schwefel").execute()
    # GA("rosenbrock").execute()

    # ES("ackley", max_iterations=10, number_of_crossovers=350).execute()
    # ES("rastrigin", max_iterations=20, number_of_crossovers=350).execute()
    ES("schwefel", max_iterations=100, number_of_crossovers=350).execute()
    # ES("rosenbrock", max_iterations=50, number_of_crossovers=350, mutation_prob=0.05).execute()


if __name__ == "__main__":
    main()