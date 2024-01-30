from genetic_algorithm import GA
from evolutionary_strategy import ES

def main():
    genetic_ackley = GA("ackley")
    genetic_rastrigin = GA("rastrigin")
    genetic_schwefel = GA("schwefel")
    genetic_rosenbrock = GA("rosenbrock")

    # genetic_ackley.execute()
    # genetic_rastrigin.execute()
    # genetic_rosenbrock.execute()

    # es_ackley = ES("ackley", max_iterations=10, number_of_crossovers=350)
    # es_ackley.execute()

    # es_rastrigin = ES("rastrigin", max_iterations=20, number_of_crossovers=350)
    # es_rastrigin.execute()

    es_schwefel = ES("schwefel", max_iterations=100, number_of_crossovers=350)
    es_schwefel.execute()

    # es_rosenbrock = ES("rosenbrock", max_iterations=50, number_of_crossovers=350, mutation_probability=0.05)
    # es_rosenbrock.execute()

if __name__ == "__main__":
    main()