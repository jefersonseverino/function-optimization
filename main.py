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

    # es_ackley = ES("ackley")
    # es_ackley.execute()

    es_rastrigin = ES("rastrigin")
    es_rastrigin.execute()

if __name__ == "__main__":
    main()