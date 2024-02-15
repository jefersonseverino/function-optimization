from genetic_algorithm import GA
from evolutionary_strategy import ES

def main():
    GA("ackley").execute()
    GA("rastrigin").execute()
    GA("schwefel").execute()
    GA("rosenbrock").execute()

    ES("ackley").execute()
    ES("rastrigin").execute()
    ES("schwefel").execute()
    ES("rosenbrock").execute()

if __name__ == "__main__":
    main()