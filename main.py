from genetic_algorithm import GA

def main():
    genetic_ackley = GA("ackley")
    genetic_rastrigin = GA("rastrigin")
    genetic_schwefel = GA("schwefel")
    genetic_rosenbrock = GA("rosenbrock")

    genetic_ackley.execute()
    # genetic_rastrigin.execute()
    # genetic_rosenbrock.execute()

if __name__ == "__main__":
    main()