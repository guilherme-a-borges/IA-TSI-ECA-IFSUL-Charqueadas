import random

# -----------------------------
# Genetic Algorithm Example
# Maximize f(x) = 15x - x²
# x is an integer in [0, 31]
# Chromosome: 5-bit binary string
# -----------------------------

CHROMOSOME_SIZE = 5
POPULATION_SIZE = 6
GENERATIONS = 20
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.05
ELITISM = True


def binary_to_int(chromosome):
    return int(chromosome, 2)


def fitness(chromosome):
    x = binary_to_int(chromosome)
    return 15 * x - x**2


def create_individual():
    return "".join(random.choice("01") for _ in range(CHROMOSOME_SIZE))


def create_population():
    return [create_individual() for _ in range(POPULATION_SIZE)]


def roulette_selection(population):
    fitness_values = [max(fitness(ind), 0) for ind in population]
    total_fitness = sum(fitness_values)

    if total_fitness == 0:
        return random.choice(population)

    pick = random.uniform(0, total_fitness)
    current = 0

    for individual, fit in zip(population, fitness_values):
        current += fit
        if current >= pick:
            return individual


def crossover(parent1, parent2):
    if random.random() > CROSSOVER_RATE:
        return parent1, parent2

    cut = random.randint(1, CHROMOSOME_SIZE - 1)

    child1 = parent1[:cut] + parent2[cut:]
    child2 = parent2[:cut] + parent1[cut:]

    return child1, child2


def mutate(chromosome):
    new_chromosome = ""

    for gene in chromosome:
        if random.random() < MUTATION_RATE:
            new_chromosome += "1" if gene == "0" else "0"
        else:
            new_chromosome += gene

    return new_chromosome


def print_population(population, generation):
    print(f"\nGeneration {generation}")
    print("-" * 40)
    print("Individual | x  | Fitness")
    print("-" * 40)

    for individual in population:
        x = binary_to_int(individual)
        fit = fitness(individual)
        print(f"{individual:^10} | {x:^2} | {fit:^7}")

    best = max(population, key=fitness)
    print(f"\nBest: {best} | x = {binary_to_int(best)} | fitness = {fitness(best)}")


def genetic_algorithm():
    population = create_population()

    for generation in range(GENERATIONS + 1):
        print_population(population, generation)

        new_population = []

        if ELITISM:
            best = max(population, key=fitness)
            new_population.append(best)

        while len(new_population) < POPULATION_SIZE:
            parent1 = roulette_selection(population)
            parent2 = roulette_selection(population)

            child1, child2 = crossover(parent1, parent2)

            child1 = mutate(child1)
            child2 = mutate(child2)

            new_population.append(child1)

            if len(new_population) < POPULATION_SIZE:
                new_population.append(child2)

        population = new_population

    best = max(population, key=fitness)

    print("\nFinal Result")
    print("=" * 40)
    print(f"Best chromosome: {best}")
    print(f"x = {binary_to_int(best)}")
    print(f"f(x) = {fitness(best)}")


genetic_algorithm()