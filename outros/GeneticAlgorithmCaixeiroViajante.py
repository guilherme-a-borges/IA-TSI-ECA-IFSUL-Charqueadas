import random

# 1. Definição do Problema (Matriz de Distâncias)
# Cidades: 0 = A, 1 = B, 2 = C, 3 = D
DISTANCES = [
    [0, 10, 15, 20],  # A
    [10, 0, 35, 25],  # B
    [15, 35, 0, 30],  # C
    [20, 25, 30, 0]   # D
]
CITIES = [0, 1, 2, 3]
CITY_NAMES = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}

# 2. Funções de Avaliação (Fitness)
def calculate_route_distance(route):
    total_dist = 0
    num_cities = len(route)
    for i in range(num_cities):
        current_city = route[i]
        next_city = route[(i + 1) % num_cities] # Retorna para a primeira cidade no fim
        total_dist += DISTANCES[current_city][next_city]
    return total_dist

def get_fitness(route):
    # O objetivo é minimizar a distância, então o fitness é o inverso
    return 1.0 / calculate_route_distance(route)

# 3. Inicialização da População
def create_individual():
    # Uma rota válida é uma permutação aleatória de todas as cidades
    return random.sample(CITIES, len(CITIES))

def create_population(pop_size):
    return [create_individual() for _ in range(pop_size)]

# 4. Seleção por Torneio (Fácil de entender e implementar)
def selection_tournament(population, k=3):
    selected = random.sample(population, k)
    # Seleciona o que tiver o melhor (maior) fitness
    return max(selected, key=get_fitness)

# 5. Crossover Baseado em Ordem (Garante que cidades não se repitam)
def crossover_ordered(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    
    # Cria o filho com posições vazias (None)
    child = [None] * size
    # Copia um segmento do primeiro pai
    child[start:end+1] = parent1[start:end+1]
    
    # Preenche o restante com os elementos do segundo pai preservando a ordem
    p2_remaining = [item for item in parent2 if item not in child]
    
    idx = 0
    for i in range(size):
        if child[i] is None:
            child[i] = p2_remaining[idx]
            idx += 1
            
    return child

# 6. Mutação por Troca (Swap Mutation)
def mutate_swap(individual, mutation_rate=0.1):
    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(len(individual)), 2)
        # Inverte a posição de duas cidades escolhidas ao acaso
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    return individual

# 7. Loop Principal do Algoritmo Genético
def genetic_algorithm(pop_size=10, generations=20, mutation_rate=0.2):
    population = create_population(pop_size)
    
    print("--- POPULAÇÃO INICIAL ---")
    for ind in population:
        print(f"Rota: {[CITY_NAMES[c] for c in ind]} | Distância: {calculate_route_distance(ind)}")

    for generation in range(generations):
        new_population = []
        
        # Elitismo: Mantém o melhor da geração anterior direto na nova
        best_of_gen = max(population, key=get_fitness)
        new_population.append(best_of_gen)
        
        # Gera o restante da nova população
        while len(new_population) < pop_size:
            p1 = selection_tournament(population)
            p2 = selection_tournament(population)
            
            child = crossover_ordered(p1, p2)
            child = mutate_swap(child, mutation_rate)
            
            new_population.append(child)
            
        population = new_population

    # Resultado Final
    best_route = max(population, key=get_fitness)
    best_distance = calculate_route_distance(best_route)
    readable_route = [CITY_NAMES[c] for c in best_route] + [CITY_NAMES[best_route[0]]]
    
    print("\n--- RESULTADO FINAL DO AG ---")
    print(f"Melhor Rota Encontrada: {' -> '.join(readable_route)}")
    print(f"Distância Mínima Obtida: {best_distance}")

# Executar o Algoritmo
if __name__ == "__main__":
    genetic_algorithm()