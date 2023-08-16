import json
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import multiprocessing

from lunarenvironment import LunarEnvironment

with open("env_config.json", "rb") as config_file:
    env_config = json.load(config_file)

lunar_env = LunarEnvironment(env_config)

lunar_env.reset()

position_array = []
velocity_array = []
forces = []

population_size = 100
mutation_rate = 0.2
num_generations = 10

num_arrays = 146
array_size = 3
min_value = -10
max_value = 10


def calculate_distance(position, target):
    return np.linalg.norm(position - target)


def fitness_function(solution):
    i = 0
    while i < len(solution):
        action = np.array(solution[i: i+3])
        state, reward, terminated, truncated, info = lunar_env.step(action)
        i += 3

    distance_to_target = calculate_distance(lunar_env.spacecraft_position, lunar_env.target_position)
    fitness = 1 / distance_to_target  # Inverse of distance as fitness
    return fitness


def initialize_population(pop_size, num_arrays, array_size, min_value, max_value):
    population = []
    for _ in range(pop_size):
        solution = [random.uniform(min_value, max_value) for _ in range(num_arrays * array_size)]
        population.append(solution)
    return population


def tournament_selection(population, num_parents):
    selected_parents = []
    for _ in range(num_parents):
        tournament_size = 5
        tournament_candidates = random.sample(population, tournament_size)
        best_candidate = max(tournament_candidates, key=fitness_function)
        selected_parents.append(best_candidate)
    return selected_parents


def mutate(solution, mutation_rate, min_value, max_value):
    mutated_solution = solution.copy()
    for i in range(len(mutated_solution)):
        if random.random() < mutation_rate:
            mutated_solution[i] = random.uniform(min_value, max_value)
    return mutated_solution


def store_array(array):
    file_path = 'tensor_file.pt'
    file = open(file_path, 'ab')

    tensor = torch.from_numpy(np.array(array))
    torch.save(tensor, file)


def process_genetic_algorithm(generation, pop_size, parents, mutation_rate, num_arrays, array_size, min_value,
                              max_value):
    offspring = []
    while len(offspring) < pop_size:
        parent1, parent2 = random.sample(parents, 2)
        crossover_point = random.randint(1, num_arrays * array_size - 1)
        child = parent1[:crossover_point] + parent2[crossover_point:]
        child = mutate(child, mutation_rate, min_value, max_value)
        offspring.append(child)

    return offspring


def genetic_algorithm_parallel(pop_size, num_generations, mutation_rate, num_processes=1):
    population = initialize_population(pop_size, num_arrays, array_size, min_value, max_value)

    pool = multiprocessing.Pool(processes=num_processes)

    for generation in range(num_generations):
        parents = tournament_selection(population, pop_size // 2)

        # Distribute the work among processes
        process_args = [(generation, pop_size, parents, mutation_rate, num_arrays, array_size, min_value, max_value) for
                        _ in range(num_processes)]
        offspring_lists = pool.starmap(process_genetic_algorithm, process_args)

        # Combine offspring from different processes
        offspring = [child for sublist in offspring_lists for child in sublist]

        population = offspring

        best_solution = max(population, key=fitness_function)
        best_fitness = fitness_function(best_solution)
        print("Generation: ", generation, "Best Fitness: ", best_fitness)

        store_array(best_solution)

    pool.close()
    pool.join()

    best_solution = max(population, key=fitness_function)
    best_fitness = fitness_function(best_solution)

    return best_solution, best_fitness


best_solution, best_fitness = genetic_algorithm_parallel(population_size, num_generations, mutation_rate)

print("Best Solution:", best_solution)
print("Best Fitness:", best_fitness)



