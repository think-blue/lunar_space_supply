import json
import numpy as np
import random
import torch
from lunarenvironment import LunarEnvironment
from deap import base, creator, tools, algorithms


def store_array_deap(array):
    file_path = 'tensor_file2.pt'
    file = open(file_path, 'ab')

    tensor = torch.from_numpy(np.array(array))
    torch.save(tensor, file)


with open("env_config.json", "rb") as config_file:
    env_config = json.load(config_file)

lunar_env = LunarEnvironment(env_config)
lunar_env.reset()

population_size = 100
mutation_rate = 0.2
num_generations = 10000

num_arrays = 150
array_size = 3
min_value = -10
max_value = 10

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, min_value, max_value)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=num_arrays * array_size)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def calculate_distance(position, target):
    return np.linalg.norm(position - target)


def evaluate(individual):
    i = 0
    while i < len(individual):
        action = np.array(individual[i: i + 3])
        state, reward, terminated, truncated, info = lunar_env.step(action)
        i += 3

    distance_to_target = calculate_distance(lunar_env.spacecraft_position, lunar_env.target_position)
    fitness = -distance_to_target  # Inverse of distance as fitness
    return fitness,


def tournament_selection(population, num_parents):
    selected_parents = []
    for _ in range(num_parents):
        tournament_size = 5
        tournament_candidates = random.sample(population, tournament_size)
        best_candidate = max(tournament_candidates, key=lambda x: x.fitness.values[0])
        selected_parents.append(best_candidate)
    return selected_parents


def mutate(individual):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = random.uniform(min_value, max_value)
    return individual,


toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", mutate)
toolbox.register("select", tournament_selection)
toolbox.register("evaluate", evaluate)


def main():
    population = toolbox.population(n=population_size)
    algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=num_generations, verbose=True)

    best_individual = tools.selBest(population, k=1)[0]
    best_fitness = best_individual.fitness.values[0]

    # print("Best Solution:", best_individual)
    print("Best Fitness:", best_fitness)

    store_array_deap(best_individual)


if __name__ == "__main__":
    main()
