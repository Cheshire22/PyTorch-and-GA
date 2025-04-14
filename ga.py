import time

from deap import base, algorithms
from deap import creator
from deap import tools

import algelitism
from neuralnetwork2 import NetControl

import random
import matplotlib.pyplot as plt
import numpy as np

import gym

env = gym.make('CartPole-v1')

NEURONS_IN_LAYERS = [4, 1, 1]  # распределение числа нейронов по слоям (первое значение - число входов)
network = NetControl(*NEURONS_IN_LAYERS)

LENGTH_CHROM = NetControl.getTotalWeights(*NEURONS_IN_LAYERS)  # длина хромосомы, подлежащей оптимизации
LOW = -1.0
UP = 1.0
ETA = 20

# константы генетического алгоритма
POPULATION_SIZE = 80  # количество индивидуумов в популяции
P_CROSSOVER = 0.9  # вероятность скрещивания
P_MUTATION = 0.1  # вероятность мутации индивидуума
MAX_GENERATIONS = 50  # максимальное количество поколений
HALL_OF_FAME_SIZE = 3

hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("randomWeight", random.uniform, -1.0, 1.0)
toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.randomWeight, LENGTH_CHROM)
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

population = toolbox.populationCreator(n=POPULATION_SIZE)


def getScore(individual):
    network.set_weights(individual)

    # Получаем observation из кортежа
    observation, _ = env.reset()  # Распаковываем кортеж

    totalReward = 0
    for _ in range(500):  # Максимум 500 шагов
        # Преобразуем observation в numpy array перед reshape
        obs_array = np.array(observation, dtype=np.float32)
        action = int(network.predict(obs_array.reshape(1, -1)))

        observation, reward, terminated, truncated, _ = env.step(action)
        totalReward += reward

        if terminated or truncated:
            break

    return totalReward,


toolbox.register("evaluate", getScore)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=LOW, up=UP, eta=ETA)
toolbox.register("mutate", tools.mutPolynomialBounded, low=LOW, up=UP, eta=ETA, indpb=1.0 / LENGTH_CHROM)

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("max", np.max)
stats.register("avg", np.mean)

population, logbook = algelitism.eaSimpleElitism(population, toolbox,
                                                 cxpb=P_CROSSOVER,
                                                 mutpb=P_MUTATION,
                                                 ngen=MAX_GENERATIONS,
                                                 halloffame=hof,
                                                 stats=stats,
                                                 verbose=True)

maxFitnessValues, meanFitnessValues = logbook.select("max", "avg")

best = hof.items[0]
print(best)

plt.plot(maxFitnessValues, color='red')
plt.plot(meanFitnessValues, color='green')
plt.xlabel('Поколение')
plt.ylabel('Макс/средняя приспособленность')
plt.title('Зависимость максимальной и средней приспособленности от поколения')
plt.show()

observation, _ = env.reset()  # Распаковываем кортеж при сбросе среды
action = int(network.predict(observation.reshape(1, -1)))

while True:
    env.render()

    # Преобразуем observation в numpy array
    obs_array = np.array(observation, dtype=np.float32)
    action = int(network.predict(obs_array.reshape(1, -1)))

    observation, reward, terminated, truncated, _ = env.step(action)

    if terminated or truncated:
        break

    # time.sleep(0.03)
    print(action)
