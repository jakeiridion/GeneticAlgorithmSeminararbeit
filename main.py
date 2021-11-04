import math
import random
import numpy
from typing import List
import matplotlib.pyplot as plt


Individual = List[int]
Population = List[Individual]


class GeneticAlgorithm:
    def __init__(self):
        self._population_size: int = 10
        self._tournament_size: int = 2
        self._fitness_function = lambda x: (-math.pow(x, 2) / 21) + 3 * x
        self._crossover_probability: float = 0.90
        self._mutation_probability: float = 0.01
        self._number_of_iterations: int = 100

    def create_initial_population(self) -> Population:
        return [list(numpy.random.randint(0, 2, 6)) for _ in range(self._population_size)]

    def tournament_selection(self, population: Population) -> Individual:
        best = self._get_random_individual(population)[:]
        for _ in range(self._tournament_size-1):
            challenger = self._get_random_individual(population)[:]
            if self.calculate_fitness(challenger) > self.calculate_fitness(best):
                best = challenger[:]
        return best

    def calculate_fitness(self, individual: Individual) -> float:
        x = int("".join(str(digit) for digit in individual), 2)
        return self._fitness_function(x)

    def one_point_crossover(self, parent1: Individual, parent2: Individual) -> Population:
        if random.random() < self._crossover_probability:
            crossover_point = random.randint(1, len(parent1)-1)
            partial_p1 = parent1[:crossover_point]
            partial_p2 = parent2[:crossover_point]
            parent1[:crossover_point] = partial_p2
            parent2[:crossover_point] = partial_p1
        return [parent1, parent2]

    def flip_bit_mutation(self, individual: Individual) -> Individual:
        for i in range(len(individual)-1):
            if random.random() < self._mutation_probability:
                individual[i] = 1 - individual[i]
        return individual

    def _get_random_individual(self, population: Population) -> Individual:
        return population[random.randint(0, len(population)-1)]

    def run(self) -> int:
        self._setup_graph()
        population = self.create_initial_population()[:]
        best = self._get_random_individual(population)[:]
        for _ in range(self._number_of_iterations):
            for individual in population:
                self._plot_dot(individual)
                if self.calculate_fitness(individual) > self.calculate_fitness(best):
                    best = individual[:]
            # Tournament Selection:
            population = [self.tournament_selection(population) for _ in range(self._population_size)][:]
            # One Point Crossover:
            crossed_over_population = []
            for _ in range(int(self._population_size/2)):
                crossed_over_population += self.one_point_crossover(self._get_random_individual(population),
                                                                    self._get_random_individual(population))
            # Flip Bit Mutation:
            population = [self.flip_bit_mutation(individual) for individual in crossed_over_population][:]
        self._show_graph()
        return int("".join(str(digit) for digit in best), 2)

    def _setup_graph(self):
        plt.axis([0, 64, 0, 50])
        plt.plot(numpy.arange(64), [genetic_algorithm._fitness_function(x) for x in range(64)])

    def _plot_dot(self, individual: List):
        plt.plot(int("".join(str(digit) for digit in individual), 2), self.calculate_fitness(individual), "ro")

    def _show_graph(self):
        plt.xlabel(f"Number of Iterations: {self._number_of_iterations}")
        plt.savefig('GA_Graph.png', bbox_inches='tight')
        plt.show()


if __name__ == '__main__':
    genetic_algorithm = GeneticAlgorithm()
    print(genetic_algorithm.run())
