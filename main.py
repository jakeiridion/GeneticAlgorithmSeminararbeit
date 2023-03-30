import math
import random
import numpy
from typing import List
import matplotlib.pyplot as plt


Chromosome = List[int]
Population = List[Chromosome]


class GeneticAlgorithm:
    def __init__(self):
        self._population_size: int = 8
        self._tournament_size: int = 2
        self._fitness_function = lambda x: (-math.pow(x, 2) / 21) + 3 * x
        self._crossover_probability: float = 0.9
        self._mutation_probability: float = 0.01
        self._number_of_iterations: int = 50

    def create_initial_population(self) -> Population:
        return [list(numpy.random.randint(0, 2, 6)) for _ in range(self._population_size)]

    def tournament_selection(self, population: Population) -> Chromosome:
        best = self._get_random_chromosome(population)
        for _ in range(self._tournament_size-1):
            challenger = self._get_random_chromosome(population)[:]
            if self.calculate_fitness(challenger) > self.calculate_fitness(best):
                best = challenger[:]
        return best

    def _get_random_chromosome(self, population: Population) -> Chromosome:
        return population[random.randint(0, len(population)-1)]

    def calculate_fitness(self, chromosome: Chromosome) -> float:
        x = self._get_individual(chromosome)
        return self._fitness_function(x)

    def _get_individual(self, chromosome: Chromosome) -> int:
        return int("".join(str(digit) for digit in chromosome), 2)

    def one_point_crossover(self, parent1: Chromosome, parent2: Chromosome) -> Population:
        if random.random() < self._crossover_probability:
            crossover_point = random.randint(1, len(parent1)-1)
            partial_p1 = parent1[:crossover_point]
            partial_p2 = parent2[:crossover_point]
            parent1[:crossover_point] = partial_p2
            parent2[:crossover_point] = partial_p1
        return [parent1, parent2]

    def flip_bit_mutation(self, chromosome: Chromosome) -> Chromosome:
        for i in range(len(chromosome)-1):
            if random.random() < self._mutation_probability:
                chromosome[i] = 1 - chromosome[i]
        return chromosome

    def run(self) -> int:
        self._setup_graph()
        population = self.create_initial_population()
        best = self._get_random_chromosome(population)
        for _ in range(self._number_of_iterations):
            for chromosome in population:
                self._plot_dot(chromosome)
                if self.calculate_fitness(chromosome) > self.calculate_fitness(best):
                    best = chromosome[:]
            # Tournament Selection:
            population = [self.tournament_selection(population) for _ in range(self._population_size)]
            # One Point Crossover:
            crossed_over_population = []
            for _ in range(int(self._population_size/2)):
                crossed_over_population += self.one_point_crossover(self._get_random_chromosome(population),
                                                                    self._get_random_chromosome(population))
            # Flip Bit Mutation:
            population = [self.flip_bit_mutation(chromosome) for chromosome in crossed_over_population]
        self._show_graph()
        return self._get_individual(best)

    def _setup_graph(self):
        plt.axis([0, 64, 0, 50])
        plt.plot(numpy.arange(64), [self._fitness_function(x) for x in range(64)])

    def _plot_dot(self, chromosome: Chromosome):
        plt.plot(self._get_individual(chromosome), self.calculate_fitness(chromosome), "ro")

    def _show_graph(self):
        plt.xlabel(f"Number of Iterations: {self._number_of_iterations}")
        plt.savefig('GA_Graph.png', bbox_inches='tight')
        plt.show()


if __name__ == '__main__':
    genetic_algorithm = GeneticAlgorithm()
    print(genetic_algorithm.run())
