import random
from typing import List, Dict, Optional
import numpy as np
import math
import matplotlib.pyplot as plt

max_pop1 = 30
max_gen1 = 50
prev_gen1 = 0.3*max_gen1
# max_gen2 = 100
# max_gen3 = 150

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return max(0.0, x)

class TempPopulation:

    def __init__(self, population):
        self.population = population

    def crossover_a(self):
        """Two point crossover, with children also having only 4 bits."""
        next_pop = []
        parent_pop = self.population
        print('here3')


        while len(next_pop) < max_pop1:
            p1 = random.choice(parent_pop)
            p2 = random.choice(parent_pop)

            c1 = [p1.phenotype[0], p2.phenotype[1], p1.phenotype[2], \
                 p2.phenotype[3]]
            c2 = [p2.phenotype[0], p1.phenotype[1], p2.phenotype[2], \
                 p1.phenotype[3]]
            next_pop.append(Individual(c1))
            next_pop.append(Individual(c2))

        if len(next_pop) > max_pop1:
            difference = len(next_pop) - max_pop1
            i = 0
            while i < difference:
                remove = random.choice(next_pop)
                # what if best individual is removed?
                next_pop.remove(remove)
                i += 1
        return TempPopulation(next_pop)


        # print(parent_pop)
        #or v2, need to allow for user based?
        # indices = []
        # for k in range(0, len(self.population)):
        #     if k%2 == 0 and k < (len(self.population) - 1):
        #         indices.append(k)
        #
        # for i in indices: #goes over 0, 2, 4 etc. indices
        #     #parents are: i, i + 1
        #     # print(self.population[i])
        #     p1 = (parent_pop[i]).phenotype
        #     p2 = (self.population[i+1]).phenotype
        #     c1, c2 = [], []
        #     #initialise c1 and c2 as empty lists
        #     c1 = [p1[0], p2[1], p1[2], p2[3]]
        #     c2 = [p2[0], p1[1], p2[2], p1[3]]
        #
        #     # for j in p1[0::2]:
        #     #     #first genetic bits, then next loop crosses learned
        #     #     c1.extend([p1[j], p2[j+1]])
        #     #     c2.extend([p2[j], p1[j+1]])
        #
        #     next_pop.append(c1)
        #     next_pop.append(c2)
        #     # children are only lists of phenotype bits, not yet Individual objects
        #
        # return TempPopulation(next_pop)

    # def crossover_b(self) -> List[List]:
    #     """Two point crossover, with children having extension in bits."""
    #     next_pop = []
    #     parent_pop = self.create_next_gen_v1()
    #     #or v2, need to allow for user based?
    #     for i in parent_pop[::2]: #goes over 0, 2, 4 etc. indices
    #         #parents are: i, i + 1
    #         p1 = parent_pop[i].phenotype
    #         p2 = parent_pop[i+1].phenotype
    #         c1, c2 = [], []
    #
    #         c1.extend([p1[0], p2[1]]) #genetic
    #         c1.extend(p1[2:])
    #         c1.extend(p2[2:])
    #         c2.extend([p2[0], p1[1]]) #genetic
    #         c2.extend(p1[2:])
    #         c2.extend(p2[2:])
    #
    #         next_pop.append(c1)
    #         next_pop.append(c2)
    #
    #     return next_pop


    def mutation_v1(self, rate=0.002):
        """Mutation for next generation using swap between benefit or cost bits, genetic or learned.
        Rate set to default of 0.2% or 2 in 1000."""
        # next_pop = self.crossover_a()
        # next_pop = self.crossover_b()
        print('here4')
        #TODO: more randomised mutation for crossover b
        for i in self.population:
            # print(i)
            temp = random.randint(1, 1001)
            if temp <= 2:
                # conduct mutation
                temp_2 = random.random()
                # if value below 0.5 then swap genetic bits, else swap learned bits
                if temp_2 < 0.5:
                    i.phenotype[0], i.phenotype[1] = i.phenotype[1], i.phenotype[0]
                else:
                    i.phenotype[2], i.phenotype[3] = i.phenotype[3], i.phenotype[2]

        # now final version of next_pop has been created
        # creating Individual objects now
        final_next_pop = []
        for i in self.population:
            final_next_pop.append(Individual(i))

        next_gen = NewPopulation(final_next_pop)
        return next_gen

class Generation:
    """Represents generations of the experiment.
    Attributes:
        gens: List of lists: [gen_number: int, Population, [best Individual, fitness of that individual]]
        optima: List of Lists [gen_num, [best individuals from each generation, fitness]]
        gen_number: int representing current/present gen"""


    def __init__(self) -> None:
        self.gens = []
        #initialise as empty dictionary and then add first population from class FirstPopulation
        self.optima = []
        self.gen_number = -1

    def update(self):
        """Updates Generation object and its attributes."""
        self.gen_number = self.gens[-1][0]
        current_pop = self.gens[-1][1]
        self.optima.append([self.gen_number, [current_pop.best, current_pop.best.fitness]])


test_1_gen = Generation()


class NewPopulation:
    """Represents second generation to last generation of populations.
    Attributes:
        individuals: list of Individual objects of the population
        best: Dictionary that stores best individual,
                    key: Individual, value: fitness
        """

    def __init__(self, population):
        """
        population: list of Individual objects of this generation's population
        """
        self.population = population
        m = 0
        for i in range(0, len(test_1_gen.gens)):
            m = test_1_gen.gens[i][0]
        if m == 0: #first generation
            test_1_gen.gens.append(
                [m, self, [self.best, self.best.fitness]])
        else:
            test_1_gen.gens.append(
                [m + 1, self, [self.best, self.best.fitness]])

        self.best = Individual([])
        test_1_gen.update()


    def calculate_pop_fit(self) -> List:
        """Calculates fitness of each individual, and returns ranked fitness values from highest to lowest
        """
        # print('here1')
        fit_list = []
        for i in self.population:
            i.calculate_fitness()
            fit_list.append(i.fitness)

        fit_list.sort()
        fit_list = fit_list[::-1]
        # print(fit_list)
        best_fit_value = fit_list[0]
        for i in self.population:
            if i.fitness == best_fit_value:
                self.best = i
        # print(fit_list)
        ranked_list_individuals = {}
        # {rank: individual}
        for i in self.population:
            for j in range(0, len(fit_list)):
                if i.fitness == fit_list[j]:
                    ranked_list_individuals[j] = i

        # if two individuals with same best fitness, only one is picked
        # print('here2')
        print(self.best.fitness)
        return ranked_list_individuals

    def create_next_gen_v1(self):
        """Creates population for next generation with same size as previous population.
        *Version 1: Stochastic Universal Sampling (SUS) based on fitness -> REDUCING POP SIZE???*
        Version 2: elitism
        Version 3: generational mixing
        Future: selfing??"""
        #mean of fitness of population

        # print("reached here1")
        # convergence check
        if test_1_gen.gen_number == max_gen1:
            return "FINAL GENERATION REACHED"
        else:
            # check best of last 20% or 30% of the generations
            # print("reached here2")
            if test_1_gen.gen_number >= 0.3*max_gen1: #start checking only after completion of 30% of all gens
                c_gen = test_1_gen.gen_number
                start = c_gen - prev_gen1

                counter = True
                for k in range(int(start), c_gen): #ERROR OF ONE since first gen has gen_number 0????
                    if self.best.fitness != test_1_gen.optima[k][1][1]:
                        counter = False
                        break

                print("reach here3")
                if counter: #same as last 30% gens
                    return "FINAL GENERATION REACHED"
            else:
                temp_sum = 0
                # TODO: call calculate pop fit to get ranked list
                ranked = self.calculate_pop_fit()

                for i in ranked:
                    temp_sum += ranked[i].fitness
                n = len(self.population)
                pop_fit_mean = temp_sum / n
                # print(pop_fit_mean)
                alpha = random.uniform(0, 1)
                i_sum = self.population[0].fitness #confirmed to be int

                # fitness of the first individual
                j = 0
                delta = alpha * pop_fit_mean
                next_pop = []
                print("reach here4")
                # represents the next generation
                for d in self.population:
                    while j < n:
                        if delta < i_sum:
                            next_pop.append(d)
                            # selecting the jth individual
                            delta += i_sum

                        else:
                            j += 1
                            i_sum += d.fitness

                random.shuffle(next_pop)

                print("reach here5")
                # to randomise order of list for parent crossover
                # print(len(next_pop))
                final = TempPopulation(next_pop)
                return final

    # def create_next_gen_v1a(self):
    #     """
    #     Selection of parents for crossover for next generation.
    #     """
    #     if test_1_gen.gen_number == max_gen1:
    #         return "FINAL GENERATION REACHED"
    #     else:
    #         # check best of last 20% or 30% of the generations
    #         # print("reached here2")
    #         if test_1_gen.gen_number >= 0.3*max_gen1: #start checking only after completion of 30% of all gens
    #             c_gen = test_1_gen.gen_number
    #             start = c_gen - prev_gen1
    #
    #             counter = True
    #             for k in range(int(start), c_gen): #ERROR OF ONE since first gen has gen_number 0????
    #                 if self.best.fitness != test_1_gen.optima[k][1][1]:
    #                     counter = False
    #                     break
    #
    #                 if counter:  # same as last 30% gens
    #                     return "FINAL GENERATION REACHED"
    #         else:
    #             ranked = self.calculate_pop_fit()
    #             n = len(ranked)
    #             prob_list = []
    #             for i in ranked:
    #                 x = i/[n*(n-1)]
    #                 prob_list.append(x)
    #             v = 1/(n - 2.001)
    #             new_pop = []
    #
    #             k = 0
    #             while k <= n:
    #                 alpha = random.uniform(0, v)
    #                 if prob_list[k] <= alpha:
    #                     new_pop.append(ranked[k])
    #                 k += 1
    #
    #
    #             return new_pop



    def create_next_gen_v2(self, elitism_rate=0.02):
        """Creates population for next generation with same size as previous population.
        Version 1: Stochastic Universal Sampling (SUS) based on fitness
        *Version 2: elitism, set to default as 2%*
        Version 3: generational mixing
        Future: selfing??"""
        if test_1_gen.gen_number == max_gen1:
            return "FINAL GENERATION REACHED"
        else:
            # check best of last 20% or 30% of the generations
            if test_1_gen.gen_number >= 0.3*max_gen1: #start checking only after completion of 30% of all gens
                c_gen = test_1_gen.gen_number
                start = c_gen - prev_gen1

                counter = True
                for i in range(start, c_gen): #ERROR OF ONE since first gen has gen_number 0????
                    if self.best.fitness != test_1_gen.optima[i][1][1]:
                        counter = False
                        break


                if counter: #same as last 30% gens
                    return "FINAL GENERATION REACHED"
            else:
                # elitism rate which needs to be an integral value
                n = len(self.population)
                final_elite_number = math.ceil(
                    n * elitism_rate)  # to ensure that it is not 0
                size_for_next_gen = n - final_elite_number

                # mean of fitness of population
                temp_sum = 0
                for i in self.population:
                    temp_sum += i.fitness

                pop_fit_mean = temp_sum / n  # should it be n or size for next gen?
                alpha = random.uniform(0, 1)
                i_sum = self.population[0].fitness
                # fitness of the first individual
                j = 0
                delta = alpha * pop_fit_mean
                next_pop = []
                # represents the next generation
                while j < size_for_next_gen:
                    if delta < i_sum:
                        next_pop.append(self.population[j])
                        # selecting the jth individual
                        delta += i_sum

                    else:
                        j += 1
                        i_sum += self.population[j].fitness

                previous_pop_ranked = self.population.calculate_pop_fit()[
                                      0:final_elite_number]
                next_pop.extend(previous_pop_ranked)
                random.shuffle(next_pop)
                # to randomise order of list for parent crossover

                return TempPopulation(next_pop)



class Individual:
    """
    Attributes:
        phenotype: List[float] first two are genetic, last two are learned
        fitness: float representing fitness of individual
    """

    def __init__(self, lst: List):
        # initialises all fitness values as 0 till calculated
        self.phenotype = lst
        self.fitness = 0

    def __getitem__(self, i):
        length = len(self.phenotype)
        if i < 0:
            i += length
        if 0 <= i < length:
            return self.phenotype[i]
        raise IndexError('Index out of range: {}'.format(i))

    def calculate_fitness(self):
        # phenotype = _ _ _ _
        # 0, 2 -> benefit, 1, 3 -> cost, 0, 1 -> genetic, 2, 3 -> learned
        # choose relu as cost function because only cost above 0 is counted
        # which is why cost is subtracted
        # print(self.phenotype)
        benefit = [self.phenotype[0], self.phenotype[2]]
        cost = [self.phenotype[1], self.phenotype[3]]

        benefit_func = (sigmoid(benefit[0]) + sigmoid(benefit[1]))/2
        cost_func = (relu(cost[0]) + relu(cost[1]))/2

        self.fitness = benefit_func - cost_func

# class FirstPopulation:
#     # Attributes: population, size
#     # selection process
#
#     def __init__(self, size):
#         # Set up list containing population with real values as bits
#         # Each individual has 4 bits, 2 genetic between -10 and 10, 2 learned ->
#         # between 0 and 1
#         self.population = []
#         self.parent_gen_size = size
#         for i in range(size):
#             self.population.append(Individual([random.uniform(-10, 10),
#                                              random.uniform(-10, 10),
#                                              random.random(), random.random()]))
#         self.fitness = {}
#         # initialise as empty dictionary
#         if test_1_gen.gens == {}:
#             test_1_gen.gens[0] = self.population
#
#         # class only used for first pop, so below not needed????
#         # else:
#         #     gen_num = max(test_1_gen.gens.keys()) + 1
#         #     test_1_gen.gens[gen_num] = self.population


def create_first_pop(size):
    """Creates first population and returns a list of Individual objects."""
    pop = []
    for i in range(size):
        pop.append(Individual([random.uniform(-10, 10),
                                            random.uniform(-10, 10),
                                            random.random(), random.random()]))
    final = NewPopulation(pop)
    # test_1_gen.gens.append([0, final, [final.best, final.best.fitness]])
    # test_1_gen.update()
    return final


"""gens: List of lists: [gen_number: int, Population, [best Individual, fitness of that individual]]
        optima: List of Lists [gen_num, [best individuals from each generation, fitness]]
        gen_number: int representing current/present gen"""

pop1 = create_first_pop(max_pop1) #NewPopulation object
# print(pop1.population)
i = 0
best_ind_fitness = []
generations = []
while i < max_gen1:
    temp_pop1 = pop1.create_next_gen_v1()  # TempPop object
    # print(temp_pop1.population)
    if type(temp_pop1) == str:
        print("end")
        break
    # print(temp_pop1.population)
    temp_pop2 = temp_pop1.crossover_a()
    pop1 = temp_pop2.mutation_v1()
    # print(test_1_gen.optima[i][1][1])
    best_ind_fitness.append(test_1_gen.optima[i][1][1])
    generations.append(i)
    i += 1
    print(test_1_gen.gen_number)


plt.plot(generations, best_ind_fitness)
plt.show()


def create_next_gen_v1(self):
    """Creates population for next generation with same size as previous population.
    *Version 1: Stochastic Universal Sampling (SUS) based on fitness*
    Version 2: elitism
    Version 3: generational mixing
    Future: selfing??"""
    # mean of fitness of population

    # print("reached here1")
    # convergence check
    if test_1_gen.gen_number == max_gen1:
        return "FINAL GENERATION REACHED."
    else:
        # check best of last 20% or 30% of the generations
        # print("reached here2")
        if test_1_gen.gen_number >= 0.3 * max_gen1:
            # start checking only after completion of 30% of all gens
            c_gen = test_1_gen.gen_number
            start = c_gen - prev_gen1

            counter = True
            for k in range(int(start),
                           c_gen):  # ERROR OF ONE since first gen has gen_number 0????
                if self.best.fitness != test_1_gen.optima[k][1][1]:
                    # only stops if best individual fitness value converges
                    # not avg fitness of pop
                    counter = False
                    break

            # print("reach here3")
            if counter:  # same as last 30% gens
                return "POPULATION CONVERGED."
        else:
            # if population has not converged or reached final gen
            # selection procedure for next gen:
            temp_sum = 0
            # TODO: call calculate pop fit to get {ranks: individuals}
            ranked = self.calculate_pop_fit()

            # for i in ranked:
            #     temp_sum += ranked[i].fitness
            # pop_fit_mean = temp_sum / number_of_parents
            # print("avg", pop_fit_mean)
            # alpha = random.uniform(0, pop_fit_mean)
            # i_sum = self.population[0].fitness
            # # confirmed to be int
            #
            # # fitness of the first individual
            # j = 0
            # delta = alpha * pop_fit_mean
            # next_pop = []
            # # print("reach here4")
            # # represents the next generation
            # for d in self.population:
            #     while j < number_of_parents:
            #         if delta < i_sum:
            #             next_pop.append(d)
            #             # selecting the jth individual
            #             delta += i_sum
            #
            #         else:
            #             j += 1
            #             i_sum += d.fitness
            #
            # random.shuffle(next_pop)
            #
            # # print("reach here5")
            # # to randomise order of list for parent crossover
            # # print(len(next_pop))
            # final = TempPopulation(next_pop)
            # # list of Individual objects
            # for i in final.population:
            #     print(i.fitness)
            # return final


def create_next_gen_v1a(self):
    """
    Selection of parents for crossover for next generation by
    Linear Rank Selection (LRS).
    """
    if test_1_gen.gen_number == max_gen1:
        return "FINAL GENERATION REACHED"
    else:
        # check best of last 20% or 30% of the generations
        # print("reached here2")
        if test_1_gen.gen_number >= 0.3 * max_gen1:  # start checking only after completion of 30% of all gens
            c_gen = test_1_gen.gen_number
            start = c_gen - prev_gen1

            counter = True
            for k in range(int(start),
                           c_gen):  # ERROR OF ONE since first gen has gen_number 0????
                if self.best.fitness != test_1_gen.optima[k][1][1]:
                    counter = False
                    break

                if counter:  # same as last 30% gens
                    return "FINAL GENERATION REACHED"
        else:

            ranked = self.calculate_pop_fit()
            pop = []
            for i in ranked:
                pop.append(ranked[i])
            # Tournament Selection (temporary)
            # larger tournament size t, weaker individuals have a chance to be selected
            # and smaller t, vice versa
            t = 4
            n = len(ranked)
            next_pop = []
            for l in range(1, number_of_parents + 1):
                tournament = random.sample(pop, t)
                fit_list = []
                for k in tournament:
                    # print(k.fitness)
                    # can use rank to get max?
                    fit_list.append(k.fitness)
                m = max(fit_list)
                for k in tournament:
                    if k.fitness == m:
                        next_pop.append(k)
                        # may have repeats
                        break

            final = TempPopulation(next_pop)
            print('len', len(next_pop))
            for i in final.population:
                i.calculate_fitness()
                print(i.fitness)

            return final

            # n = len(ranked)
            #
            # # l below refers to rank of individual
            # total_fitness = 0
            # for l in ranked:
            # #     # assign probabilities
            # #     ranked[l].prob = ranked[l].rank/n
            #     total_fitness += ranked[l].fitness
            # #
            #
            #
            #
            #     x = l/(n*(n-1))
            #     ranked[l].rank = x
            #     # ranked[l] refers to individual object
            #     # prob_list.append(x)
            # v = 1/(n - 2.001)
            # new_pop = []
            #
            # for k in self.population:
            #     alpha = random.uniform(1, v)
            #     if k.prob <= alpha:
            #         new_pop.append(k)
            # final = TempPopulation(new_pop)
            # print('len', len(new_pop))
            #
            # return final





















































