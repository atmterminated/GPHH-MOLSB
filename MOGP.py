import operator
import math
import random
import pickle
import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
from deap.benchmarks.tools import diversity, convergence, hypervolume, igd

import brokeringdataset

import numpy as np
import matplotlib.pyplot as plt

latencymap = brokeringdataset.latencymap
datacenter = brokeringdataset.datacenter

traindata = brokeringdataset.traindata2
testdata = brokeringdataset.testdata2



def getLatencyList(userID):
    latencylist = latencymap[userID]
    return latencylist


def latency_DatacenterToUser(userID):
    latencylist = latencymap[userID]
    dc = list(col[0] for col in datacenter)
    latency = []
    for i in range(len(dc)):
        l = [latencylist[dc[i]], dc[i]]
        latency.append(l)
    return latency


def findMinimumLatency(latencymap):
    sorted(latencymap, key=lambda l: l[1])
    return latencymap[0][0], latencymap[0][1]


def findClosestDatacenter(userID):
    map = latency_DatacenterToUser(userID)
    latency, dcID = findMinimumLatency(map)
    return latency, dcID


def chooseVM(vmcorecount, memory):
    if vmcorecount <= 1 and memory <= 4:
        return 0
    elif vmcorecount <= 2 and memory <= 8:
        return 1
    elif vmcorecount <= 4 and memory <= 16:
        return 2
    elif vmcorecount <= 8 and memory <= 32:
        return 3
    else:
        return 4


def getprice(VMtype, dcID):
    for i in range(len(datacenter)):
        if datacenter[i][0] == dcID:
            pricelist = datacenter[i][2]
            return pricelist[VMtype]
    return None


def getpricelist(dcID):
    for i in range(len(datacenter)):
        if datacenter[i][0] == dcID:
            pricelist = datacenter[i][2]
            return pricelist
    return None


def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1


pset = gp.PrimitiveSet("MAIN", 5)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.sin, 1)
pset.addPrimitive(max, 2)
pset.addPrimitive(min, 2)


pset.addEphemeralConstant("rand100", lambda: random.random() * 100)
pset.addEphemeralConstant("rand101", lambda: random.randint(-1, 1))
# pset.renameArguments(ARG0='cpu',ARG1='memory',ARG2='time',ARG3='location')	
pset.renameArguments(ARG0='cpu', ARG1='memory', ARG2='time', ARG3='location', ARG4='price')		###Add more attributes

import sys
import inspect
from inspect import isclass

# creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()

toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


def calculate_cost(price, time):
    return price * time


def normalize(x, max, min):
    xnorm = (x - min) / (max - min)
    return xnorm


def evaluate(individual, traindata):
    func = toolbox.compile(expr=individual)
    # f_list = []
    sum_points = []
    cost_sum = 0
    latency_sum = 0
    points = []
    cost_l = []
    latency_l = []
    count = 0
    for i in range(len(traindata)):
        request_list = traindata[i]
        for j in range(len(request_list)):
            # for each request in training data
            count = count + 1
            request = request_list[j]
            userid = request[0]  # location
            cpu = request[1]
            memory = request[2]
            time = request[3]  # hr
            # get VM type
            vmtype = chooseVM(cpu, memory)
            latency_list = latency_DatacenterToUser(userid)
            f_list = []
            for lat in range(len(latency_list)):
                dcid = latency_list[lat][1]
                price = getpricelist(dcid)

                f_value = func(cpu, memory, time, latency_list[lat][0], price[0])  

                f_list.append([f_value, latency_list[lat]])
                # print(f_list)
            ### sort by f value, f_list: f_value,location

            f_max = f_list[0]
            for k in range(len(f_list)):
                if f_max[0] > f_list[k][0]:
                    f_max = f_list[k]
            latency = f_max[1][0]
            price = getprice(vmtype, f_max[1][1])
            cost = calculate_cost(price, time)
            cost_l.append(cost)
            latency_l.append(latency)
            points.append([cost, latency])
    for i in range(len(points)):
        cost_sum = cost_sum + points[i][0]
        latency_sum = latency_sum + points[i][1]
    return cost_sum, latency_sum / count


def sum_generation(individual, traindata):
    func = toolbox.compile(expr=individual)
    sum_points = []
    cost_sum = 0
    latency_sum = 0
    for i in range(len(traindata)):
        request_list = traindata[i]
        for j in range(len(request_list)):
            # for each request in training data
            request = request_list[j]
            userid = request[0]  # location
            cpu = request[1]
            memory = request[2]
            time = request[3]  # hr
            # get VM type
            vmtype = chooseVM(cpu, memory)
            latency_list = latency_DatacenterToUser(userid)
            f_list = []
            for lat in range(len(latency_list)):
                f_value = func(cpu, memory, time, latency_list[lat][0])

                f_list.append([f_value, latency_list[lat]])

            ### sort by f value, f_list: f_value,location

            f_max = f_list[0]
            for k in range(len(f_list)):
                if f_max[0] > f_list[k][0]:
                    f_max = f_list[k]
            latency = f_max[1][0]
            price = getprice(vmtype, f_max[1][1])
            cost = calculate_cost(price, time * 3600.0)

    return cost_sum, latency_sum  ###


toolbox.register("evaluate", evaluate, traindata=traindata)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.register("select", tools.selNSGA2)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=7))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=7))


def main():
    NGEN = 100
    MU = 1024
    CXPB = 0.9

    pop = toolbox.population(n=MU)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean, axis=0)
    stats.register("std", numpy.std, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    pareto = tools.ParetoFront()

    pop_init = pop[:]
    hof = tools.HallOfFame(1)
    pop_hist = []
    pop_hist.append(pop)
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]

    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if hof is not None:
        hof.update(pop)
    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    pop = toolbox.select(pop, len(pop))

    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    print(logbook.stream)
    sum_list_gen = []

    for gen in range(0, NGEN - 1):
        # Vary the population
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):

            if random.random() <= CXPB:
                toolbox.mate(ind1, ind2)

            if random.random() <= 0.1:
                toolbox.mutate(ind1)
                toolbox.mutate(ind2)

            del ind1.fitness.values, ind2.fitness.values
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        if hof is not None:
            hof.update(offspring)

        if gen == 0 or gen % 20 == 0:
            sum_list = []
            for ind in pop:
                cost_sum, latency_sum = evaluate(ind, traindata)
                sum_point = [cost_sum, latency_sum]
                sum_list.append(sum_point)

            sum_list_gen.append(sum_list)

        pop = toolbox.select(pop + offspring, MU)
        pop_hist.append(pop)

        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        print(logbook.stream)

    pareto.update(pop)

    return pop, logbook, pop_init, hof, sum_list_gen, pareto, pop_hist


def get_pareto(population):
	# Given a population of individuals, return the pareto front
    pareto = []
    for ind1 in population:
        dominance = True
        cost1 = ind1.fitness.values[0]
        latency1 = ind1.fitness.values[1]

        # check dominance
        for ind2 in population:
            cost2 = ind2.fitness.values[0]
            latency2 = ind2.fitness.values[1]
            if (cost1 > cost2 and latency1 > latency2) or (cost1 == cost2 and latency1 > latency2) or (
                    cost1 > cost2 and latency1 == latency2):
                dominance = False
        if dominance:
            pareto.append(ind1)
    return pareto


if __name__ == "__main__":

    run_30 = []
    pop_30 = []
    pop_hist_30 = []
    for seed in range(0, 30):
        print('Run:' + str(seed))
        random.seed(seed)
        pop, log, pop_init, hof, sum_list_gen, pareto, pop_hist = main()
        run_30.append([pop, log, pop_init, hof, sum_list_gen, pareto, pop_hist])
        pop_hist_30.append(pop_hist)
        for ind in pop:
            pop_30.append(ind)
    pareto_30 = get_pareto(pop_30)  ###To calculate IGD
    pareto_30_fit = np.array([list(pareto_30[i].fitness.values) for i in range(len(pareto_30))])
    pop_30_fit = np.array([list(pop_30[i].fitness.values) for i in range(len(pop_30))])

    plt.plot(pop_30_fit[:, 0], pop_30_fit[:, 1], "r.", label="10 Runs Total")
    # plt.plot(pareto_30_fit[:,0], pareto_30_fit[:,1], "b.", label="Pareto")
    plt.legend(loc="upper right")
    plt.title("fitnesses")
    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.grid(True)
    plt.show()
