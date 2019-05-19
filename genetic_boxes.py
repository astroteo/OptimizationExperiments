#!/usr/bin/python2.7
# -*-coding:Utf-8 -*
import numpy as np
import random

#step-1: create a fitness function
def fitness (password, test_word):

	if (len(test_word) != len(password)):
		print "taille incompatible"
		return
	else:
		score = 0
		i = 0
		while (i < len(password)):
			if ( test_word[i] == password[i] ):
				score+=1

			i+=1

		return score * 100 / len(password)

#step-2: create individuals
#               individual ==> word {!! fitness evaluates one individual at once !! }
#               population ==> words ={w1,w2,...wn }
def generateAWord (length):
	i = 0
	result = ""
	while i < length:
		letter = chr(97 + int(26 * random.random()))
		result += letter
		i +=1
	return result

def generateFirstPopulation(sizePopulation, length):
	population = []
	i = 0
	while i < sizePopulation:
		population.append(generateAWord(length))
		i+=1
	return population




#step-3: from one population to the next
#               3a) evaluate the population ==> sort evaluated individuals by score
#               3b) select best individual(s) ==> HyperParameter: HOW MANY INDIVIDUALS TO BE KEPT AFTER SORTING
# key-point1: shuffle after sorting !! Don't be soooo stupid !!

def computePerfPopulation(population, password):
	populationPerf = {}
	for individual in population:
		populationPerf[individual] = fitness(password, individual)
	return sorted(populationPerf.items(), key=lambda kv: kv[1],reverse=True)

def selectFromPopulation(populationSorted, best_sample, lucky_few):
	nextGeneration = []
	for i in range(best_sample):
		nextGeneration.append(populationSorted[i][0])

	for i in range(lucky_few):
		nextGeneration.append(random.choice(populationSorted)[0])

	random.shuffle(nextGeneration)# CRUCIAL: shuffle after sorting.
	return nextGeneration


#step-4: mutations & Child
def createChild(individual1, individual2):
	child = ""
	for i in range(len(individual1)):
		if (int(100 * random.random()) < 50):
			child += individual1[i]
		else:
			child += individual2[i]
	return child

def createChildren(breeders, number_of_child):
	nextPopulation = []
	for i in range(len(breeders)/2):
		for j in range(number_of_child):
			nextPopulation.append(createChild(breeders[i], breeders[len(breeders) -1 -i]))
	return nextPopulation

def mutateWord(word):
	index_modification = int(random.random() * len(word))
	if (index_modification == 0):
		word = chr(97 + int(26 * random.random())) + word[1:]
	else:
		word = word[:index_modification] + chr(97 + int(26 * random.random())) + word[index_modification+1:]
	return word

def mutatePopulation(population, chance_of_mutation):
	for i in range(len(population)):
		if random.random() * 100 < chance_of_mutation:
			population[i] = mutateWord(population[i])
	return population


#step-5: compute next population
def nextGeneration (firstGeneration, password, best_sample, lucky_few, number_of_child, chance_of_mutation):
	 populationSorted = computePerfPopulation(firstGeneration, password)
	 nextBreeders = selectFromPopulation(populationSorted, best_sample, lucky_few)
	 nextPopulation = createChildren(nextBreeders, number_of_child)
	 nextGeneration = mutatePopulation(nextPopulation, chance_of_mutation)
	 return nextGeneration


#get best individual
def getBestIndividualFromPopulation (population, password):
	return computePerfPopulation(population, password)[0]

if __name__ =='__main__':
    d ={'a':3, 'bb':5, 'z': 1}
    print d
    ds = sorted(d.items(), key = lambda kv: kv[1])
    print ds

    password = 'aeiouuoiea'
    length = len(password)
    print len('clwhovboyd')
    print len(password)

    print '------------------------FIRST ITERATION------------------------'
    firstPopulationTest = generateFirstPopulation(1000,length);
    selectedAndSortedPopulation = computePerfPopulation(firstPopulationTest, password)
    bestIndividuals = selectFromPopulation(selectedAndSortedPopulation, 10, 2)# keep best 10, and 2 lucky bastards

    print  bestIndividuals

    print '---------------------------------------------------------------'

    print '------------------------ OPTIMIZATION ------------------------'
    generation =400;
    nFirstPopulation = 100;
    safeIndividuals  =20 #nFirstPopulation/10;
    luckyBastards = 20   #nFirstPopulation/100;


    firstPopulation = generateFirstPopulation(nFirstPopulation,length);
    population = firstPopulation
    number_of_child = 5
    chance_of_mutation = 5

    # SUPER-KEY_POINT:  ((best_sample + lucky_few) / 2 * number_of_child = size_population)

    while generation >0 or password_got == password:
        population = nextGeneration (population, password, safeIndividuals, luckyBastards, number_of_child, chance_of_mutation)

        #safeIndividuals /= 2;
        #luckyBastards /=2;
        #print population[1:3]
        print generation
        password_got = getBestIndividualFromPopulation(population, password)
        generation -=1;
    print '---------------------------------------------------------------'


    print password_got
