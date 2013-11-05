# Programmed by Christian Eubank

from subprocess import call
import math

# decision stump classifier
class Stump:
	def __init__(self, index, orientation) :
		self.index = index
		self.orientation = orientation

	def classify(self, vector) :
		if self.orientation == True:
			if float(vector[self.index]) < 0.5:
				return -1
			return 1
		else:
			if float(vector[self.index]) < 0.5:
				return 1
			return -1

# boost classifier used for LPBoost and AdaBoost
class BoostClassifier:
	def __init__(self, alphas, hypotheses) :
		self.alphas = alphas
		self.hypotheses = hypotheses

	def classify(self, vector) :
		class_sum = 0.0
		for i in range(0,len(self.alphas)):
			class_sum += float(self.alphas[i]) * float(self.hypotheses[i].classify(vector))

		if class_sum < 0.0:
			return -1
		return 1


#examples is vector of vectors
def getClassifierStump(weights, examples, labels) :
	highestError = float("-inf")

	for i in range(0,len(examples[0])) :
		stump1 = Stump(i,True)
		stump2 = Stump(i,False)
		sum1 = 0.0
		sum2 = 0.0

		for j in range(0,len(examples)) :
			sum1 += float(stump1.classify(examples[j])) * float(labels[j]) * float(weights[j])
			sum2 += float(stump2.classify(examples[j])) * float(labels[j]) * float(weights[j])

		if sum1 > highestError:
			highestError = sum1
			bestLearner = stump1
		if sum2 > highestError:
			highestError = sum2
			bestLearner = stump2

	return bestLearner

# get the weights out of the file corresponding to the dual solution
def extractWeightsFromDual(file_name):
	weights = []
	f = open(file_name, "r")
	for line in f:
		if len(line.strip()) < 3:
			continue
		components = line.split()
		if len(components) == 3 and 'u' in components[0] and "=" in components[1]:
			weights.append(float(components[2]))
		if len(components) == 3 and 'Beta' in components[0] and "=" in components[1]:
			beta = float(components[2])

	f.close()
	return (beta, weights)

# get the weights out of the file corresponding to the primal solution
def extractWeightsFromPrimal(file_name):
	alphas = []
	f = open(file_name, "r")
	for line in f:
		if len(line.strip()) < 3:
			continue
		components = line.split()
		if len(components) == 3 and 'alpha' in components[0] and "=" in components[1]:
			alphas.append(float(components[2]))

	f.close()

	return alphas

## print the dual model
def printDualModel(file_name="/Users/Christian/Desktop/duck.txt", examples=[[0,0], [0,1]], labels=[-1,1], hypotheses=[Stump(0,True), Stump(1,False)], D=0.7):
	f = open(file_name, "w")

	# declare variables
	f.write("var Beta;\n")
	for i in range(0,len(examples)):
		f.write("var u" + str(i) + ";\n")
	f.write("\n")

	f.write("minimize margin: Beta;\n")


	for j in range(0, len(hypotheses)) :
		f.write("subject to con" + str(j) + ": ")
		for i in range(0, len(examples)) :
			f.write(str(hypotheses[j].classify(examples[i])) +  " * " + str(labels[i]) + " * " + "u" + str(i) + " + ")

		f.write(" 0.0 <= Beta;\n")

	# print that u's should fall into range
	for i in range(0, len(examples)):
		f.write("subject to nneg" + str(i) + ": u" + str(i) + " >= 0;\n")
		f.write("subject to upru" + str(i) + ": u" + str(i) + " <= " + str(D) + ";\n")

	# print everything sums to 1
	f.write("subject to prob: u0")
	for i in range (1, len(examples)):
		f.write(" + u" + str(i))
	f.write(" = 1;\n")

	# print results
	f.write("solve;\n")
	f.write("display Beta;\n")
	for i in range(0,len(examples)):
		f.write("display u" + str(i) + ";\n")

	f.close()


def printPrimalModel(file_name="/Users/Christian/Desktop/duck2.txt", examples=[[0,0], [0,1]], labels=[-1,1], hypotheses=[Stump(0,True), Stump(1,False)], D=0.7):
	f = open(file_name, "w")

	# declare variables
	f.write("var rho;\n")
	for i in range(0,len(examples)):
		f.write("var zeta" + str(i) + ";\n")
	f.write("\n")
	for i in range(0,len(hypotheses)):
		f.write("var alpha" + str(i) + ";\n")
	f.write("\n")

	# objective function
	f.write("maximize objective: rho")
	for i in range(0,len(examples)):
		f.write(" - " + str(D) + " * " + "zeta" + str(i))
	f.write(";\n")


	# write contraints
	for i in range(0, len(examples)) :
		f.write("subject to con" + str(i) + ": " )
		for j in range(0, len(hypotheses)) :
			f.write(str(hypotheses[j].classify(examples[i])) +  " * " + str(labels[i]) + " * " + "alpha" + str(j) + " + ")

		f.write(" zeta" + str(i) + " >= rho;\n")

	# print that a's and zetas should fall into range
	for i in range(0, len(hypotheses)):
		f.write("subject to nnega" + str(i) + ": alpha" + str(i) + " >= 0;\n")

	for i in range(0, len(examples)):
		f.write("subject to nnegz" + str(i) + ": zeta" + str(i) + " >= 0;\n")

	f.write("subject to rhonneg: rho >= 0;\n")

	# print everything sums to 1
	f.write("subject to prob: alpha0")
	for i in range (1, len(hypotheses)):
		f.write(" + alpha" + str(i))
	f.write(" = 1;\n")

	# print results
	f.write("solve;\n")
	f.write("display rho;\n")
	for i in range(0,len(hypotheses)):
		f.write("display alpha" + str(i) + ";\n")

	f.close()

### BEGIN ACTUAL ALGORITHM
def LPBoost(examples, labels, v=0.7, epsilon=0.001):
	M = len(examples)
	D = 1.0 / (M * v)
	beta = 0.0
	weights = [1.0 / float(M) for i in range(0,M)]
	alphas = []
	hypotheses = []
	
	while True:
		candHyp = getClassifierStump(weights, examples, labels)
		classSum = 0.0
		for i in range(0, len(examples)) :
			classSum += float(weights[i]) * float(labels[i]) * float(candHyp.classify(examples[i]))

		if classSum <= beta + epsilon or len(hypotheses) >= M:
			break

		hypotheses.append(candHyp)

		printDualModel("./dual_in.txt", examples, labels, hypotheses, D)
		call("ampl ./dual_in.txt > ./dual_out.txt", shell=True)
		(beta, weights) = extractWeightsFromDual("./dual_out.txt")

		printPrimalModel("./primal_in.txt", examples, labels, hypotheses, D)
		call("ampl ./primal_in.txt > ./primal_out.txt", shell=True)
		alphas = extractWeightsFromPrimal("./primal_out.txt")

	return BoostClassifier(alphas, hypotheses)

def AdaBoost(examples, labels, rounds=50) :
	M = len(examples)
	weights = [1.0 / float(M) for i in range(0,M)]
	alphas = []
	hypotheses = []

	for t in range(0, rounds):
		candHyp = getClassifierStump(weights, examples, labels)
		hypotheses.append(candHyp)

		error = 0.0
		for i in range(0, len(examples)):
			if float(labels[i]) * float(candHyp.classify(examples[i])) < 0.0:
				error += float(weights[i])
		alpha = 0.5 * math.log(((1- error) / error))
		alphas.append(alpha)

		for i in range(0, len(examples)) :
			if float(labels[i]) * float(candHyp.classify(examples[i])) < 0.0:
				weights[i] *= math.exp(alpha)
			else:
				weights[i] *= math.exp(-1.0 * alpha)

		norm = sum(weights)
		for i in range(0, len(examples)) :
			weights[i] /= norm

	return BoostClassifier(alphas, hypotheses)
