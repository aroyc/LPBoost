# Programmed by Christian Eubank

import lpboost as lpb
import random
import sys
#from sklearn.ensemble import RandomForestClassifier

# build the examples and labels from the file
def buildVectors(filename) :
	examples = []
	labels = []
	f = open(filename, "r")
	for line in f:
		vec = []
		if len(line.strip()) > 1:
			vals = line.split()
			labels.append(vals[0])
			for i in range (1, len(vals)):
				if "0" in str(vals[i]) or "1" in str(vals[i]):
					vec.append(vals[i])
			examples.append(vec)

	f.close()
	return (examples, labels)

def classifyExperiment(name, lab_noise = 0.25, content_noise=0.0):
	(tr_exp, tr_labels) = buildVectors("./data/train_" + name + ".txt")
	(test_exp, test_labels) = buildVectors("./data/test_" + name + ".txt")

	# add label noise
	for i in range(0, len(tr_labels)):
			if random.random() < lab_noise:
				tr_labels[i] = float(tr_labels[i]) * -1.0

	# add content noise
	for i in range(0, len(tr_exp)):
		for j in range (0, len(tr_exp[i])):
			if random.random() < content_noise:
				tr_exp[i][j] = 1 - int(tr_exp[i][j])

	#learner = lpb.LPBoost(tr_exp, tr_labels, .07, 0.00001)
	learner = lpb.AdaBoost(tr_exp, tr_labels)
	#learner = RandomForestClassifier(n_estimators = 10)
	#learner.fit(tr_exp, tr_labels)

	total = 0.0
	correct = 0.0
	for i in range(0, len(test_exp)) :
		total += 1.0
		#print learner.predict(test_exp[i])[0]
		#if float(learner.predict(test_exp[i])[0]) * float(test_labels[i]) > 0.0:
		if float(learner.classify(test_exp[i])) * float(test_labels[i]) > 0.0:
			correct += 1.0

	print correct / total

# MAIN
if __name__ == '__main__':
	classifyExperiment(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]))


