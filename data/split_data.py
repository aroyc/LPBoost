# Programmed by Christian Eubank

import random

# split data into training and test
def split_data(root_name, prob=.75) :
	fraw = open("./raw_" + root_name + ".txt", "r")
	ftrain = open("./train_" + root_name + ".txt", "w")
	ftest = open("./test_" + root_name + ".txt", "w")

	for line in fraw:
		if len(line.strip()) > 0:
			p = random.random()
			if p < prob:
				ftrain.write(str(line))
			else:
				ftest.write(str(line))

	fraw.close()
	ftrain.close()
	ftest.close()

split_data("38")
split_data("49")
split_data("votes")
split_data("cars")
split_data("shrooms")
