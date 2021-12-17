import pandas
import numpy
from numpy import asarray
from numpy import savetxt

from factor_analyzer import FactorAnalyzer

import matplotlib.pyplot as pyplot

#Load data:
dataset = pandas.read_csv("dataset_final.csv")
# print(dataset)
print(dataset.head())

# Drop what won't be used in the factor analyzer:
dataset.drop(['country','Unnamed: 0'], axis=1, inplace=True)

# Nan are coded as 0. 
count = (dataset == 0).sum().sum() # Inspect how many there are
print("\nCount zeros:",count) #133598
dataset.replace(0, numpy.nan, inplace=True) # replace 0 with Nan
print(dataset) # Check 
dataset.dropna(inplace=True) # drop Na
print(dataset) # Now, 15192 observations

# FINDING THE NUMBER OF FACTORS:

# Start from the maximum : each question corresponding to one factor
machine = FactorAnalyzer(n_factors=40, rotation=None)
machine.fit(dataset)
ev, v = machine.get_eigenvalues()
print(ev)

#Data visualization
pyplot.scatter(range(1, dataset.shape[1]+1), ev)
pyplot.savefig("plot.png")
pyplot.close()

# Conclusion: both the ev and the plot indicate 4 as the number of factors 

#Trying n_factor = 4

machine = FactorAnalyzer(n_factors=4, rotation='varimax')
machine.fit(dataset)
loadings = machine.loadings_
numpy.set_printoptions(suppress=True)
print(loadings)

print("factor loadings:\n")
print(loadings)
print(machine.get_factor_variance())

dataset = dataset.values #transforms the data into array

# Multiply matrices to get results (each person's score on the characteristics)
result = numpy.dot(dataset, loadings) 

print(result)
print(result.shape)

# save numpy array as csv file
# define data
data = asarray(result)
# save to csv file
savetxt('data_factors.csv', data, delimiter=',')
