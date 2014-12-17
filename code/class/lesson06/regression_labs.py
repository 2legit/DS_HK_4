import pandas as pd
import matplotlib.pyplot as plt
from numpy import log, exp, mean, random
from sklearn import linear_model, feature_selection
from sklearn.preprocessing import PolynomialFeatures

DATA_DIR = '../../../data/'

def SSE(pred, resp):
 	return mean((pred - resp) ** 2)

"""
Find the best fitting model to predict breaking distance for car speed
"""

stop = pd.read_csv(DATA_DIR + 'cars1920.csv')

speed = [[x] for x in stop['speed']]
dist = stop['dist'].values

# Inspect the distribution to visually check for linearity and normality
plt.scatter(speed, dist, c='b', marker='o')
plt.show()

# Attempt 1 : Simple Linear Model
regr = linear_model.LinearRegression()
regr.fit(speed, dist)

print "\nSpeed | Distance"
print "SSE : %0.4f" % (SSE(regr.predict(speed), dist)) # SSE : 227.0704
print "R2 : %0.4f" % (regr.score(speed, dist)) # R2 : 0.6511

plt.scatter(speed, dist, c='b', marker='o')
plt.plot(speed, regr.predict(speed), color='green')
plt.show()

# Attempt 2 : Lasso Regression Model with 2nd order polynominal
stop['speed_squared'] = stop['speed'] ** 2
speed_squared = stop[['speed','speed_squared']].values

lasso = linear_model.Lasso()
lasso.fit(speed_squared, dist)

print "\nSpeed | Distance"
print "SSE : %0.4f" % (SSE(lasso.predict(speed_squared), dist)) # SSE : SSE : 217.3360
print "R2 : %0.4f" % (lasso.score(speed_squared, dist)) # R2 : 0.6660

plt.scatter(speed, dist, c='b', marker='o')
plt.plot(speed, lasso.predict(speed_squared), color='green')
plt.show()

# Attempt 3 : Ridge Regression Model with 2nd order polynominal
ridge = linear_model.Ridge()
ridge.fit(speed_squared, dist)

print "\nSpeed | Distance"
print "SSE : %0.4f" % (SSE(ridge.predict(speed_squared), dist)) # SSE : 216.4946
print "R2 : %0.4f" % (ridge.score(speed_squared, dist)) # R2 : 0.6673

plt.scatter(speed, dist, c='b', marker='o')
plt.plot(speed, ridge.predict(speed_squared), color='green')
plt.show()

# Attempt 4 : Ridge Regression Model with 3rd order polynominal
stop['speed_boxed'] = stop['speed'] ** 3
speed_boxed = stop[['speed','speed_squared','speed_boxed']].values

ridge = linear_model.Ridge()
ridge.fit(speed_boxed, dist)

print "\nSpeed | Distance"
print "SSE : %0.4f" % (SSE(ridge.predict(speed_boxed), dist)) # SSE : SSE : 212.8165
print "R2 : %0.4f" % (ridge.score(speed_boxed, dist)) # R2 : 0.6730

plt.scatter(speed, dist, c='b', marker='o')
plt.plot(speed, ridge.predict(speed_boxed), color='green')
plt.show()


# Attempt 5 : Ridge Regression Model with 3nd order polynominal, custom hyperparamter

plt.scatter(speed, dist, c='b', marker='o')

for a in [0.1,0.5,1,5,10]:

	ridge = linear_model.Ridge(alpha=a)
	ridge.fit(speed_boxed, dist)

	print "\nSpeed | Distance @ " + str(a)
	print "SSE : %0.4f" % (SSE(ridge.predict(speed_boxed), dist)) # @1 SSE : 212.8165
	print "R2 : %0.4f" % (ridge.score(speed_boxed, dist)) # @1 R2 : 0.6730
	plt.plot(speed, ridge.predict(speed_boxed), c=random.rand(3,1))

plt.show()


# Attempt 6 : Ridge Regression Model with PolynomialFeatures

poly_features = PolynomialFeatures(3)
speed_poly = poly_features.fit_transform(speed)
ridge = linear_model.Ridge()
ridge.fit(speed_poly, dist)

print speed_poly

print "\nSpeed | Distance"
print "SSE : %0.4f" % (SSE(ridge.predict(speed_poly), dist)) # SSE : SSE : 212.8165
print "R2 : %0.4f" % (ridge.score(speed_poly, dist)) # R2 : 0.6730


plt.scatter(speed, dist, c='b', marker='o')
plt.plot(speed, ridge.predict(speed_poly), color='green')
plt.show()


"""
Find the best fitting model to predict mileage for gallon
"""

cars = pd.read_csv(DATA_DIR + 'cars93.csv')

cars_input = cars._get_numeric_data()
cars_input = cars_input.dropna(axis=0)

mpg = cars_input['MPG.city']
cars_input = cars_input.drop(['MPG.highway','MPG.city'],1)

fp_value = feature_selection.univariate_selection.f_regression(cars_input, mpg)

p_value = zip(cars_input.columns.values,fp_value[1])

sorted(p_value,key=lambda x: x[1])

best_five = [x[0] for x in p_value][:5]

X = cars_input[best_five].values

poly_features = PolynomialFeatures(3)
poly_feat = poly_features.fit_transform(X)

for a in [0.1,0.5,1,5,10]:

	ridge = linear_model.Ridge(alpha=a)
	ridge.fit(poly_feat, mpg)

	print "\nSpeed | Distance @ " + str(a)
	print "SSE : %0.4f" % (SSE(ridge.predict(poly_feat), mpg)) # @1 SSE : 4.0625
	print "R2 : %0.4f" % (ridge.score(poly_feat, mpg)) # @1 R2 : 0.8686