# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 01:39:35 2021

@author: carlo
"""
#Final coding proyect
#import packages-------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA

#data cleaning and motives----------------------------------------------------

data = pd.read_csv('middleSchoolData.csv')
data = data.replace(r'^\s*$', np.nan, regex=True)
data = data.dropna()
data = data.reset_index(drop=True)

#if we drop missing rows with nan's, we will reduce rows from 594 to 449, losing 145 schools
#where we already know that 109 of them are charter schools and 36 normal schools

#if we want to use the charter schools, what should we do then?
#---------------------------------------------------------------------------

#descriptive stats
column_sum1 = data['applications'].sum()
column_sum2 = data['acceptances'].sum()
general_rate = column_sum2 / column_sum1
#19% of all the people passed

#displays correlation matrix
dataMatrix1 = data.corr(method='pearson', min_periods=1)
#display correlation matrix using spearman
dataMatrix2 = data.corr(method='spearman', min_periods=1)

#Question 1------------------------------------------------------------------

#displays the correlation between applications and admissions
corr1 = data['applications'].corr(data['acceptances'])

corr1 = format(corr1, ".3f")

plt.scatter(data['applications'], data['acceptances'])
plt.title('correlation between num. appplications and acceptances r = ' + str(corr1))
plt.xlabel('Applications')
plt.ylabel('Acceptances')
plt.show()


#Question 2------------------------------------------------------------------

#Now we create an application rate column and compare its usability with d1
data['application_rate'] = (data['applications']/data['school_size'])

#And then we compute the correlation with application_rate

corr2 = data['application_rate'].corr(data['acceptances'])

corr2 = format(corr2, ".3f")

plt.scatter(data['application_rate'], data['acceptances'])
plt.title('correlation between appplication rate and acceptances r = ' + str(corr2))
plt.xlabel('Application rate')
plt.ylabel('Acceptances')
plt.show()


#which one is better? corr1???

#Question 3------------------------------------------------------------------

#For this question, we could either compute the acceptance rate of each school
#where acceptance is divided by the number of applications

#here we compute the acceptance rate
data['acceptance_rate'] = (data['acceptances']/data['applications'])
#now we get the maximum acceptance rate
max_index = data["acceptance_rate"]. max()

#pr = data['school_name'].where(data['acceptance_rate'] == max_index)

indexSchool = data.index[data['acceptance_rate'] == max_index].tolist()

bestSchoolAdmissionRate = str(data._get_value(indexSchool[0], 'school_name'))
      
#and we output the name of the school by sorting the columns by acceptance_rate
#display:print(data.sort_values(by ='acceptance_rate', ascending=False))
# or data1 = data.sort_values(by ='acceptance_rate', ascending=False) and data1.head()

#which is THE CHRISTA MCAULIFFE SCHOOL\I.S. 187 school with a 0.8167 acceptance rate
#20K187

#Second option to compute this111111111111111111111111111111111111111

#or we can compute an acceptance rate based on acceptance divided by number of students
#which will output less indexes, but it would represent the populations of students
#who apply and didn't apply to capture the odds

data['acceptance_rate2'] = (data['acceptances']/data['school_size'])
#now we get the maximum acceptance rate
max_index1 = data["acceptance_rate2"]. max()
#and we output the name of the school by sorting the columns by acceptance_rate
#display:print(data.sort_values(by ='acceptance_rate2', ascending=False))
# or data1 = data.sort_values(by ='acceptance_rate2', ascending=False) and data1.head()
#which one is better???


indexSchool1 = data.index[data['acceptance_rate2'] == max_index1].tolist()

bestSchoolAdmissionRate1 = str(data._get_value(indexSchool1[0], 'school_name'))

#THE CHRISTA MCAULIFFE SCHOOL\I.S. 187 again, has the best per student results with
#a 0.234822 percent

#third option regarding "odds" definition from event happening/event not happening
#this means that we must do acceptances/school_size-acceptances which will be 
data['acceptance_rate3'] = (data['acceptances']/(data['school_size']-data['acceptances']))
max_index2 = data["acceptance_rate3"]. max()


indexSchool2 = data.index[data['acceptance_rate3'] == max_index2].tolist()

bestSchoolAdmissionRate2 = str(data._get_value(indexSchool2[0], 'school_name'))

#display:print(data.sort_values(by ='acceptance_rate3', ascending=False))


#with a 0.306886 percent, THE CHRISTA MCAULIFFE SCHOOL\I.S. 187 again

#So we know that's the best school, for sure, when talking about this

#and we could also use these methods for accounting for charter schools

#how to visualize this? with a bar chart of the higher admission rates

data['index'] = range(1, len(data) + 1)

plt.bar(data['index'], data['acceptance_rate'])
plt.xlabel("school index")
plt.ylabel("acceptance rate")
plt.title("acceptance rates per school")
plt.show()

#As we can see, there is one school with an admission rate close to 0.8
#and an index close to 270, so we are cool here



#question 4-------------------------------------------------------------------


#Is there a relationship between how students perceive their school 
#(as reported in columns L-Q) and how the school performs on objective measures 
#of achievement (as noted in columns V-X)

#to answer this question, we could dimensionally reduced these spaces to a handable
#number of dimensions and then seek for relationships between dimensions
#therefore, we must apply PCA and then seek for reduccioness

#first, lets look at the data


# Plot the data:
#plt.imshow(dataMatrix1) 
#plt.colorbar()

#we can see that there are certain correlated spots, but the data is too big to
#distinguish this

#lets do this on smaller datasets

#now we separate what we care about

datapart1 = data[["rigorous_instruction","collaborative_teachers","supportive_environment",
             "effective_school_leadership", "effective_school_leadership", 
             "strong_family_community_ties", "trust"]].to_numpy()

datapart2 = data[["student_achievement","reading_scores_exceed",
             "math_scores_exceed"]].to_numpy()

dataMatrixTest1 = np.corrcoef(datapart1,rowvar=False)
dataMatrixTest2 = np.corrcoef(datapart2,rowvar=False)

#here we show the first part

#plt.imshow(dataMatrixTest1) 
#plt.colorbar()

#as we can see, there are high correlations between this dataparts
#therefore, we should expect a considerable dimension reduction

#now we display the second part
#jfbejfnwoefjnnnnnnnnnnjfewwwwwwwwwwwwwwwwwwwwww

#plt.imshow(dataMatrixTest2) 
#plt.colorbar()
#same thing is true for this part, highly correlated, and it has less variables

#now less dimensionally reduce them by brute force and see HOW IT GOESSS

#for this proccess, we should dimensionally reduce each one, and then compare it
#with each other at the end of the proccess

#IS THIS TRUE>?????? ^^^^^^^

#so we start with the first part, columns L to Q

zscoredData1 = stats.zscore(datapart1)

pca = PCA().fit(zscoredData1)

eigVals = pca.explained_variance_

loadings = pca.components_

rotatedData = pca.fit_transform(zscoredData1)

covarExplained = eigVals/sum(eigVals)*100

#now we plug them and see how it goes (praying)

numPredictors = 7
plt.bar(np.linspace(1,numPredictors,numPredictors),eigVals)
plt.title('Columns L to Q')
plt.xlabel('Factors')
plt.ylabel('Eigenvalues')

#here, we can see that we can reduce it to one eigenvalue


whichPrincipalComponent = 0 # Try a few possibilities (at least 1,2,3 - or 0,1,2 that is - indexing from 0)

# 1: The first one accounts for almost everything, so it will probably point 
# in all directions at once
# 2: Challenging/informative - how much information?
# 3: Organization/clarity: Pointing to 6 and 5, and away from 16 - structure?

plt.bar(np.linspace(1,7,7),loadings[whichPrincipalComponent,:]*-1)
plt.xlabel('Question')
plt.ylabel('Loading')

#they dont seem to point into any particular direction, probably 4 and 5
#so we sumarize this principal component with a name based in questions 4 and 5


#now the second part 

zscoredData2 = stats.zscore(datapart2)

pca = PCA().fit(zscoredData2)

eigVals1 = pca.explained_variance_

loadings1 = pca.components_

rotatedData1 = pca.fit_transform(zscoredData2)

covarExplained1 = eigVals1/sum(eigVals1)*100

#now we plug them and see how it goes (praying)

numPredictors = 3
plt.bar(np.linspace(1,numPredictors,numPredictors),eigVals1)
plt.title('Columns V to X')
plt.xlabel('Factors')
plt.ylabel('Eigenvalues')

#Only one passed the kaiser method of picking

whichPrincipalComponent = 0 # Try a few possibilities (at least 1,2,3 - or 0,1,2 that is - indexing from 0)


plt.bar(np.linspace(1,3,3),loadings1[whichPrincipalComponent,:]*-1)
plt.xlabel('assign variables')
plt.ylabel('Loading')

#they point negatively towards question 2 and 3
#what do we do in case of negative pointing??

#In the interpretation of PCA, a negative loading simply means that a certain 
#characteristic is lacking in a latent variable associated with the given 
#principal component

#so we should just create a variable named after question 2 and 3 seems it points there

#so now that we have two variables, we can apply a linear regression model
#or a spearman relationship just to check

#fit a model to see relationship
from sklearn.linear_model import LinearRegression
xlinear = (rotatedData[:,0]*-1).reshape(-1, 1)
ylinear = (rotatedData1[:,0]*-1).reshape(-1, 1)

linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(xlinear,ylinear)  # perform linear regression
Y_pred = linear_regressor.predict(xlinear)
coeffNot = linear_regressor.coef_[0,0]
coefficient = format(coeffNot, ".3f")


data['xlinear'] = xlinear
data['ylinear'] = ylinear

correlation = str(format((data['xlinear']).corr(data['ylinear']),".3f"))

#What is the difference between correlation coefficient and slope?
#Differences. The value of the correlation indicates the strength of 
#the linear relationship. The value of the slope does not. The slope 
#interpretation tells you the change in the response for a one-unit 
#increase in the predictor.

plt.scatter(xlinear, ylinear)
plt.plot(xlinear, Y_pred, color='red')
plt.plot()
plt.xlabel("How students percieve their school")
plt.ylabel("Measurements of achievement")
plt.title("How a school is percieved and how it performs")
plt.xlim([-6, 6])
plt.ylim([-4, 4])
plt.text(0.44, 3.1, "Correlation coefficient =" + correlation)
plt.text(3, 2.3, "Slope =" + coefficient)
plt.show()


proof = stats.linregress(data['xlinear'], data['ylinear'])

print(f"R-squared: {proof.rvalue**2:.6f}")
#r-square is 0.127003
print(f"P-value: {proof.pvalue:.12f}")
#p-value is 0.00000000 so its not due to chance




#where x is how its percieved and y how it performs overall 
#so name the columns and shit.
#the better they percieve their school, the worst it performs (low coef though)


#THIS CODE IS TO PROVE DISTRIBUTION OF FACTORS
#plt.plot(rotatedData[:,0]*-1,rotatedData1[:,0]*-1,'o',markersize=5)
#plt.xlabel('Overall course quality')#sumarizing question 4 and 5 from all the 7
#plt.ylabel('Hardness of course')#sumarizing question 2 and 3 from the 3 

#QUESTION 5-------------------------------------------------------------------
#Hypothesis testing: what do you want to test?

#test this hypothesis:
    
#rich schools are better, they should have higher objective measurements

#first, take the median of all incomes

#then a binary row where 1 is above median income and 0 below

#with this categorical data, see if above median income schools (1)
#have a logistic correlation with higher objective measurements

#then perform an non-parametric test
medianIncome = data['per_pupil_spending'].median()

data['aboveOrBellowCat'] = np.where(data['per_pupil_spending'] <= medianIncome, "low income", "high income")

data['achieve'] = np.where(ylinear <= 0, "low achieve", "high achieve")
crosstable = pd.crosstab(data['achieve'], data['aboveOrBellowCat'])


data['m'] = 'nyu'

crosstable1 = pd.crosstab(data['achieve'], data['m'])

from scipy.stats import chi2_contingency

chiVal, pVal, df, exp = chi2_contingency(crosstable)

#pretty cool
#reject null hypothesis

#h1 hypothesis is that high income plays a role in achievement measures

#we can also perform a  point-biserial correlation coefficient
#continous achievement = ylinear
#and categorical independent (x) = data['aboveOrBellow']
data['aboveOrBellowBi'] = np.where(data['per_pupil_spending'] <= medianIncome, 0.0, 1.0)

#low income is 0 



resultx = stats.pearsonr(data['aboveOrBellowBi'], (rotatedData1[:,0]*-1))

num  = format(resultx[0],".3f")

plt.bar(data['aboveOrBellowBi'], (rotatedData1[:,0]*-1), align='center', width=1)
plt.plot()
plt.xlabel("Income of the students (1 is above median income, and 0 is bellow)")
plt.ylabel("Measurements of achievement")
plt.title("Cumulative achievements based on income levels")
plt.xlim([-1, 2])
plt.ylim([-4, 4])
plt.text(0.44, 3.1, "Correlation coefficient =" + str(num))
plt.show()


plt.scatter((rotatedData1[:,0]*-1),data['aboveOrBellowCat'])
plt.plot()
plt.xlabel("Measurements of achievement")
plt.ylabel("Income")
plt.title("Achievement results based on income")
plt.xlim([-4, 4])
plt.ylim([-4, 4])
plt.text(0.38, 3, "Correlation coefficient =" + str(num))
plt.show()
#(0.4426199031565975, 5.780077095384254e-23) resultx results












#QUESTION 6------------------------------------------------------------------

#hypothesis is that the higher the class size is, the worst they perform
#whereas when talking about admission rates, there are possible confounders such as
#their motivation to apply given the class size, which are uncontrolable variables

#so we must study class size and measurements of achievement (reduction done before)

#from research, we say that "Researchers generally agree a class size of no larger 
#than 18 students is required to produce the desired benefit"
#https://www.hmhco.com/blog/class-size-matters
#if average size class is higher than 18 then its an inneficient size

#lets test if this is trueeeeeeeeeeeeeeeeeeeeeeeeee

#fit a model to see relationship

npDataSize = data['avg_class_size'].to_numpy()
npDataIncome = data['per_pupil_spending'].to_numpy()

xcomp = (npDataSize).reshape(-1,1)
ycomp = ylinear.reshape(-1,1)
zcomp = (npDataIncome).reshape(-1,1)
#------------------------
data['tempa'] = ycomp
data['tempb'] = xcomp
data['tempc'] = zcomp
#------------------------
    
linear_regressor2 = LinearRegression()  # create object for the class
linear_regressor2.fit(xcomp,ycomp)  # perform linear regression
Y_pred2 = linear_regressor2.predict(xcomp)

coeffNot2 = linear_regressor2.coef_[0,0]

coefficient2 = format(coeffNot2, ".3f")

correlation2 = str(format(data['tempb'].corr(data['tempa']),".3f"))
      
#data['avg_class_size']
#plt.scatter(data['avg_class_size'], ylinear)
plt.scatter(xcomp, ycomp)
plt.xlabel("Average class size")
plt.ylabel("measurements of achievement")
plt.plot(xcomp, Y_pred2, color='red')
plt.text(21, 3.1, "correlation coefficient =" + correlation2)
plt.text(28, 2.3, "Slope =" + coefficient2)
plt.ylim([-4, 4])
plt.title("Does class size has a relationship with measurements of achievement?")
plt.show()

import pingouin as pg

pg.partial_corr(data=data, x='tempb', y='tempa', covar='tempc').round(3)


#so class size does have a somewhat strong relationship with mesuarements
#of achievement

#DO T TESTTT OR P VALUE TO ASSESS IF THERE ARE ENOUGH PROOFS??




#what relatioships can we obtain here? linear model??

#QUESTION 7-------------------------------------------------------------------
#Listen to mateus and calculate this with a for loop:
    
#7) What proportion of schools accounts for 90% of all students accepted
# to HSPHS? 

#So we first calculate the amount of students accepted to hsphs which is the sum
#of the acceptance column


overallAcceptances = column_sum2

#Now we could create a column in which we write what portion each school accounts for
#by dividing their acceptance factor by overallAcceptances
#and then summing and incresing the counter until it reachers 90%


data['acceptances_portion'] = (data['acceptances']/ int(overallAcceptances))

data = data.sort_values('acceptances_portion', ascending=False)

data['cumsum'] = data['acceptances_portion'].cumsum()

#when cumsum is equal to 0.9, then we take the index of that level
        
indexChecker = data.index[(data['cumsum'] > 0.8999)<=0.9].tolist()

finalNumber = len(indexChecker)

print(finalNumber)

#446 starting from lower to higher proportions
#92 starting from higher proportions
#so 392 schools account for 90% of all acceptance
#391 out of 449

#we check for double-counting

#pd.set_option("display.max_rows", None, "display.max_columns", None)

#is not counting right. Lets create a counting column and return that value

data['index'] = range(1, len(data) + 1)

z = data['index'].where((data['cumsum'] > 0.8999999)<0.9)

#392 is the resulttttttttttt 

#so 392 schools represent 90% of all students accepted to HSPHS

#and 392 out of 449 represents the 87.30 percent of all schools

#so 87% of all schools fully public s. represent for 90% of all students accepted 

#to HSPHS
data.sort_values(by='acceptance_rate', ascending=False)
plt.bar(data['index'], data['acceptances_portion'])
plt.xlabel("Number of schools")
plt.ylabel("acceptance portion per school")
plt.title("acceptance rates per school")
plt.show()


#DOOOOO THISSSSSSSSSSSSSSSSSSS

#But lets do it by including the charter schools, so only detele nans in acceptance column


#datax = pd.read_csv('middleSchoolData.csv')

#overall = datax['applications'].sum()

#25349 is the value of overall


#datax['acceptances_portion1'] = (datax['acceptances']/ int(overall))

#datax.sort_values('acceptances_portion1', ascending=False)

#datax['cumsum1'] = datax['acceptances_portion1'].cumsum()

#when cumsum is equal to 0.9, then we take the index of that level

#xx = datax['school_name'].where((datax['cumsum1'] > 0.8999)<0.9)
        
#yy = datax.index[(datax['cumsum1'] > 0.8999)<0.9].tolist()

#593 out of 594????

#couldnt really do it


#QUESTION 8---------------------------------------------------------------

#for this question we must do somethinggggg

#factors that could affect mesurement achievements are:
#average class size, per_pupil_spending,
from sklearn import linear_model
import statsmodels.api as sm
data['climate'] = (rotatedData[:,0]*-1)
data['objectivePerformance'] = (rotatedData1[:,0]*-1)


import statsmodels.api as sm
X = data[['applications', 'per_pupil_spending', 'avg_class_size',
          'school_size']]

y = data['objectivePerformance']

regr = linear_model.LinearRegression()
regr.fit(X, y)

print(regr.coef_)

# with statsmodels
X = sm.add_constant(X) # adding a constant
 
model = sm.OLS(y, X).fit()
predictions = model.predict(X) 
 
print_model = model.summary()
print(print_model)




#---------------------------------------------------------------------------


#we also found the same here, were we can only reduce it to one eigenvalue


#debugging-----------------------------------------------------------------------

print(data)
#print(D4)
print(corr1)
print(corr2)
print(max_index)
print(column_sum1)
print(column_sum2)
print(general_rate)


#then we can see that corr1 is higher than corr2
#therefore, raw applications is a better predictor

