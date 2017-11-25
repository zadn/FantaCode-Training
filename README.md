# FantaCode Training

This is the repository for updating training programs learned as an intern at FantaCode Solutions.


### Beginner's Game

Beginner's game is a RPG game with no class or inheritance implementations. It simply uses if-else conditions to match a user input to the specified set of inputs.

## First Week 

### Intermediate RPG

Intermediate RPG is an implementation of OOP concepts in python. It uses classes and functions to create a mini text based RPG.

## Second Week

### Data Analysis and Prediction

Analysed, wrangled, cleansed and predicted the data on [Titanic Machine Learning from Disaster](https://www.kaggle.com/c/titanic) using Jupyter Notebook.

I tried to implement a [sample notebook solution](https://www.kaggle.com/startupsci/titanic-data-science-solutions/notebook) for the Titanic competition on Kaggle. I learned many things while completing the full implementation. Some of them are:
 - Read csv file to python using Pandas DataFrame
 - Visualize related data in the dataframe using Seaborn and matplotlib
 - Create incomplete data in the dataframe
 - Combining relatively dependent data
 - Dividing continuous numeric data into ranges and then assigning values to each range
 - Mapping discrete non numeric values into discrete numeric values
 - Used multiple predictive modelling algorithms to study which algorithm gives better accuracy for our data


## Third Week

#### Tasks for Third Week
 
Learn the basics of Machine Learning and implement it in Scikit Learn
 - [x] K-means Clustering
 - [x] Linear Regression
 - [x] Logistic Regression


 ##### K-means Clustering

 K-means Clustering is when there are a number of means in a plane and we want to add some points to each cluster, according to the distances to these means. But when the points get added to the clusters, the mean of the cluster gets dislocated. So we will again cluster them according to the distances to the mean. This process is repeated till there will not be any change in the means (centroids).

 ![K-means Clustering](images/Kmeans.JPG)


 In this work, I'm trying to implement K-means clustering on Handwritten digits data. The project was inspired from [this scikit-learn example](http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html).

The data it uses is from Scikit Learn datasets

The clustering performance were evaluated with these metrics:
- Adjusted Rand Index (ARI)
- Adjusted Mutual Information (AMI)
- Homogeneity score
- Completeness score
- V-measure
- Silhouette coefficient

All those metrics have their own aspects.

We evaluate different initialization strategies for K-means using these metrics.
After evaluating, we visualize the Cluster using meshgrid in numpy. 

![K-means Clustedred](images/Visualised.JPEG)

#### Regression Analysis

Regression analysis is a set of statistical methods used to estimate relationships between the variables. This is mostly used to find correlation between independent and dependant variables.

##### Linear Regression

Linear Regression is a regression model where we relate an dependent variable to one or more independant variables. We use this in machine learning to predict numerical value of some indepedent value corresponding to dependent variables. 

For the training, I implemented Linear Regression using an [example in scikit-learn.org](http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py) in a [jupyter notebook](Third%20Week/LinearRegressionExample.ipynb)

The dataset used in this file is the 'diabetes' dataset that comes with the sklearn.datasets, imported using ```from sklearn.datasets import load_diabetes```. The last 20 values were taken to test the regression and got a variance score of `.41`. The output was plotted using ```matplotlib.pyplot.scatter```. 

The output is as shown below: 

![Linear Regression Output](images/LinearRegressionOut.jpeg)


##### Logistic Regression

Logistic regression is a regression model, which helps to relate dependent variable to independent variables where the dependent variables are categorical. It means that the output will be discrete values. When one refers to 'Logistic Regression', he is referring to the binary logistic regression, where there will be only 2 outputs, '1' or '0' (binary). 

The main difference among Linear regression and Logistic regression is that when Linear regression produces an output numerals in an infinite range, Logistic regression results in binary values. This is specifically suitable to many applications. We can use Linear regression where logistic regression are applicable, but it is not efficient. Logistic regression's efficiency comes from it's cross-entropy error function instead of least squares.

For the training, I implemented a [jupyter notebook](Third%20Week/LogisticRegression_Affairs.ipynb) inspired by a [classification of finding affairs in women](http://nbviewer.jupyter.org/gist/justmarkham/6d5c061ca5aee67c4316471f8c2ae976).


The dataset was collected from a survey conducted by Redbook magazine. The data had information like age, occupation, number of children, how they rated their married life, husband's occupation, number of extramarital affairs, letc. Information like education and occupation were already assigned values according to each categories, so data cleansing was not necessary here. First the data was preprocessed and prepared. Then the data was visualised to see how some dependents held up against the affair value. After that the data was fitted to the logistic regression model with the necessary data. Then the model was used to predict for some arguments and ran some evaluation metrics. The model turned out to be 73% accurate for predicting if a woman have affairs with the given data.