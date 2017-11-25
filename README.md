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
 - [ ] Linear Regression
 - [ ] Logistic Regression


 ##### K-means Clustering

 K-means Clustering is when there are a number of means in a plane and we want to add some points to each cluster, according to the distances to these means. But when the points get added to the clusters, the mean of the cluster gets dislocated. So we will again cluster them according to the distances to the mean. This process is repeated till there will not be any change in the means (centroids).

 ![K-means Clustering](/images/Kmeans.JPG)


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

![K-means Clustedred](/images/Visualised.JPEG)
