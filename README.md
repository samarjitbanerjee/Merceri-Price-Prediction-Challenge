# Merceri-Price-Prediction-Challenge
# Table of Contents:
Introduction
Business Problem
Prerequisites
Data Source
Understanding the data
EDA
Data Preprocessing
Benchmark Solution
First Cut Solution
Deep Learning-Based Solution
Conclusion
Deployment and Predictions

# Introduction
Mercari, Japan’s biggest community-powered shopping app, knows this problem deeply. They’d like to offer pricing suggestions to sellers, but this is tough because their sellers are enabled to put just about anything, or any bundle of things, on Mercari’ s marketplace. This provides a platform where customers can sell items that are no longer useful/unused products. It tries to make all the processes hassle-free by providing at-home pickups, same-day delivery, and many other advantages. The company website displays more than 350k items listed every day on the website which reflects its popularity among users.

# Business Problem
The problem is easy to understand where, given the details of the product, the price for the product should be the output. When we pose this as a machine learning problem we call this out as a Regression Problem as the output is the real number(price). It can be treated as a Price Prediction Challenge for a product given its details.

# Prerequisites
Here are the assumptions, reader should be having the understanding of Machine Learning, Deep Learning tool-kits and basic understanding of the ML/DL algorithms. As this is a regression problem predicting selling price of the products. Here are some ML algorithms understanding must have to continue with the rest of the content.
Linear Regression with L1&L2 Regularization.
Decision Tree Regressor
Basic understanding of Clustering with K-Means Algorithm
Basic understanding of MLP, Embedding Layer

# Evaluation
The evaluation metric for this competition is Root Mean Squared Logarithmic Error.
The RMSLE is calculated as

Where:
epsilon is the RMSLE value (score)
n is the total number of observations in the (public/private) data set,
pi is your prediction of price, and
ai is the actual sale price for i.
log(x) is the natural logarithm of (x)
