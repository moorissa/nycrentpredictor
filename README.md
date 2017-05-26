# Predicting NYC Renting Prices using Lasso Regression
An attempt to find the best predictive linear model using scikit-learn.

Link: https://github.com/moorissa/nycrentpredictor/

Contributors:
* Moorissa Tjokro
* Arman Uygur

<p>We create functions to load_data that imports and parses a dataset directly from a link as a csv file, standardizes, and imputes null values of each categorical and continuous feature in the dataset. It also splits dataset into two training sets and two test sets, which are X_train, y_train, X_test and y_test. In this case it we assign 33% test data for the default split cutoff.</p>


<p>To make sure that we only use features that apply to pricing an apartment that is not currently rented, we match the features from Vacant Housing file with that of Occupied Housing file when loading our dataset. This leaves us with 10,138 rows, 41 independent variables, and a response variable.</p>



<p>In dealing with missing values, we perform imputation by defining a DataFrameImputer function, that impute categorical columns of dtype object using the most frequent value and continuous columns by taking the mean.</p>



<p>We started off modeling by first running a base model. In this case, we used three main linear models for regression, which are linear regression, ridge regression, and lasso. Note that we didn't use logistic regression or SVM approach because we wanted to predict in continuous value. The base model outputs R^2 = 0.25 for the training set and R^2 = 0.26 for the testing set, which means the model is underfitting in this case. </p>



<p>To accomplish a higher R^2, we tried adding more variables to the existing common 41 by analyzing each variable in the Occupied Housing file that can apply properly to the Vacant Housing files, such as number of bedrooms, units in building, and elevator in building, which also represent binary variables. We ended up adding 19 variables to our existing model and performed Lasso, which gives us 0.32 for the training R^2 and 0.35 for the testing R^2.</p>



<p>Once the new variables improved our model performance, we tried increasing it by looking at scatterplot for individual variables and see if there are any interesting patterns found in the variables. We found a few correlations, which we ended up deleting. This was also confirmed by performing Lasso that took off coefficients for different variables.</p>



<p>Removing insignificant variables left us with 55 independent variables and one target. We then performed One-Hot-Encoding to create dummy variables for the categorical variables. Adding dummies gave us a training set score of 0.55 and a test set score of 0.54. We improved our model by a significant amount. We again iteratively performed linear regression, ridge regression, and lasso. In ridge regression and lasso, we tried tweaking the alpha values, ranging from 0.01 to 10 to 100, to see if with different model complexities we can get higher R^2. However, looking at all performance from these models we saw a consistent result of R^2 = 0.54 across three models.</p>



<p>We decided to look at other models such as grid search across different cv values, its combinations with linear regression models, and also recursive feature eliminations with cross validations (rfe) approaches. We performed Gaussian distribution to normalize a 'fw' variable (weight of household), which was quite significant in our model. Nevertheless, the methods still give us similar performance, if not worse. Which we did not prefer to use. However, these models let us see which features are better than the other, so we decided to include some features in a list of variables to remove. After trying out different sets of to-delete-list, we arrived at a slightly better R^2 value of 0.55. We were able to evaluate using k-fold cross validations as well.</p>



<p>Finally, we found that variables like new control status recode (new_csr), condition of walls and stairways, are among very important variables that make up our model. This makes sense because the condition of housings, size of the apartment, and borough areas (location) are usually the most important aspect that people look at when they in search for renting a good apartment. Therefore we validated our final model using Lasso regression with R^2 value of 0.55.</p>
