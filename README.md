# Credit Card_fraud Detection

Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

There are a total of <b> 284,807 </b> transactions with only 492 of them being fraud.
The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-sensitive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.


plot1

We can observe that the genuine transactions are over 99%! This is not good.
So, will apply scaling techniques on the “Amount” feature to transform the range of values.
The variable ‘Amount’ ranges from 0 to 25,691.16. To reduce its wide range, we use Standardization to remove the mean and scale to unit variance, so that 68% of the values lie in between (-1, 1).

plot 2

We can observe that the amount from 0- 5000 is high whereas amount 20000 in less.


plot3
The time does not seem to be a crucial feature in distinguishing normal vs fraud cases.


Let’s train different models on our dataset and observe which algorithm works better for our problem. This is actually a binary classification problem as we have to predict only 1 of the 2 class labels. We can apply a variety of algorithms for this problem like Random Forest, Decision Tree, Support Vector Machine algorithms, etc.

In this project, I build <b>Random Forest, Decision Tree classifiers and Deep Learning DNN Sequential Model</b> and see which one works best. We address the “class imbalance” problem by picking the best-performed model.


#### The Metrics and Confusion Matrix Without Handling the Imbalanced Data

#### Decision Tree
dtcm

dt_metrics_ib


#### Random Forest

rf_cm

rf_metrics


#### DNN model

dnn_cm

dnn_metrics



### The Metrics and Confusion Matrix After Handling the Imbalanced Data

#### Decision Tree
dtcm

dt_metrics_ib


#### Random Forest

rf_cm

rf_metrics


#### DNN model

dnn_cm

dnn_metrics



