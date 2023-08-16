# Credit Card_fraud Detection

Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

There are a total of <b> 284,807 </b> transactions with only 492 of them being fraud.
The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-sensitive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.



![plot1](https://user-images.githubusercontent.com/67755812/210863918-297f81cb-b73a-4727-a01d-c3073beda967.png)

We can observe that the genuine transactions are over 99%! This is not good.
So, will apply scaling techniques on the “Amount” feature to transform the range of values.
The variable ‘Amount’ ranges from 0 to 25,691.16. To reduce its wide range, we use Standardization to remove the mean and scale to unit variance, so that 68% of the values lie in between (-1, 1).



![plot2](https://user-images.githubusercontent.com/67755812/210863983-6474501d-a311-491d-8129-0c1388dc54ff.png)

We can observe that the amount from 0- 5000 is high whereas amount 20000 in less.



![plot3](https://user-images.githubusercontent.com/67755812/210864017-43a57149-2db3-4688-8b7a-3b6e4cc1516c.png)

The time does not seem to be a crucial feature in distinguishing normal vs fraud cases.



Let’s train different models on our dataset and observe which algorithm works better for our problem. This is actually a binary classification problem as we have to predict only 1 of the 2 class labels. We can apply a variety of algorithms for this problem like Random Forest, Decision Tree, Support Vector Machine algorithms, etc.

In this project, I build <b>Random Forest, Decision Tree classifiers and Deep Learning DNN Sequential Model</b> and see which one works best. We address the “class imbalance” problem by picking the best-performed model.


#### The Metrics and Confusion Matrix Without Handling the Imbalanced Data

#### Decision Tree


##### Decision Tree Confusion Matrix

![dt_cm](https://user-images.githubusercontent.com/67755812/210864089-369481f4-fee5-47f7-a4f7-0af18daf15ab.png)


##### Decision Tree Metrics
![metrics_dt_im](https://user-images.githubusercontent.com/67755812/210864154-c19d5d54-5d4e-480e-9d19-5da272c60c99.png)


#### Random Forest


##### Random Forest Confusion Matrix

![rf_cm](https://user-images.githubusercontent.com/67755812/210864480-aea59212-b6f7-4ccc-a4e4-7221e8ae961e.png)


##### Random Forest Metrics

![metrics_rf_im](https://user-images.githubusercontent.com/67755812/210864580-c8d4e159-a24c-45c9-ab81-70ecb5e0a846.png)


#### DNN model


##### DNN model Confusin Matrix

![dnn_cm](https://user-images.githubusercontent.com/67755812/210864604-c8b01ed7-fced-43ad-84f3-061789fc6a55.png)


##### DNN model Metrics

![metrics_dnn_im](https://user-images.githubusercontent.com/67755812/210864644-f08fcfa4-1cb6-439c-8ede-adefc5f6fe81.png)



### The Metrics and Confusion Matrix After Handling the Imbalanced Data

#### Decision Tree

##### Decision Tree Confusion Matrix

![dt_cm_b](https://user-images.githubusercontent.com/67755812/210864772-88f4851f-73c7-41b0-b343-aba75b05c3ca.png)


##### Decision Tree Metrics

![metrics_dt_b](https://user-images.githubusercontent.com/67755812/210864792-6bb61b57-3984-4b06-afd3-7242bec0b93d.png)

#### Random Forest


#### Random Forest Confusion Matrix

![rf_cm_b](https://user-images.githubusercontent.com/67755812/210864865-b388f54e-ad62-4615-91f9-16bec3ee79f3.png)


#### Random Forest Metrics

![metrics_rf_b](https://user-images.githubusercontent.com/67755812/210864909-695f0e2d-1359-43ee-86ee-f9aec684f8e0.png)


#### DNN Model


##### DNN Model Confusion Matrix

![dnn_cm_b](https://user-images.githubusercontent.com/67755812/210864681-591ea401-697f-4c25-8b31-c378b0025b0a.png)


#### DNN model Metrics

![dnn_metrics_b](https://user-images.githubusercontent.com/67755812/210864933-125a97bd-135d-48d9-9a5b-55498b1c762e.png)




