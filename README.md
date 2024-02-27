# Credit Risk Analysis Report

![Credit Report Analysis](https://cdn.merchantmaverick.com/wp-content/uploads/2018/08/bigstock-167517734-768x512.jpg)

Credit risk poses a classification problem that’s inherently imbalanced because healthy loans usually outnumber risky loans. In this analysis, I employ machine learning techniques such as RandomOverSampler and LogisticRegression to train and evaluate historical lending activity data, aiming to build a model capable of identifying the creditworthiness of borrowers.

* The lending activity data is sourced from a peer-to-peer lending services company and includes features such as loan sizes, interest rates, borrower income, number of accounts, derogatory marks, total debt, and loan status.

* I utilize the loan status feature to initialize a variable y, splitting the values into 0 (healthy loans) and 1 (high-risk loans). The distribution of the loan_status variables is as follows: 

0:  75036,
1:   2500,
Name: count, dtype: int64

* The steps involved in this machine learning process include:
    - Splitting the data into training and testing sets
    - Creating a logistic regression model with the original data
    - Predicting a regression model with resampled training data

* I imported the lending_data CSV file into a Pandas DataFrame, then created label sets y from the loan_status column and X from the remaining features of the DataFrame. I proceeded to split the label sets data (X, y) into training and testing datasets. I instantiated the logistic regression model and used the training data to fit the logistic regression model and made predictions using the testing data and predicted model. I also oversampled the training dataset using the RandomOverSampler function to fit the logistic regression model and evaluated the performances of both models to determine the accuracy in predicting both the 0 (healthy loans) and the 1 (high-risk loans).

![Logistic Regression Model](https://1.bp.blogspot.com/-HKbBq_CeZz8/XluCouqSZtI/AAAAAAAABrk/eWnmV4M9-OQQDA4I9TGkJjq35C8iwS9kgCNcBGAsYHQ/s1600/mlintro.png)

## Machine learning model 1 (Logistic Regression)

A balanced accuracy score of 0.967989851522121 using logistic regression indicates that the model achieved a high level of accuracy in predicting the correct labels for both classes, while considering the imbalance in the dataset.

For '0' (healthy loan), the model has high precision, recall, and F1-score, indicating it performs very well in predicting healthy loans. It also shows a relatively high specificity, suggesting that it correctly identifies the majority of healthy loans.

For '1' (high-risk loans), the model has slightly lower precision, recall, and F1-score compared to healthy loans, but they are still acceptable. The specificity is also very high, indicating the model effectively identifies high-risk loans with few false positives.

Balanced accuracy score: 0.967989851522121

Confusion matrix:
[[18655   110]
 [   36   583]]

 Classification report:
                    pre       rec       spe        f1       geo       iba       sup

          0       1.00      0.99      0.94      1.00      0.97      0.94     18765
          1       0.84      0.94      0.99      0.89      0.97      0.93       619

avg / total:      0.99      0.99      0.94      0.99      0.97      0.94     19384


![Resampled Training Data](https://miro.medium.com/max/1200/1*VhF3Ibprkk9DsMAyQZ20eg.jpeg)

## Machine learning model 2: (Resampled Training Data)

A balanced accuracy score of 0.9935981855334257 when fitting the resampled training data indicates that the model achieved a higher level of accuracy in predicting both the 0 (healthy loan) and 1 (high-risk loan) labels, while also considering imbalances in the dataset.

For Class 0 (healthy loans), the model effectively predicts healthy loans with very few false positives.

For Class 1 (high-risk loans), the model identifies high-risk loans with few false negatives.

Balanced accuracy score: 0.9935981855334257

Confusion matrix:
[[18646   119]
 [    4   615]]

 Classification report:
                    pre       rec       spe        f1       geo       iba       sup

          0       1.00      0.99      0.99      1.00      0.99      0.99     18765
          1       0.84      0.99      0.99      0.91      0.99      0.99       619

avg / total:      0.99      0.99      0.99      0.99      0.99      0.99     19384


## Summary

Overall, both models performed and predicted very well, indicating a strong predictive performance of the machine learning models. For '1' (high-risk loans), the Machine learning model (Resampled Training Data) performed better with a recall of 0.99 and F1-score of 0.91 compared to the Machine learning model (Logistic Regression). The model trained on the resampled data performed slightly better overall, particularly in correctly identifying high-risk-loans.

I choose the Model trained with the resampled data for this task as it provided a higher balanced accuracy for both the 0 and 1! It also was able to provide better performance when solving for 1 compared to the model trained only on the logistic regression.
