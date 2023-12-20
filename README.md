# Syria_Tel--Customer-Churn-Analysis
Phase Three Project

BUSINESS UNDERSTANDING 
For the telecommunication industry, one of the most important metrics to monitor and optimize is the churn rate, which measures the percentage of customers who stop using the service within a given period. 
A high churn rate indicates customer dissatisfaction, loss of revenue, and reduced market share. 
Therefore, it is essential to investigate the factors that influence customer churn and develop predictive models that can identify customers who are at risk of leaving. 
![image](https://github.com/ChiuriCynthia/Syria_Tel--Customer-Churn-Analysis/assets/128600244/fd365706-2952-432c-87d2-a88f31f96604)

In this project, we build a model to predict how likely a customer will churn by analyzing its characteristics:
(1) Customer Service Calls
(2) Chrn rate in different area codes
(3) Subscriptions, Internation as well as Voice mail.
The objective is to obtain a data-driven solution that will allow us to reduce churn rates and, as a consequence, to increase customer satisfaction and corporation revenue.

Data Understanding 
*Dataset used:*
DataSet Link: https://www.kaggle.com/datasets/becksddf/churn-in-telecoms-dataset/
We then made the below observation:
The dataset contains 3333 entries and 21 columns.
The columns represent various customer attributes, including state, account length, area code, phone number, international plan, voice mail plan, number of voice mail messages, call durations and charges for different time periods and international calls, customer service calls, and churn status.
The dataset does not have any missing values, as indicated by the non-null counts.
The data types of the columns include bool, float64, int64, and object.
The bool column represents the churn status, indicating whether a customer discontinued the service (True) or not (False).
The float64 columns represent numerical values for call durations and charges.
The int64 columns represent numerical values for account length, area code, number of voice mail messages, call counts and customer service calls.
The object columns include state, phone number, international plan, and voice mail plan, which are categorical variables.
We then conducted an in-depth analyses and predictive modeling to tackle the issue of customer churn.

What Pacckages were required?
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from imblearn.pipeline import Pipeline as ImPipeline
from imblearn.over_sampling import SMOTE,SMOTENC
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_curve, auc, confusion_matrix

Exploratory Data Analysis and Data Cleaning
Exploratory data analysis consists of analyzing the main characteristics of a data set usually by means of visualization methods and summary statistics. The objective is to understand the data, discover patterns and anomalies, and check assumptions before performing further evaluations.

Steps followed

One-Hot Encoding
Normalization
Setting a baseline
Splitting the data in training and testing sets
Assessing multiple algorithms : In this project, we compare 3 different algorithms, all of them already implemented in Scikit-Learn.
Algorithm selected: Gradient Boosting
Hyperparameter tuning
Performace of the model

Drawing conclusions — Summary

In this post, we have walked through a complete end-to-end machine learning project using the Telco customer Churn dataset. We started by cleaning the data and analyzing. Then, to be able to build a machine learning model, we transformed the categorical data into numeric variables (feature engineering). After transforming the data, we tried 6 different machine learning algorithms using default parameters. Finally, we tuned the hyperparameters of the Gradient Boosting Classifier (best performance model) for model optimization, obtaining an accuracy of nearly 80% (close to 6% higher than the baseline).




