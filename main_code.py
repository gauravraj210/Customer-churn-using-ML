# Import libraries and methods/functions
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression


# load the dataFrames
df1 = pd.read_csv("telecom_demographics.csv")
df2 = pd.read_csv("telecom_usage.csv")

# joining the two datafrmes
df = df1.merge(df2, on="customer_id")

#finding the chrun rate 
churn_rate = df['churn'].value_counts() / len(df)

# to understand the categorical variables 
df.info()

# Future encoding
df['gender'] = df['gender'].replace({'M':0,'F':1})

#one hot encoding some variables 
df = pd.get_dummies(df, columns=['telecom_partner','state','city','registration_event'])

# Feature Scaling
scaler = StandardScaler()

# 'customer_id' is not a feature
features = df.drop(['customer_id', 'churn'], axis=1)
features_scaled = scaler.fit_transform(features)

# Target variable
target = df['churn']

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

# Instantiate the Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Logistic Regression predictions
logreg_pred = logreg.predict(X_test)
print(y_test)

# Logistic Regression evaluation
print(confusion_matrix(y_test, logreg_pred))
print(classification_report(y_test, logreg_pred))

# Instantiate the Random Forest model
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# Random Forest predictions
rf_pred = rf.predict(X_test)

# Random Forest evaluation
print(confusion_matrix(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

