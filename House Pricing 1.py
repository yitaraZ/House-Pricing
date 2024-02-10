#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv('C:/Users/ASUS/OneDrive/เอกสาร/work/ml/Housing.csv')
print(df.head())
print("-------------------------------------")

#--------------------------------------------------

# cleaning data
label_encoder = LabelEncoder()
columns_to_encode = ['basement','furnishingstatus', 'guestroom', 'hotwaterheating', 'mainroad', 'airconditioning', 'prefarea']

# Loop through each column and encode
for column in columns_to_encode:
    df[column + '_en'] = label_encoder.fit_transform(df[column])

# Drop the original categorical columns
df.drop(columns=columns_to_encode, inplace=True)

print(df.head(10))

#--------------------------------------------------

# Separate data and target columns
X = df.drop(columns='price',axis=1)
y = df['price']

# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)

#--------------------------------------------------

#Linear Regression

le = LinearRegression()
le.fit(X_train,y_train)

y_pred = le.predict(X_test)
msele = mean_squared_error(y_test, y_pred)

print("LinearRegression MSE: ", msele)

#--------------------------------------------------

sv = SVC()
sv.fit(X_train,y_train)

y_predsvc = sv.predict(X_test)
msesv = mean_squared_error(y_test, y_predsvc)

print("SVC MSE: ", msesv)

#--------------------------------------------------

de = DecisionTreeClassifier()
de.fit(X_train,y_train)

y_predde = de.predict(X_test)
msede = mean_squared_error(y_test, y_predde)

print("DecisionTreeClassifier MSE: ", msede)

#--------------------------------------------------

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)

y_predknn = knn.predict(X_test)
mseknn = mean_squared_error(y_test, y_predknn)

print("KNeighborsClassifier MSE: ", mseknn)

#--------------------------------------------------

mse = [msele, msesv, msede, mseknn]
min_value = min(mse)
print(min_value)

#--------------------------------------------------

new_data = pd.DataFrame({
    'area': [7000],  # Replace with the appropriate values
    'bedrooms': [1],
    'bathrooms': [1],
    'stories': [2],
    'mainroad': [1],
    'guestroom_en': [1],
    'basement_en': [0],
    'hotwaterheating_en': [0],
    'airconditioning_en': [1],
    'furnishingstatus_en ': [0],
    'parking_en': [1],
    'prefarea_en': [0]
    # Add more columns as needed, make sure the column names match your feature names
})
predicted_prices = le.predict(new_data)

print("Predicted Prices:", predicted_prices)


# In[ ]:




