# Imports
from EDA import df
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# CS473-2105A Data Mining
# Prof. Chintan Thakkar
# December 12, 2021
# Lauren White

# This program is being created to ingest the Titanic dataset,
# display details about each of the features, and apply a k-means
# clustering algorithm to predict survival.

# Creating feature and target arrays.
X = df.drop('survived', axis=1).values  # Features = everything that isn't survived
y = df['survived'].values  # Target = survived

# Scaling the data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Building the model
model = KMeans(n_clusters=2)
model.fit(X_scaled)

# Evaluate.

correct = 0

for i in range(len(X_scaled)):  # For every record in X_new
    predict = np.array(X_scaled[i].astype(float))
    predict = predict.reshape(-1, len(predict))
    prediction = model.predict(predict)  # Run predictions
    if prediction[0] == y[i]:  # If == to reality
        correct += 1  # Increase correct variable

print(correct / len(X_scaled))
