# Spotify and Youtube dataset.       
# Link: https://www.kaggle.com/datasets/salvatorerastelli/spotify-and-youtube

import pandas as pd

df = pd.read_csv('Spotify_Youtube.csv')     

# Remove unused columns.
df = df.drop(['Unnamed: 0', 'Url_spotify', 'Uri', 'Url_youtube', 'Description'], axis=1)
# Fill missing values with the mean of the columns.
df = df.fillna(df.mean())
# Remove duplicates.
df= df.drop_duplicates()

print('5. Bonus: Build a linear regression model.')
print()

print(f'a. Find feature variable, which is to find which column is the most correlated to Views column.')
print()

import matplotlib.pyplot as plt
import seaborn as sns

grouped = df.groupby(['Likes', 'Comments', 'Stream'], as_index=False)['Views'].sum()

# Visualize the data using scatter plots.
sns.pairplot(grouped, x_vars=['Likes', 'Comments', 'Stream'], y_vars='Views', height=4, aspect=1, kind='scatter')
plt.show()
print()

# Visualize the data using heatmap.
sns.heatmap(grouped.corr(), cmap="YlGnBu", annot = True)
plt.show()
print()

print(f'As we can see from the above graphs, the Likes column seems most correlated to Views column (0.89).')
print()
print(f'Therefore, let use the Likes column as feature variable.')
print()

print('b. Create training set and validation set.')
print()

# Creating X and y.
X = grouped['Likes']   # Feature variable.
y = grouped['Views']

# Splitting the varaibles as training and validation sets.
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size = 0.7, test_size = 0.3, random_state = 100)

print(f'X_train:')
print(X_train)
print()
print(f'y_train:')
print(y_train)
print()

print('c. Find and then visualize the regression line.')
print()

# Add additional column to the train and test sets.
X_train = X_train.values.reshape(-1,1)
X_valid = X_valid.values.reshape(-1,1)

print(f' The shape of X_train is: {X_train.shape}')
print(f' The shape of X_valid is: {X_valid.shape}')
print()

from sklearn.linear_model import LinearRegression

# Create an object of Linear Regression.
lm = LinearRegression()

# Fit the model using .fit() method.
lm.fit(X_train, y_train)

# Intercept value.
print("Intercept:",lm.intercept_)

# Slope value.
print('Slope:',lm.coef_)

print('Hence, the regression line is:')
print('Views = 1822709 + 141.4886 * Likes')
print()

# Visualize the regression line.
plt.scatter(X_train, y_train)
plt.plot(X_train, 1822709 + 141.4886*X_train, 'r')

plt.title('Likes and Views Linear Regression', fontsize=18, fontweight='bold')
plt.xlabel('Likes', fontsize=14, fontweight='bold')
plt.ylabel('Views', fontsize=14, fontweight='bold')

plt.show()
print()

print('d. Find MSE of the model.')
print()

# Make predictions of y_value.
y_train_pred = lm.predict(X_train)
y_valid_pred = lm.predict(X_valid)

from sklearn.metrics import mean_squared_error

# MSE = Mean Squared Error
mse_train = mean_squared_error(y_train,y_train_pred)
mse_valid = mean_squared_error(y_valid,y_valid_pred)

print(f'MSE of training set is: {mse_train: .5f}')
print(f'MSE of valid set is: {mse_valid: .5f}')
print()

# Visualize the line on the validation set with the y predictions.
plt.scatter(X_valid, y_valid)
plt.plot(X_valid, y_valid_pred, 'r')
plt.show()
