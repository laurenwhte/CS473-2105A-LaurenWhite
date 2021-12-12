# Imports
from helpers import null_values, unique_values, value_range
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# CS473-2105A Data Mining
# Prof. Chintan Thakkar
# December 12, 2021
# Lauren White

# This program is being created to ingest the Titanic dataset,
# display details about each of the features, and apply a k-means
# clustering algorithm to predict survival.

# Load the dataset into pandas and confirm it loaded.
df = pd.read_excel('CS473_Titanic_data.xls', header=0)
print(df.head())

# INITIAL EXPLORATION
print("Confirmation the data loaded: \n")
print(df.head())
print("\nInfo about the dataset: \n")
print(df.info())
null_values(df)
unique_values(df)
value_range(df)

# TARGET
# We are looking for survivability, so I want to move our target column
# to the end just for formatting.
df = df[['pclass', 'name', 'sex', 'age', 'sibsp', 'parch', 'ticket', 'fare',
         'cabin', 'embarked', 'boat', 'body', 'home.dest', 'survived']]

# PRE-PROCESSING
# From the EDA I can see that we need to handle the null values and convert the object types to
# numeric ones before we move forward.

# Handling the nulls - these are some large % of missing items. I think it would be fair to fill in age with
# the most common value, but the others should be dropped.
#            Total     %
# body        1188  90.8
# cabin       1014  77.5
# boat         823  62.9
# home.dest    564  43.1
# age          263  20.1


# Find new values for age nulls.
print("\nValue counts for age: \n")
print(df.age.value_counts())
print("\nPandas description for age column:\n")
print(df.age.describe())

# Fill null ages with most common value.
df['age'] = df['age'].fillna(df['age'].value_counts().index[0])
# Verify it worked by rechecking the null values.
null_values(df)

# Drop the columns with too many missing values.
df = df.drop(['body', 'cabin', 'boat', 'home.dest'], axis=1)
# Verify it worked by rechecking the null values.
null_values(df)

# Now we have only a small amount of nulls to replace.
print("\nValue counts for embarked: \n")
print(df.embarked.value_counts())

# Fill in missing data points with S and check to make sure it was done.
df['embarked'].fillna('S', inplace=True)
is_done = df['embarked'].isnull().sum()
print('Null count: ', is_done, '\n')

# For the missing fare values, we may be able to look at the records' class to determine a reasonable replacement
# value for fare.
# Pivot on the class to show average fare by class ticket.
print('Average fare by pclass: \n')
print(df.pivot_table(index='pclass', values='fare', aggfunc='mean', fill_value=0), '\n')

# Look at the record with missing fare to see what class it is.
df1 = df[df.isna().any(axis=1)]
print(df1, '\n')

# Replace missing fare amount with mean of 3rd class fare. Also, converting fares to int64.
df['fare'].fillna(13.30, inplace=True)
df['fare'] = df['fare'].astype('int64')
is_done = df['fare'].isnull().sum()
print('Null count: ', is_done, '\n')

# Recheck all columns now to make sure all NaN values are handled by calling the null_values() function again.
null_values(df)

# ENCODING - Now that all null values have been removed or replaced, we need ot encode the remaining
# object features so that an ML algorithm can read them.

# Replace female and male with 0 and 1 - needs to be numbers to be used in linear regression.
df['sex'] = df['sex'].replace(['female'], 0)
df['sex'] = df['sex'].replace(['male'], 1)
df.info()

# It looks like we could convert the remaining object column values to a numeric, so I will use the
# categories data type to do so.
cols = df.select_dtypes('object').columns
df[cols] = df[cols].stack().astype('category').cat.codes.unstack()
df = df.astype('float64')  # Convert all to one dtype
print('\nConfirmation of changed types: \n')
df.info()

# VISUALIZATION -
# Let's make a heatmap using the Pearson's correlation co-efficient.
# This is a measure of the linear correlation between two sets of data.
corrPearson = df.corr(method="pearson")

figure = plt.figure(figsize=(12, 12))
sns.heatmap(corrPearson, annot=True, cmap='RdYlGn', vmin=-1, vmax=+1)
plt.xlabel("Features")
plt.ylabel("Features")
plt.savefig('pearsonCorrelation.png')
plt.show()


# Key:
# 1 = positive correlation, when one value increases so does the other.
# -1 = negative correlation, when one variable increases the other decreases.
# 0 = No correlation, the variables change without impacting each other.

# We can see from the plot that there is a negative correlation with class and sex,
# which means the higher those numbers (1 being male, 3 being third class) the less
# likely that person was to survive. There is a positive correlation with fare, which
# means the higher the fare, the more likely the person was to survive.

# There is almost no correlation between survival and name or ticket, so I am
# dropping those features as well.
df = df.drop(['name', 'ticket'], axis=1)
print('\nConfirmation of final dataset: \n')
print(df.info())

# I think we are ready to move on to the model now.


