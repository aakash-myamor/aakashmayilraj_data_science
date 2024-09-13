# Importing the needed libraries
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

# My Project Name : Company Bankruptcy Prediction 

# To Read the dataset
# =======================================================

Company_Bankrupt_data = pd.read_csv("data.csv")

warnings.filterwarnings("ignore")

# To Display option to show rows and columns
# =======================================================
"""
pd.set_option("display.max_columns", None)
print("Display maximum columns")
print(Company_Bankrupt_data)

pd.set_option('display.max_rows', None)
print("Display maximum rows")
print(Company_Bankrupt_data)

pd.set_option('display.width', None)
print("Display.width")
print(Company_Bankrupt_data)

pd.set_option('display.max_colwidth', None)
print("Display.max_colwidth")
print(Company_Bankrupt_data)
"""
# Exploratory data analysis
# =======================================================

# To check correlation between feature and targeted data using correlation matrix

cor_matx = Company_Bankrupt_data.corr()
plt.figure(figsize=(10,5))
sns.heatmap(cor_matx, cmap="coolwarm")
plt.title("Correlation Head map ")
plt.show()

# To print first five rows 

print("To print first five rows")
print(Company_Bankrupt_data.head())


# To print last five rows

print("To print last five rows")
print(Company_Bankrupt_data.tail())

# To get the quick summery of a dataframe

print("To get the quick summery of a dataframe")
print(Company_Bankrupt_data.info())

# To count no of rows and columns

print("To count no of rows and columns")
print(Company_Bankrupt_data.shape)

# To get a quick overview of data in a DataFrame

print("To get a quick overview of data in a DataFrame")
print(Company_Bankrupt_data.describe())

# To returns the count of True values which actually corresponds to the number of NaN values

print("To returns the count of True values which actually corresponds to the number of NaN values")
print(Company_Bankrupt_data.isnull().sum())

# To know the details of the columns

print("To know the details of the columns")
print(Company_Bankrupt_data.columns)

# To count the number of NaN values in a specific column in a Pandas DataFrame,

print("To count the number of NaN values in a specific column in a Pandas DataFrame")
print(Company_Bankrupt_data.isna().sum())

# To check all of our columns and return True if there are any missing values or NaN's

print("check all of our columns and return True if there are any missing values or NaN's")
print(Company_Bankrupt_data.isna().any())

# To find data frame have any duplicates of elements with smaller subscript

print("To find data frame have any duplicates of elements with smaller subscript")
print(Company_Bankrupt_data.duplicated().any())

# To find and return a list of unique values in a range or list.

print(" To find and return a list of unique values in a range or list.")
print(Company_Bankrupt_data['Bankrupt?'].unique())

# Machine Learning to Split the data for training and testing
# =============================================================

# Distribution of bankrupt column

targeted_data = "Bankrupt?"
x = Company_Bankrupt_data.drop(columns=[targeted_data])
y = Company_Bankrupt_data[targeted_data]

print("x.shape", x.shape)
print("y.shape", y.shape)

# Machine learning through train_test_split model selection

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,shuffle=True, random_state=42)
shapes = {'x_train': x_train.shape[0],
         'y_train': y_train.shape[0],
         'x_test': x_test.shape[0],
         'y_test': y_test.shape[0]
         }
plt.figure(figsize=(15, 6))
plt.bar(shapes.keys(), shapes.values())
plt.xlabel("Company Bankruptcy dataset")
plt.ylabel("Number of instance ")
plt.title("Distribution of training and validation")
plt.show()

# scaling the data with  StandardScaler, MinMaxScaler, LabelEncoder,OneHotEncoder.

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

# Logistic Regression

log_model = LogisticRegression()
log_model.fit(x_train,y_train)
y_prediction = log_model.predict(x_test)
print(y_prediction)

confu = confusion_matrix(y_test,y_prediction)
sns.heatmap(confu, annot=True, cmap='viridis', cbar=True)
plt.show()

print("_Classification Report_")
print("classification_report is ", classification_report(y_test ,y_prediction))

# Decision Tree( DecisionTreeClassifier )

decision_tree = DecisionTreeClassifier()

params = {
    'criterion' : ['gini', 'entropy'],
    'max_depth' : [5, 10, 15, 20, 25],
    'min_samples_leaf' : [10, 20, 30, 40, 50]}
grid_search_model = GridSearchCV(decision_tree, params, cv=5)
grid_search_model.fit(x_train, y_train)

print(grid_search_model.best_params_)
print(grid_search_model.best_score_)

dec_tree = DecisionTreeClassifier(criterion='gini', max_depth=5, min_samples_leaf=30 )
dec_tree.fit(x_train,y_train)

decision_tree_prediction = dec_tree.predict(x_test)
print("decision_tree_prediction")
print(decision_tree_prediction)


# Random forest RandomForestClassifier

random_forest_classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
random_forest_classifier.fit(x_train, y_train)
y_prediction = random_forest_classifier.predict(x_test)

conf_matx = confusion_matrix(y_test, y_prediction)
print("confusion_matrix")
print(conf_matx)

acc_sco_matx = accuracy_score(y_test, y_prediction)
print("accuracy_score")
print(acc_sco_matx)

# PCA Principal component analysis

pca = PCA(n_components=2)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
print("Principal component analysis")
print(x_train.shape,x_test.shape)

# visualization of data to check whether the distribution is skewed significantly or not
# =========================================================================================

# histogram visualisation

Company_Bankrupt_data[" Net Income to Total Assets"].hist()
plt.xlabel(" Net Income to Total Assets")
plt.ylabel("counts")
plt.title("Distribution of net income to total assets ratio ")
plt.show()

# To calculate the relative frequencies of the classes

sns.displot(x='Bankrupt?', data=Company_Bankrupt_data, color='blue')
plt.xlabel("Bankruptcy Classes")
plt.ylabel("Frequency")
plt.title("Classes balancing")
plt.show()

# Distribution of Profit / Net Income by ratios

sns.boxenplot(x="Bankrupt?", y=" Net Income to Total Assets", data=Company_Bankrupt_data, color="blue")
plt.xlabel("Bankruptcy Classes")
plt.ylabel("Net Income to Total Assets")
plt.title("Distribution of net asset with classes ")
plt.show()

# graph of all the machine learning model

mylist=[]
mylist2=[]
mylist.append(confu)
mylist2.append("Logistic Regression")
mylist.append(conf_matx)
mylist2.append("RandomForestClassifier 1")
mylist.append(acc_sco_matx)
mylist2.append("RandomForestClassifier 2")
plt.rcParams['figure.figsize']=22,10
sns.set_style("darkgrid")
ax = sns.barplot(x={mylist2.all}, y={mylist.all}, palette = "coolwarm", saturation =1.5)
width, height = p.get_width(), p.get_height()
x, y = p.get_xy()
ax.annotate(f'{height:.2%}', (x + width/2, y + height*1.02), ha='center', fontsize = 'x-large')
plt.xlabel("Classification Models", fontsize = 20 )
plt.ylabel("Accuracy", fontsize = 20)
plt.title("Accuracy of different Classification Models", fontsize = 20)
plt.xticks(fontsize = 11, horizontalalignment = 'center', rotation = 8)
plt.yticks(fontsize = 13)
plt.show()


