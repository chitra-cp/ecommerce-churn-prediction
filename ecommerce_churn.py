#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("C:Users/chitr/OneDrive/Documents/Desktop/CHITRA/BA/ecommerce_churn.csv")


# In[3]:


df.head()


# In[4]:


df.dropna(inplace=True)
df.isnull().sum()


# In[5]:


#Checking if there are any duplicated rows in the table

df.duplicated().sum()
print("Duplicate rows: " + str(df.duplicated().sum()))


# In[6]:


# Replace values in multiple columns
df['PreferredLoginDevice'] = df['PreferredLoginDevice'].replace(['Phone', 'Mobile Phone'], 'Mobile')

df['PreferedOrderCat'] = df['PreferedOrderCat'].replace('Mobile Phone', 'Mobile')

df['PreferredPaymentMode'] = df['PreferredPaymentMode'].replace({
    'COD': 'Cash on Delivery',
    'CC': 'Credit Card'
})


# In[7]:


print(df['PreferredLoginDevice'].value_counts())
print()
print(df['PreferedOrderCat'].value_counts())
print()
print(df['PreferredPaymentMode'].value_counts())


# In[10]:


# Set style and context
sns.set_style('white')
sns.set_context('talk')

# Create the plot
plt.figure(figsize=(5,5))
ax = sns.countplot(x=df['PreferedOrderCat'], palette="pastel", edgecolor='black')

# Add title and axis labels
plt.title('Customer Preferred Order Category', fontsize=16, weight='bold')
plt.xlabel('Preferred Order Category', fontsize=14, weight='bold')
plt.ylabel('Count', fontsize=14, weight='bold')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, fontsize=12)

# Annotate bars with the count values
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=12, weight='bold')

# Add grid lines for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Display the plot
plt.tight_layout()
plt.show()


# In[9]:


# Set the style and context for better visualization
sns.set_style("whitegrid")
sns.set_context("talk")

# Create the plot with customizations
plt.figure(figsize=(12, 6))

# Using a vibrant palette and black edge for clarity
ax = sns.countplot(x=df['PreferredPaymentMode'], palette="coolwarm", edgecolor='black')

# Add a title and axis labels with more styling
plt.title('Customer Preferred Payment Mode', fontsize=20, weight='bold', color='darkblue')
plt.xlabel('Preferred Payment Mode', fontsize=14, weight='bold')
plt.ylabel('Count', fontsize=14, weight='bold')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, fontsize=12)

# Annotate bars with the count values
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', xytext=(0, 8), textcoords='offset points', fontsize=12)

# Remove top and right spines for a cleaner look
sns.despine()

# Add a grid for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Tight layout for better spacing
plt.tight_layout()

# Display the plot
plt.show()


# In[10]:


# Aggregate the data
device_counts = df['PreferredLoginDevice'].value_counts()

# Create a pie chart
plt.figure(figsize=(5,5))
plt.pie(device_counts, labels=device_counts.index, autopct='%1.1f%%', colors=sns.color_palette("pastel"),
        startangle=140, wedgeprops=dict(edgecolor='black'))

# Add title
plt.title('Customer Preferred Login Device', fontsize=16, weight='bold')

# Display the plot
plt.tight_layout()
plt.show()


# In[11]:


# Custom color palette based on the uploaded image
custom_palette = ['#4B3B71', '#3A6F7A', '#32937F', '#3CB44B', '#A4D06A']

# Group the data by 'PreferedOrderCat' and sum the 'Num_of_Purchases'
order_category_data = df.groupby('PreferedOrderCat')['Num_of_Purchases'].sum().reset_index()

# Create the plot
plt.figure(figsize=(12, 8))

# Plot the bar chart with the custom color palette and edge color
barplot = sns.barplot(
    x='PreferedOrderCat', 
    y='Num_of_Purchases', 
    data=order_category_data, 
    palette=custom_palette,  # Use the custom color palette
    edgecolor='black'        # Add a black edge to each bar
)

# Customize the plot
plt.title('Total Number of Purchases by Preferred Order Category', fontsize=20, fontweight='bold', color='darkblue')
plt.xlabel('Preferred Order Category', fontsize=14, fontweight='bold')
plt.ylabel('Number of Purchases', fontsize=14, fontweight='bold')

# Rotate the x-axis labels for better readability
plt.xticks(rotation=45, fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')

# Tight layout for better spacing
plt.tight_layout()

# Show the plot
plt.show()


# In[12]:


# Set Seaborn theme for better aesthetics
sns.set_theme(style="whitegrid")

# Grouping data by 'Age' and calculating the mean of 'HourSpendOnApp'
age_group_data = df.groupby('Age')['HourSpendOnApp'].mean()

# Plotting the data
plt.figure(figsize=(12, 7))
barplot = sns.barplot(x=age_group_data.index, y=age_group_data.values, palette='coolwarm')

# Adding labels and title with custom fonts and sizes
plt.xlabel('Age Range', fontsize=14, fontweight='bold')
plt.ylabel('Average Hours Spent', fontsize=14, fontweight='bold')
plt.title('Average Hours Spent by Age Range', fontsize=16, fontweight='bold')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, fontsize=12)

# Add gridlines
plt.grid(True, axis='y', linestyle='--', linewidth=0.6)

# Adding a tight layout for proper spacing
plt.tight_layout()

# Show the plot
plt.show()


# In[13]:


top_10_customers = df.nlargest(10, 'Total_Spend')[['CustomerID', 'Total_Spend']]

plt.figure(figsize=(8, 6))
plt.scatter(top_10_customers['CustomerID'], top_10_customers['Total_Spend'], color='skyblue')
plt.title('Top 10 Customers by Total Spend')
plt.xlabel('CustomerID')
plt.ylabel('Total Spend')
plt.xticks(rotation=45)
plt.show()


# In[31]:


# Plotting a histogram for 'Last_Purchase_Days_Ago'
plt.figure(figsize=(10, 6))
plt.hist(df['Last_Purchase_Days_Ago'].dropna(), bins=30, edgecolor='black', color='skyblue')
plt.title('Distribution of Last Purchase Days Ago', fontsize=16)
plt.xlabel('Days since Last Purchase', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# In[8]:


names = ['Yes', 'No']
values = df['Complain'].value_counts()

colors = ['#32937F','#3A6F7A'] 
plt.bar(names, values, color=colors)
plt.title("Any complaint has been raised in last month", fontsize=15)
plt.show()


# In[9]:


# Drop unnecessary columns
df.drop(['CustomerID'], axis=1, inplace=True)


# In[10]:


from sklearn.preprocessing import LabelEncoder

# Encode categorical columns using LabelEncoder
cat_cols = df.select_dtypes(include=['object', 'bool']).columns
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])


# In[11]:


# Splitting data into features and target
X = df.drop(columns='Churn') 
y = df['Churn']              


# In[12]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Feature Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Handle imbalanced data using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split the resampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Hyperparameter tuning with GridSearchCV for Random Forest
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Initialize RandomForestClassifier
rf = RandomForestClassifier(random_state=42)

# Perform GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Retrieve the best model from GridSearchCV
best_rf = grid_search.best_estimator_

# Generate predictions on the test set
y_pred = best_rf.predict(X_test)


# In[13]:


from sklearn.metrics import accuracy_score, classification_report

# Calculate accuracy and classification report
accuracy = accuracy_score(y_test, y_pred)
print(f'Optimized Model Accuracy: {accuracy * 100:.2f}%')
print("Classification Report:\n", classification_report(y_test, y_pred))


# In[14]:


from sklearn.metrics import roc_auc_score, log_loss

# Assuming it's a binary classification
y_pred_proba = best_rf.predict_proba(X_test)

# Roc-AUC Score: for binary classification, we take the probabilities of the positive class
roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])

# Log loss: it can handle the full probability array (for binary or multiclass)
logloss = log_loss(y_test, y_pred_proba)

print("Roc-Auc Score:", roc_auc)
print("Log loss:", logloss)


# In[21]:


from sklearn.metrics import confusion_matrix

# Create the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()


# In[24]:


new_data=[[25,0,10.1,1,4000.00,200.0,1,5,4,0,30,5,5,3,2,0,100,200,1,2]]
predicted_class=best_rf.predict(new_data)
print("prediction:",predicted_class)


# In[ ]:




