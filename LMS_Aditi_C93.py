# Open Sublime text editor, create a new Python file, copy the following code in it and save it as 'penguin_app.py'.

# Importing the necessary libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier

# Load the DataFrame
csv_file = 'penguin.csv'
df = pd.read_csv(csv_file)

# Display the first five rows of the DataFrame
df.head()

# Drop the NAN values
df = df.dropna()

# Add numeric column 'label' to resemble non numeric column 'species'
df['label'] = df['species'].map({'Adelie': 0, 'Chinstrap': 1, 'Gentoo':2})


# Convert the non-numeric column 'sex' to numeric in the DataFrame
sex_dict = {'Male':0,'Female':1}
df['sex'] = df['sex'].map(sex_dict)

# Convert the non-numeric column 'island' to numeric in the DataFrame
island_dict = {'Biscoe': 0, 'Dream': 1, 'Torgersen':2}
df['island'] = df['island'].map(island_dict)


# Create X and y variables
X = df[['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]
y = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)


# Build a SVC model using the 'sklearn' module.
svc_model = SVC(kernel = 'linear')
svc_model.fit(X_train, y_train)
svc_score = svc_model.score(X_train, y_train)

# Build a LogisticRegression model using the 'sklearn' module.
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg_score = log_reg.score(X_train, y_train)

# Build a RandomForestClassifier model using the 'sklearn' module.
rf_clf = RandomForestClassifier(n_jobs = -1)
rf_clf.fit(X_train, y_train)
rf_clf_score = rf_clf.score(X_train, y_train)

#Prediction function
def prediction(model,island,bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g,sex):
	species = model.predict([[island,bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g,sex]])[0]
	if species == 0:
		species = "Adelie"
	elif species == 1:
		species = "Chinstrap"
	else:
		species = "Gentoo"
	return species

#UI Design
st.title("Penguin Species Classification")

st.sidebar.title("Select Features:")
bill_length = st.sidebar.slider("Bill Length in mm", float(df['bill_length_mm'].min()), float(df['bill_length_mm'].max()))
bill_depth = st.sidebar.slider("Bill Depth in mm", float(df['bill_depth_mm'].min()), float(df['bill_depth_mm'].max()))
flipper_length = st.sidebar.slider("Flipper Length in mm", float(df['flipper_length_mm'].min()), float(df['flipper_length_mm'].max()))
body_mass = st.sidebar.slider("Body Mass in gms", float(df['body_mass_g'].min()), float(df['body_mass_g'].max()))

sex = st.sidebar.selectbox("Sex",list(sex_dict.keys()))
island = st.sidebar.selectbox("Island",list(island_dict.keys()))

sex = sex_dict.get(sex)
island = island_dict.get(island)

classifier = st.sidebar.selectbox("Select Classifier",['Random Forest Classifier','Logistic Regression','Support Vector Classifier'])

if st.sidebar.button("Predict"):
	if classifier == "Random Forest Classifier":
		predicted = prediction(rf_clf,island,bill_length,bill_depth,flipper_length,body_mass,sex)  
		accuracy = rf_clf_score
	elif classifier == "Logistic Regression":
		predicted = prediction(log_reg,island,bill_length,bill_depth,flipper_length,body_mass,sex)
		accuracy = log_reg_score
	else:
		predicted = prediction(svc_model,island,bill_length,bill_depth,flipper_length,body_mass,sex)
		accuracy = svc_score

	st.write(f"Predicted species of penguin: {predicted}")
	st.write(f"Accuracy score: {round(accuracy,2)}")