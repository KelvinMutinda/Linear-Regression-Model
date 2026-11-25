import utils.session
medical_change_url='https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv'

from urllib.request import urlretrieve

urlretrieve(medical_change_url,'medical.csv')


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

medical_df=pd.read_csv('medical.csv')

medical_df

medical_df.info()

#import jovian
#jovian.commit()

import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
matplotlib.rcParams['font.size']=14
matplotlib.rcParams['figure.figsize']=(10,6)
matplotlib.rcParams['figure.facecolor']='#00000000'

#shows disribution of ages on a histograph

"""
medical_df.age.describe()
fig=px.histogram(medical_df,x='age',marginal='box',nbins=47,title='Distribution of Age')
fig.update_layout(bargap=0.1)
fig.show()

fig=px.histogram(medical_df,x='bmi',marginal='box',nbins=47,title='Distribution of BMI')
fig.update_layout(bargap=0.1)
fig.show()

#show charges distribution
fig=px.histogram(medical_df,
                 x='charges',
                 marginal='box',
                 color='smoker',
                 color_discrete_sequence=['red','green'],
                 nbins=47,
                 title='Distribution of BMI')
fig.update_layout(bargap=0.1)
fig.show()

#relationship betweeen Age and Gharges

fig=px.scatter(medical_df,
                 x='age',
                 y='charges',
                 color='smoker',
                 hover_data=['sex'],
                 opacity=0.5,
                 title='Age vs Charges')

fig.update_traces(marker_size=5) 

fig.show()

"""
# Use numeric_only=True to ignore the text columns
print(medical_df.corr(numeric_only=True) )
print(medical_df['charges'].corr(medical_df['age']))


""""
# Filter the data
non_smoker_df = medical_df[medical_df.smoker == 'no'] 
#smoker_df = medical_df[medical_df.smoker == 'no']  
ages = non_smoker_df['age']
#ages = smoker_df['age']
target = non_smoker_df.charges

# Our manual function
def estimate_charges(ages, w, b):
    return ages * w + b

w = model.coef_[0]
b = model.intercept_
predicted_charges = estimate_charges(ages, w, b)



# --- PLOTTING TIME ---
plt.figure(figsize=(10, 6)) # Create a fresh, clean figure

# 1. Plot the ESTIMATE (Blue line)
plt.plot(ages, predicted_charges, color='blue', alpha=0.9, label='Estimate')

# 2. Plot the ACTUAL data (Orange dots)
plt.scatter(ages, target, s=8, color='orange', alpha=0.9, label='Actual')

# 3. Add labels and title
plt.xlabel('Age')
plt.ylabel('Charges')
plt.title('Age vs Charges')
plt.legend() # This will now perfectly match the colors

# 4. Show the plot
plt.show()


#RMSE ERROR CALCULATONS 
def rmse(targets,predicted_charges):
    return np.sqrt(np.mean(np.square(targets-predicted_charges)))

# Call the function and print the result
error = rmse(target, predicted_charges)
print(f"Root Mean Squared Error (RMSE): {error:.2f}")
# 1. Initialize the model
model = LinearRegression()

# 2. Fit the model to your data
# Note: Scikit-learn requires the input X to be a 2D array (a DataFrame), 
# which is why we use double brackets: non_smoker_df[['age']]
model.fit(non_smoker_df[['age']], target)

# 3. Get the computer's calculated w and b
perfect_w = model.coef_[0]
perfect_b = model.intercept_

print(f"Computer found w (slope): {perfect_w:.2f}")
print(f"Computer found b (intercept): {perfect_b:.2f}")

# 4. Make predictions using the model
predicted_charges = model.predict(non_smoker_df[['age']])
"""
# Filter the data
non_smoker_df = medical_df[medical_df.smoker == 'no'] 
ages = non_smoker_df['age']
target = non_smoker_df.charges

# --- 1. TRAIN THE MODEL FIRST ---
model = LinearRegression()
model.fit(non_smoker_df[['age','bmi']], target)

# Get the computer's calculated w and b
w1 = model.coef_[0]
w2= model.coef_[1]
b = model.intercept_

print(f"Computer found w (slope): {w1:.2f}")
print(f"Computer found w (slope): {w2:.2f}")
print(f"Computer found b (intercept): {b:.2f}")

# Make predictions using the model (this replaces your estimate_charges function)
predicted_charges = model.predict(non_smoker_df[['age','bmi']])

"""# --- 2. PLOTTING TIME ---
plt.figure(figsize=(10, 6)) 

# Plot the ESTIMATE (Blue line)
plt.plot(ages, predicted_charges, color='blue', alpha=0.9, label='Estimate')

# Plot the ACTUAL data (Orange dots)
plt.scatter(ages, target, s=8, color='orange', alpha=0.9, label='Actual')

plt.xlabel('Age and BMI')
plt.ylabel('Charges')
plt.title('Age vs Charges')
plt.legend() 
plt.show()
"""

"""
# --- 2. PLOTTING TIME ---
plt.figure(figsize=(10, 6)) 

# Scatter plot of Actual vs Predicted charges
plt.scatter(target, predicted_charges, s=8, color='blue', alpha=0.5, label='Predictions')

# Draw a perfect diagonal red line (if the model was 100% perfect, all dots would be on this line)
plt.plot([0, 60000], [0, 60000], color='red', linestyle='--', label='Perfect Prediction')

plt.xlabel('Actual Charges')
plt.ylabel('Predicted Charges')
plt.title('Actual vs Predicted Charges (Age & BMI)')
plt.legend() 
plt.show()
"""


"""
# --- 2. PLOTTING TIME ---
plt.figure(figsize=(10, 6)) 


plt.scatter(ages, target, s=8, color='orange', alpha=0.9, label='Actual')
plt.plot(ages, predicted_charges, color='blue', alpha=0.9, label='Predicted')

plt.xlabel('Age')
plt.ylabel('Charges')
plt.title('Age vs Charges (Predicted by Age & BMI)')
plt.legend() 
plt.show()
"""
# --- ONE-HOT ENCODING & FULL MODEL ---

# 1. Create a new dataframe with only the columns we need
medical_features = medical_df[['age', 'bmi', 'smoker']]

# 2. Apply One-Hot Encoding
# This will turn 'smoker' (yes/no) into 'smoker_yes' (1/0)
encoded_df = pd.get_dummies(medical_features, columns=['smoker'], drop_first=True)

# 3. Define Features (X) and Target (y)
X = encoded_df[['age', 'bmi', 'smoker_yes']] 
y = medical_df['charges']

# 4. TRAIN THE MODEL
model = LinearRegression()
model.fit(X, y)

# Get the weights
w_age = model.coef_[0]
w_bmi = model.coef_[1]
w_smoker = model.coef_[2] # The penalty for smoking!
b = model.intercept_

print(f"\nComputer found w1 (Age): {w_age:.2f}")
print(f"Computer found w2 (BMI): {w_bmi:.2f}")
print(f"Computer found w3 (Smoker penalty): {w_smoker:.2f}")
print(f"Computer found b (Intercept): {b:.2f}")

# Make predictions
predicted_charges = model.predict(X)

# 5. RMSE ERROR CALCULATIONS
def rmse(targets, predicted_charges):
    return np.sqrt(np.mean(np.square(targets - predicted_charges)))

error = rmse(y, predicted_charges)
print(f"\nRoot Mean Squared Error (RMSE): {error:.2f}")

# --- 6. PLOTTING TIME ---
plt.figure(figsize=(10, 6)) 

# 1. Plot the ACTUAL data (Scatter dots)
plt.scatter(medical_df['age'], y, s=8, color='orange', alpha=0.3, label='Actual Data')

# 2. Draw the PREDICTED LINES
# We use the weights (w_age, w_bmi, w_smoker, b) the model found!
# Let's assume an average BMI of 30 for our lines
average_bmi = 30

# Create a list of ages from 18 to 64 for our lines
age_range = np.array([18, 64])

# Calculate the line for NON-SMOKERS (smoker_yes = 0)
# Formula: Charges = (Age * w_age) + (BMI * w_bmi) + (0 * w_smoker) + b
line_non_smoker = (age_range * w_age) + (average_bmi * w_bmi) + b
plt.plot(age_range, line_non_smoker, color='green', linewidth=3, label='Predicted: Non-Smoker')

# Calculate the line for SMOKERS (smoker_yes = 1)
# Formula: Charges = (Age * w_age) + (BMI * w_bmi) + (1 * w_smoker) + b
line_smoker = (age_range * w_age) + (average_bmi * w_bmi) + w_smoker + b
plt.plot(age_range, line_smoker, color='red', linewidth=3, label='Predicted: Smoker')

# 3. Labels and Title
plt.xlabel('Age')
plt.ylabel('Charges')
plt.title('Actual Data vs Model Prediction Lines')
plt.legend() 
plt.show()


#Test the predections 
# ==========================================
# 🚀 INTERACTIVE PREDICTION (USER INPUT)
# ==========================================

print("\n" + "="*40)
print("🏥 MEDICAL CHARGE PREDICTOR")
print("="*40)

# 1. Get inputs from the user
try:
    user_age = int(input("Enter Age (e.g., 25): "))
    user_bmi = float(input("Enter BMI (e.g., 28.5): "))
    user_smoker_str = input("Is the person a smoker? (yes/no): ").strip().lower()

    # 2. Convert the text 'yes'/'no' into the 1/0 the model expects
    if user_smoker_str == 'yes' or user_smoker_str == 'y':
        smoker_val = 1
        smoker_text = "Smoker"
    else:
        smoker_val = 0
        smoker_text = "Non-Smoker"

    # 3. Create a DataFrame for the model
    user_data = pd.DataFrame({
        'age': [user_age],
        'bmi': [user_bmi],
        'smoker_yes': [smoker_val]
    })

    # 4. Make the prediction
    predicted_cost = model.predict(user_data)[0]

    # 5. Print the result nicely
    print("\n" + "-"*40)
    print(f"Patient Profile: {user_age} years old | BMI: {user_bmi} | {smoker_text}")
    print(f"💰 Predicted Medical Charges: ${predicted_cost:,.2f}")
    print("-"*40 + "\n")

except ValueError:
    print("\n❌ Error: Please enter a valid number for Age and BMI.")