#Import Libraries and Load the Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the dataset
url = "D:/Job Prep/Ericsson Data Analyst/Call Center Data/Call Center Data.csv"
df = pd.read_csv(url)

# Display the first few rows of the dataset
print(df.head())
# Drop Index column
df = df.drop('Index', axis=1)
#Data Preprocessing
# Check for missing values
print(df.isnull().sum())

#Visualize Data Distribution 

import matplotlib.pyplot as plt
import seaborn as sns

# Plot distribution of incoming calls
plt.figure(figsize=(10,6))
sns.histplot(df['Incoming Calls'], kde=True)
plt.title("Distribution of Incoming Calls")
plt.show()

#Exploratory Data Analysis (EDA)
#Visualize the Data: Plot graphs to identify any trends or correlations between variables.
# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Plot Incoming Calls vs. Answered Calls
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Incoming Calls', y='Answered Calls')
plt.title("Incoming Calls vs. Answered Calls")
plt.show()

#Check for Outliers: You can use box plots to identify any outliers that could affect the model.
# Boxplot for Incoming Calls
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['Incoming Calls'])
plt.title("Boxplot for Incoming Calls")
plt.show()

#Prepare the Data for Modeling
#Feature Selection: Since we’re predicting resource allocation (e.g., number of agents),
#we can focus on features like Incoming Calls, Answered Calls, Service Level, and Waiting Time.
# Select relevant features
X = df[['Incoming Calls','Answer Rate','Abandoned Calls','Answer Speed (AVG)','Talk Duration (AVG)','Waiting Time (AVG)','Service Level (20 Seconds)']]
y = df['Answered Calls']  
X = X.apply(pd.to_numeric, errors='coerce')
y = y.apply(pd.to_numeric, errors='coerce')
X.fillna(0, inplace=True)
y.fillna(0, inplace=True)

#Train-Test Split: Split the data into training and testing sets.
# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Model Building
#We'll start with a simple Linear Regression model to predict the number of answered calls based on the selected features.
# Initialize the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

#Model Evaluation
# Calculate MAE and RMSE
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

#Visualization of Predictions
# Visualize predictions vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Answered Calls')
plt.ylabel('Predicted Answered Calls')
plt.title('Actual vs Predicted Answered Calls')
plt.show()

residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.xlabel('Predicted Answered Calls')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.axhline(y=0, color='r', linestyle='--')
plt.show()



from scipy.optimize import linprog

# Step 1: Define costs per agent type
agent_costs = [10, 12, 15]  # Cost per agent for 3 types of shifts

# Step 2: Define constraints
# Rows = constraints, Columns = agent types
A = [
    [1, 1, 1],  # Total agents needed
    [2, 1, 0],  # Peak hour constraint (e.g., more morning agents)
]
b = [50, 70]  # [Total demand, Peak hour demand]

# Step 3: Define bounds for variables (e.g., no negative agents)
x_bounds = [(0, None) for _ in agent_costs]

# Step 4: Solve optimization problem
result = linprog(
    c=agent_costs,      # Minimize total cost
    A_ub=A,             # Coefficients for constraints
    b_ub=b,             # Right-hand side of constraints
    bounds=x_bounds,    # Bounds for variables
    method="highs"      # Solver method
)

# Step 5: Display results
if result.success:
    print("Optimized Number of Agents (by type):", result.x)
    print("Minimum Total Cost:", result.fun)
else:
    print("Optimization failed.")


#Create an Interactive Dashboard
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linprog

# Function to optimize resource allocation
def optimize_allocation(total_demand, peak_demand):
    agent_costs = [10, 12, 15]
    A = [[1, 1, 1], [2, 1, 0]]
    b = [total_demand, peak_demand]
    x_bounds = [(0, None) for _ in agent_costs]

    result = linprog(c=agent_costs, A_ub=A, b_ub=b, bounds=x_bounds, method="highs")
    return result

# Streamlit Dashboard
st.title("Resource Allocation Optimization Dashboard")

# User inputs
st.sidebar.header("Scenario Parameters")
total_demand = st.sidebar.slider("Total Call Volume Demand", min_value=30, max_value=100, value=50)
peak_demand = st.sidebar.slider("Peak Hour Demand", min_value=20, max_value=80, value=70)

# Optimize allocation
result = optimize_allocation(total_demand, peak_demand)

if result.success:
    st.subheader("Optimized Resource Allocation")
    agent_types = ["Morning", "Afternoon", "Evening"]
    optimized_agents = result.x
    costs = [10, 12, 15]
    total_cost = result.fun

    # Display results in a table
    df = pd.DataFrame({
        "Agent Type": agent_types,
        "Number of Agents": optimized_agents,
        "Cost per Agent": costs,
        "Total Cost": [n * c for n, c in zip(optimized_agents, costs)]
    })
    st.table(df)

    # Visualize allocation
    fig, ax = plt.subplots()
    ax.bar(agent_types, optimized_agents, color="skyblue")
    ax.set_title("Optimized Number of Agents by Type")
    ax.set_xlabel("Agent Type")
    ax.set_ylabel("Number of Agents")
    st.pyplot(fig)

    st.write(f"Minimum Total Cost: {total_cost}")
else:
    st.error("Optimization failed.")
