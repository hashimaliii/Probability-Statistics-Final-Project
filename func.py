# Import necessary libraries for data manipulation, statistical analysis, and visualization
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import scipy.stats as stats

# Import machine learning libraries from sklearn for regression analysis
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

# Import train_test_split for splitting the dataset into training and testing sets
from sklearn.model_selection import train_test_split

# Import poisson from scipy.stats for Poisson distribution
from scipy.stats import poisson

# Import LabelEncoder for encoding categorical variables
from sklearn.preprocessing import LabelEncoder

# Import statsmodels for building and analyzing statistical models
import statsmodels.api as sm

# Import stats from scipy for statistical functions
from scipy import stats

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Function to load data from a CSV file located at a specific URL
def load_data_from_github():
    url = 'https://raw.githubusercontent.com/hashimaliii/Probability-Statistics-Final-Project/main/dataset/Employee_Salaries.csv'
    # Return the loaded data as a pandas DataFrame
    return(pd.read_csv(url))

# Function to load data from a CSV file into a DataFrame
def load_data_from_csv():
    # Use pandas read_csv function to read the CSV file
    # and return the resulting DataFrame
    return pd.read_csv('dataset/Employee_Salaries.csv')

# Function to add a new record to the DataFrame
def add_record(df):
    # Prompt the user for input for each field in the record
    Department = input("Enter Department: ")
    Department_Name = input("Enter Department Name: ")
    Division = input("Enter Division: ")
    Gender = input("Enter Gender (M/F): ")
    Base_Salary = float(input("Enter Base Salary: "))  # Convert input to float
    Overtime_Pay = float(input("Enter Overtime Pay: "))  # Convert input to float
    Longevity_Pay = float(input("Enter Longevity Pay: "))  # Convert input to float
    Grade = input("Enter Grade: ")

    # Create a dictionary from the user's input
    new_record = {'Department': Department, 'Department_Name': Department_Name, 'Division': Division, 'Gender': Gender, 'Base_Salary': Base_Salary, 'Overtime_Pay': Overtime_Pay, 'Longevity_Pay': Longevity_Pay, 'Grade': Grade}

    # Add the new record to the DataFrame at the end
    df.loc[len(df)] = new_record

    # Return the updated DataFrame
    return df

palette = seaborn.color_palette("Reds", 4)

# Function to plot department-wise salary distribution
def plot_depwise_salary_distribution(df):
    # Create a new figure with specified size
    plt.figure(figsize=(10, 14))
    # Group the DataFrame by 'Department_Name', calculate the mean of 'Base_Salary' for each group,
    # sort the values, and plot a horizontal bar chart
    df.groupby('Department_Name')['Base_Salary'].mean().sort_values().plot(kind='barh', color=palette)
    # Set the title of the plot
    plt.title('Average Salary by Department')
    # Set the label for the x-axis
    plt.xlabel('Average Salary')
    # Set the label for the y-axis
    plt.ylabel('Department')
    # Display the plot
    plt.show()

# Function to plot a pie chart of gender distribution
def plot_pie_chart(df):
    # Create a new figure with specified size
    plt.figure(figsize=(8, 6))
    # Count the occurrences of each unique value in the 'Gender' column,
    # and plot a pie chart with percentage labels
    df['Gender'].value_counts().plot(kind='pie', autopct='%1.2f%%', colors=palette)
    # Set the title of the plot
    plt.title('Gender Distribution')
    # Remove the label for the y-axis
    plt.ylabel('')
    # Display the plot
    plt.show()

# Function to plot a bar chart of gender distribution
def plot_bar_chart(df):
    # Create a new figure with specified size
    plt.figure(figsize=(8, 6))
    # Count the occurrences of each unique value in the 'Gender' column,
    # and plot a bar chart
    df['Gender'].value_counts().plot(kind='bar', color=palette)
    # Set the title of the plot
    plt.title('Gender Distribution of Employees')
    # Set the label for the x-axis
    plt.xlabel('Gender')
    # Set the label for the y-axis
    plt.ylabel('Count')
    # Display the plot
    plt.show()

# Function to plot pairwise scatter plot of numerical variables
def plot_scatter_pairwise(df):
    # Use seaborn's pairplot function to create a grid of scatter plots
    seaborn.pairplot(df[['Base_Salary', 'Overtime_Pay', 'Longevity_Pay']], diag_kind='kde', plot_kws={'color': palette[0]})
    plt.show()

# Function to display descriptive statistics of employee salaries
def plot_descriptive_statistics(df):
    # Get only the numeric columns from the DataFrame
    numeric_df = df._get_numeric_data()

    # Calculate and print descriptive statistics
    descriptiveStateDF = numeric_df.describe()

    # Append median to the descriptive statistics DataFrame
    descriptiveStateDF = pd.concat([descriptiveStateDF,median(numeric_df)],axis = 0)
    descriptiveStateDF.rename(index={0:'median'}, inplace=True)
    
    # Append variance to the descriptive statistics DataFrame
    descriptiveStateDF = pd.concat([descriptiveStateDF,variance(descriptiveStateDF)],axis = 0)
    descriptiveStateDF.rename(index={0:'variance'}, inplace=True)

    print(descriptiveStateDF)

    # Print mode statistics of employee salaries
    print("\nMode Statistics of Employee Salaries:")
    print(descriptiveStateDF.mode())

# Function to calculate median of salary components
def median(df):
    # Calculate median for each salary component and return as DataFrame
    medianDF = pd.DataFrame([{'Base_Salary':df['Base_Salary'].median(), 'Overtime_Pay':df['Overtime_Pay'].median(), 'Longevity_Pay':df['Longevity_Pay'].median()}])
    return medianDF

# Function to calculate variance of salary components
def variance(df):
    # Calculate variance for each salary component and return as DataFrame
    varianceDF = pd.DataFrame([{'Base_Salary':float(df['Base_Salary']['std'])**2, 'Overtime_Pay':float(df['Overtime_Pay']['std'])**2, 'Longevity_Pay':float(df['Longevity_Pay']['std'])**2}])
    return varianceDF

# Function to plot a box plot of 'Base_Salary' by 'Gender'
def boxPlot(df):
    df.boxplot(column='Base_Salary', by='Gender', grid=False, color=dict(boxes="#990000", whiskers="#ff0000", medians="#ff3333", caps="#ff6666"))
    plt.title('Base Salary by Gender')
    plt.xlabel('Gender')
    plt.ylabel('Base Salary')
    plt.show()

# Function to plot a heatmap of the correlation matrix
def plot_heat_map(df):
    plt.figure(figsize=(10, 10))
    numeric_df = df._get_numeric_data()
    seaborn.heatmap(numeric_df.corr(), annot=True)
    plt.title('Pairwise correlation of columns', fontsize=16)
    plt.show()

# Function to plot histograms and skewness of salary components
def plot_Skew(df):
    fig, axs = plt.subplots(1, 3, figsize=(12, 10))
    salaries = ['Base_Salary', 'Overtime_Pay', 'Longevity_Pay']
    skew_label_func = lambda skew: 'Right Skew' if skew > 0 else ('Left Skew' if skew < 0 else 'No Skew')
    for salary, ax in zip(salaries, axs):
        seaborn.histplot(df[salary], bins=20, color='green', ax=ax, edgecolor='black', kde=True)
        skewness = df[salary].skew()
        ax.set_title(f'{salary} (Skew: {skewness:.2f})', fontstyle='italic')
        ax.text(0.05, 0.9, skew_label_func(skewness), transform=ax.transAxes, fontsize=10, bbox=dict(facecolor='red', alpha=0.5))
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
    plt.show()

# Function to display the correlation matrix
def display_correlation(df):
    df = df[['Overtime_Pay', 'Longevity_Pay', 'Base_Salary']]
    df = df._get_numeric_data()
    correlation_matrix = df.corr()
    print("\nCorrelation Matrix:")
    print(correlation_matrix)

# Function to display the covariance matrix
def display_covariance(df):
    df = df[['Overtime_Pay', 'Longevity_Pay', 'Base_Salary']]
    df = df._get_numeric_data()
    covariance_matrix = np.cov(df, rowvar=False)
    cov_df = pd.DataFrame(covariance_matrix, columns=df.columns, index=df.columns)
    print("\nCovariance Matrix:")
    print(cov_df)

# Function to fit a regression model and perform hypothesis testing
def display_regression_hypothesisTesting(df):
    X = df[['Overtime_Pay', 'Longevity_Pay']]  # Independent variables
    y = df['Base_Salary']  # Dependent variable
    X = sm.add_constant(X)  # Adds a constant term to the predictor
    model = sm.OLS(y, X).fit()
    t_test = model.t_test('Overtime_Pay = 0')  # Test if the coefficient of 'Overtime_Pay' is zero
    print(model.summary())
    print(t_test)

def predict_regression_model(df, confidence):
    # Encode categorical variables
    for c in df.columns:
        if df[c].dtype == 'object': 
            df[c] = df[c].fillna('N')
            lbl = LabelEncoder()
            lbl.fit(list(df[c].values))
            df[c] = lbl.transform(df[c].values)

    # Define target variable
    target_data = ['Grade']
    X = df.drop(target_data, axis=1)
    y = df['Grade']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    prediction_score = model.score(X_test, y_pred)
    original_score = model.score(X_test, y_test)

    # Calculate confidence intervals
    n = len(y_test)
    std_err = np.std(y_test - y_pred)
    margin_err = std_err * stats.t.ppf((1 + (confidence / 100)) / 2, n - 1)
    lower_bound = y_pred - margin_err
    upper_bound = y_pred + margin_err

    # Print results
    print("Model: Linear Regression")
    print("Train Score: {:.2f}".format(train_score))
    print("Test Score: {:.2f}".format(test_score))
    print("Prediction Score: {:.2f}".format(prediction_score))
    print("Original Score: {:.2f}".format(original_score))
    print()

    # Plot histograms of confidence intervals
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    counts, bins,_  = ax1.hist(lower_bound, bins=20, density=True, alpha=0.7, color='y')
    ax1.set_title('Lower Bound Confidence Interval')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Density')

    # Plot Poisson distribution
    mu_lower = np.mean(lower_bound)
    poisson_dist_lower = poisson(mu_lower)
    x_lower = np.arange(0, max(bins), 1)
    ax1.plot(x_lower, poisson_dist_lower.pmf(x_lower), color='b', label='Poisson Distribution')
    ax1.legend()

    counts, bins,_  = ax2.hist(upper_bound, bins=20, density=True, alpha=0.7, color='r')
    ax2.set_title('Upper Bound Confidence Interval')
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Density')

    # Plot Poisson distribution
    mu_upper = np.mean(upper_bound)
    poisson_dist_upper = poisson(mu_upper)
    x_upper = np.arange(0, max(bins), 1)
    ax2.plot(x_upper, poisson_dist_upper.pmf(x_upper), color='g', label='Poisson Distribution')
    ax2.legend()
    plt.tight_layout()
    plt.show()

# Function to clear the console screen
def clear_screen():
    clear = lambda: os.system('cls')
    clear()
