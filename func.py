import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import altair as alt
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from scipy.stats import poisson, gaussian_kde
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def load_data():
    return pd.DataFrame(pd.read_csv(r'dataset/Employee_Salaries.csv'))

def plot_line_chart(df):
    plt.figure(figsize=(10,10))
    x = df['Department_Name']
    y = df['Base_Salary']

    plt.plot(x,y)
    plt.xlabel('Department_Name')
    plt.ylabel('Base_Salary')
    plt.title('Employee Salaries', loc='center')
    plt.grid(True)
    plt.xticks(rotation=90, ha='right')
    plt.show()

def plot_depwise_salary_distribution(df):
    # Visualization : Department-wise salary distribution
    plt.figure(figsize=(10, 14))
    df.groupby('Department_Name')['Base_Salary'].mean().sort_values().plot(kind='barh', color=['#6a51a3', '#807dba', '#3f007d', '#54278f'])
    plt.title('Average Salary by Department')
    plt.xlabel('Average Salary')
    plt.ylabel('Department')
    plt.show()

def plot_pie_chart(df):
    # Visualization : Pie chart of gender distribution
    plt.figure(figsize=(8, 6))
    df['Gender'].value_counts().plot(kind='pie', autopct='%1.2f%%', colors=['#6a51a3', '#807dba', '#3f007d', '#54278f'])
    plt.title('Gender Distribution')
    plt.ylabel('')
    plt.show()

def plot_bar_chart(df):
    plt.figure(figsize=(8, 6))
    df['Gender'].value_counts().plot(kind='bar', color=['#6a51a3', '#807dba', '#3f007d', '#54278f'])
    plt.title('Gender Distribution of Employees')
    plt.xlabel('Gender')
    plt.ylabel('Count')
    plt.show()

def plot_scatter_pairwise(df):
    # Visualization 8: Pairwise scatter plot of numerical variables
    seaborn.pairplot(df[['Base_Salary', 'Overtime_Pay', 'Longevity_Pay']], diag_kind='kde', plot_kws={'color': '#807dba'})
    plt.show()

def plot_descriptive_statistics(df):
    numeric_df = df._get_numeric_data()

    print("\nDescriptive Statistics of Employee Salaries:")
    descriptiveStateDF = numeric_df.describe()

    # Median
    descriptiveStateDF = pd.concat([descriptiveStateDF,median(numeric_df)],axis = 0)
    descriptiveStateDF.rename(index={0:'median'}, inplace=True)
    
    # Variance
    descriptiveStateDF = pd.concat([descriptiveStateDF,variance(descriptiveStateDF)],axis = 0)
    descriptiveStateDF.rename(index={0:'variance'}, inplace=True)

    print(descriptiveStateDF)

    print("\nMode Statistics of Employee Salaries:")
    print(descriptiveStateDF.mode())

def median(df):
    baseSalary = df['Base_Salary'].median()
    overtimePay = df['Overtime_Pay'].median()
    longevityPay = df['Longevity_Pay'].median()
    medianDF = pd.DataFrame([{'Base_Salary':baseSalary, 'Overtime_Pay':overtimePay, 'Longevity_Pay':longevityPay}])
    return medianDF

def variance(df):
    baseSalary = float(df['Base_Salary']['std'])**2
    overtimePay = float(df['Overtime_Pay']['std'])**2
    longevityPay = float(df['Longevity_Pay']['std'])**2
    varianceDF = pd.DataFrame([{'Base_Salary':baseSalary, 'Overtime_Pay':overtimePay, 'Longevity_Pay':longevityPay}])
    return varianceDF

def boxPlot(df):
    # Visualization : Box plot of base salary by gender
    df.boxplot(column='Base_Salary', by='Gender', grid=False, color=dict(boxes='#6a51a3', whiskers='#807dba', medians='#3f007d', caps='#54278f'))
    plt.title('Base Salary by Gender')
    plt.xlabel('Gender')
    plt.ylabel('Base Salary')
    plt.show()

def plot_heat_map_correlation(df):
    plt.figure(figsize=(10, 10))
    numeric_df = df._get_numeric_data()
    seaborn.heatmap(numeric_df.corr(), annot=True)
    plt.title('Pairwise correlation of columns', fontsize=16)
    plt.title('Pairwise correlation of columns', loc='center')
    plt.show()

def plot_Skew(df):
    fig, axs = plt.subplots(1, 3, figsize=(12, 10))
    salaries = ['Base_Salary', 'Overtime_Pay', 'Longevity_Pay']
    data = [df[salary] for salary in salaries]
    for i, ax in enumerate(axs.flat):
        seaborn.histplot(data[i], bins=20, color='skyblue', ax=ax, edgecolor='black', kde=True)
        skewness = data[i].skew()
        ax.set_title(f'{salaries[i]} (Skew: {skewness:.2f})', fontstyle='italic')

        if skewness > 0:
            ax.text(0.05, 0.9, 'Right Skew', transform=ax.transAxes, fontsize=10, bbox=dict(facecolor='orange', alpha=0.5))
        elif skewness < 0:
            ax.text(0.05, 0.9, 'Left Skew', transform=ax.transAxes, fontsize=10, bbox=dict(facecolor='orange', alpha=0.5))
        else:
            ax.text(0.05, 0.9, 'No Skew', transform=ax.transAxes, fontsize=10, bbox=dict(facecolor='orange', alpha=0.5))

        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')

    plt.show()

def labelencoder(df):
    for c in df.columns:
        if df[c].dtype=='object': 
            df[c] = df[c].fillna('N')
            lbl = LabelEncoder()
            lbl.fit(list(df[c].values))
            df[c] = lbl.transform(df[c].values)
    return df

def preprocess_data(df):
    labelencoder(df)
    targetData = ['Grade']
    X = df.drop(targetData, axis = 1)
    y = df['Grade']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def train_models(X_train, y_train):
    linear_reg = LinearRegression()
    rf_reg = RandomForestRegressor()
    gb_reg = GradientBoostingRegressor()
    linear_reg.fit(X_train, y_train)
    rf_reg.fit(X_train, y_train)
    gb_reg.fit(X_train, y_train)
    return {'Linear Regression': linear_reg, 'Random Forest': rf_reg, 'Gradient Boosting': gb_reg}

def evaluate_models(models, X_train, X_test, y_train, y_test,confidance):
    results = {}
    for name, model in models.items():
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        y_pred = model.predict(X_test)
        prediction_score = model.score(X_test, y_pred)
        original_score = model.score(X_test, y_test)
        n = len(y_test)
        std_err = np.std(y_test - y_pred)
        margin_err = std_err * stats.t.ppf((1 + (confidance/100)) / 2, n - 1)
        lower_bound = y_pred - margin_err
        upper_bound = y_pred + margin_err

        results[name] = {'Train Score': train_score, 'Test Score': test_score, 
                         'Prediction Score': prediction_score, 'Original Score': original_score,
                         'Lower Bound': lower_bound, 'Upper Bound': upper_bound}
    return results

def train_predict_regression_model(df,confidance):
    X_train, X_test, y_train, y_test = preprocess_data(df)
    models = train_models(X_train, y_train)
    evaluation_results = evaluate_models(models, X_train, X_test, y_train, y_test,confidance)
    return models, evaluation_results

def plot_confidence_interval(lower_bound, upper_bound, confidence):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    counts, bins,_  = ax1.hist(lower_bound, bins=20, density=True, alpha=0.7, color='b')
    ax1.set_title('Lower Bound Confidence Interval')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Density')

    mu_lower = np.mean(lower_bound)
    poisson_dist_lower = poisson(mu_lower)
    x_lower = np.arange(0, max(bins), 1)
    ax1.plot(x_lower, poisson_dist_lower.pmf(x_lower), color='r', label='Poisson Distribution')
    ax1.legend()

    counts, bins,_  = ax2.hist(upper_bound, bins=20, density=True, alpha=0.7, color='g')
    ax2.set_title('Upper Bound Confidence Interval')
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Density')

    mu_upper = np.mean(upper_bound)
    poisson_dist_upper = poisson(mu_upper)
    x_upper = np.arange(0, max(bins), 1)
    ax2.plot(x_upper, poisson_dist_upper.pmf(x_upper), color='r', label='Poisson Distribution')
    ax2.legend()
    plt.tight_layout()
    plt.show()

def prediction_results(df,confidance):
    models, evaluation_results = train_predict_regression_model(df,confidance)
    for name, metrics in evaluation_results.items():
        print(f"Model: {name}")
        print(f"Train Score: {metrics['Train Score']:.2f}")
        print(f"Test Score: {metrics['Test Score']:.2f}")
        print(f"Prediction Score: {metrics['Prediction Score']:.2f}")
        print(f"Original Score: {metrics['Original Score']:.2f}")
        print()
    plot_confidence_interval(metrics['Lower Bound'],metrics['Upper Bound'],confidance)