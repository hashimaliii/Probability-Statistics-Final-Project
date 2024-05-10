# Import the necessary functions from your 'func' module
from func import *

# This is the main function that will be run when the script is executed
def main(df):
    # This loop will continue until the user chooses to exit
    while True:
        # Clear the screen for a clean interface each time the menu is displayed
        clear_screen()

        # Display the menu options to the user
        print("\nPlease choose an option:")
        # Each print statement corresponds to a different functionality of the program
        print("1. Plot Pie Chart")
        print("2. Plot Bar Chart")
        print("3. Plot Correlation Scatter Pairwise")
        print("4. Plot Department-wise Salary Distribution")
        print("5. Box Plot")
        print("6. Plot Descriptive Statistics")
        print("7. Plot Heat Map")
        print("8. Plot Skew")
        print("9. Predict Regression Model & Poisson Distribution")
        print("10. Display Correlation")
        print("11. Display Covariance")
        print("12. Display Regression & Hypothesis Testing")
        print("13. Add Record in the Dataset")
        print("14. Display Head Dataset")
        print("15. Display Tail Dataset")
        print("16. Display Complete Dataset")
        print("0. Exit")

        # Prompt the user to enter their choice
        choice = input("\nEnter your choice: ")
        print("\n")

        # Depending on the user's choice, call the appropriate function
        if choice == '1':
            plot_pie_chart(df)
        elif choice == '2':
            plot_bar_chart(df)
        elif choice == '3':
            plot_scatter_pairwise(df)
        elif choice == '4':
            plot_depwise_salary_distribution(df)
        elif choice == '5':
            boxPlot(df)
        elif choice == '6':
            plot_descriptive_statistics(df)
        elif choice == '7':
            plot_heat_map(df)
        elif choice == '8':
            plot_Skew(df)
        elif choice == '9':
            confidence_level = float(input("Enter Confidence: "))
            predict_regression_model(df, confidence_level)
        elif choice == '10':
            display_correlation(df)
        elif choice == '11':
            display_covariance(df)
        elif choice == '12':
            display_regression_hypothesisTesting(df)
        elif choice == '13':
            df = add_record(df)
        elif choice == '14':
            print(df.head())
        elif choice == '15':
            print(df.tail())
        elif choice == '16':
            print(df)
        elif choice == '0':
            break
        else:
            print("Invalid choice. Please choose a valid option.")
        
        # Pause the program until the user presses a key, so they have time to read the output
        input("\nPress any key to continue...")

# This code runs when the script is executed directly (not when imported as a module)
if __name__ == "__main__":
    # Initialize choice variable
    choice = -1

    # This loop will continue until the user chooses to exit
    while True:

        # Display the menu options to the user
        print("\nPlease choose an option:")
        print("1. Load Data from Github")
        print("2. Load Data from CSV")
        print("0. Exit")

        # Prompt the user to enter their choice
        choice = input("\nEnter your choice: ")

        # If the user chooses to load data from Github
        if choice == '1':
            # Call the function to load data from Github
            df = load_data_from_github()
            # Notify the user that the dataset has been loaded successfully
            print("\nDataset Loaded Successfully!")
            # Pause the program until the user presses a key, so they have time to read the output
            input("\nPress any key to continue...")
            # Call the main function with the loaded DataFrame
            main(df)
            break
        # If the user chooses to load data from a CSV file
        elif choice == '2':
            # Call the function to load data from a CSV file
            df = load_data_from_csv()
            # Notify the user that the dataset has been loaded successfully
            print("\nDataset Loaded Successfully!")
            # Pause the program until the user presses a key, so they have time to read the output
            input("\nPress any key to continue...")
            # Call the main function with the loaded DataFrame
            main(df)
            break
        # If the user chooses to exit
        elif choice == '0':
            # Break the loop to end the program
            break
        # If the user enters an invalid choice
        else:
            # Notify the user that their choice is invalid
            print("\nInvalid choice. Please choose a valid option.")
