import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp
import matplotlib_inline

# Title of the app
st.title("📊 Linear Regression Analysis")
st.write('### Introduction:')
st.markdown("""
Linear regression is a statistical method used to model the relationship between a dependent variable and one 
or more independent variables. It assumes a linear relationship, meaning the change in the dependent variable 
is proportional to the change in the independent variable(s). In its simplest form—simple linear regression—it 
fits a straight line to the data to predict outcomes. This technique is widely used in data analysis and machine 
learning for tasks like forecasting, trend analysis, and understanding variable relationships.""")


st.markdown("""One of the key aspects of linear regression is the calculation of the best-fit line,
which minimizes the sum of the squares of the vertical distances (residuals) between the observed data points and the line itself.
 This method is known as "Ordinary Least Squares" (OLS). The equation of the best-fit line can be expressed as 
$y = mx + b$, where $m$ is the slope and $b$ is the y-intercept. The slope indicates how much the dependent variable 
changes for a unit change in the independent variable, while the y-intercept represents the value of the dependent 
variable when the independent variable is zero.
""")
st.markdown("""First we will plot the data points in a scatter plot, and then we will 
            find the best fit line using linear regression. Upload a csv file with two columns, 
            where the first column is the independent variable (X) and the second column is the dependent variable (Y).""")


st.markdown("### Scatter Plot:")

# Function to create a scatter plot
@st.cache_data
def scatter_plot(df, x, y):
    # Plot the scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df[x], df[y], marker='o', color='red')
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"Scatter Plot of {x} vs {y}")
    ax.grid(True)
    st.pyplot(fig)
    return fig

#function to calculate various values for linear regression
@st.cache_data
def calculate_regression_values(df, x_axis, y_axis):
    n, m = df.shape
    X_bar = round(df[x_axis].mean(), 3)
    Y_bar = round(df[y_axis].mean(), 3)
    S_xx = ((df[x_axis] - X_bar) ** 2).sum()
    S_yy = ((df[y_axis] - Y_bar) ** 2).sum()
    S_xy = ((df[x_axis] - X_bar) * (df[y_axis] - Y_bar)).sum()
    slope = round(S_xy / S_xx, 3)
    b = round(1/n * (df[y_axis].sum() - slope * df[x_axis].sum()),3)
    R_squared = round(S_xy ** 2 / (S_xx * S_yy), 3)
    r = round(R_squared**0.5, 3)

    return slope, b, R_squared, r, X_bar, Y_bar, S_xx, S_yy, S_xy

#function to display regression line equation
@st.cache_data
def display_regression_equation(slope, intercept):
    x, y = sp.symbols('x y')
    
    # Create the equation: y = slope * x + intercept
    eqn = sp.Eq(y, slope * x + intercept)
    
    # Display the equation in LaTeX using Streamlit
    return sp.latex(eqn)


#function to display regression line:
@st.cache_data
def display_regression_line(df, x_axis, y_axis, slope, intercept):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df[x_axis], df[y_axis], marker='o', color='red', label='Data Points')
    ax.plot(df[x_axis], slope * df[x_axis] + intercept, color='blue', label='Regression Line')
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.set_title(f"Scatter Plot with Regression Line of {y_axis} vs {x_axis}")
    plt.grid(True)
    plt.legend()
    st.pyplot(fig)
    return fig


#predicting values using the regression line
@st.cache_data
def predict_value(slope, intercept, x_value):
    """
    Predicts the value of Y for a given X using the regression line equation.
    
    Parameters:
    slope (float): The slope of the regression line.
    intercept (float): The y-intercept of the regression line.
    x_value (float): The value of X for which to predict Y.
    
    Returns:
    float: The predicted value of Y.
    """
    return slope * x_value + intercept



# File uploader for CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# Check if a file has been uploaded
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    #st.write("Preview of uploaded data:")
    #st.dataframe(df.head())

    # Check if the dataframe has at least two numeric columns
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    if len(numeric_columns) < 2:
        st.warning("The uploaded file must contain at least two numeric columns for plotting.")
    else:
        # Select columns for scatter plot
        x_axis = st.selectbox("Select X-axis column", numeric_columns)        
        y_axis = st.selectbox("Select Y-axis column", numeric_columns, index=1)

        figure = scatter_plot(df, x_axis, y_axis)
        st.markdown(figure)


        st.markdown("""Does the data align along a straight line? Lets find line that best fits the data""")
        st.markdown('The best fit line has an equation of the form: $\\bar{y} = mx + b$, where $m$ is the slope and $b$ is the y-intercept.')
        st.markdown('We calculate the slope and y-intercept using the following formulas by minimizing the sum of squares of the residuals:')


        #calling the function to calculate regression values
        slope, b, R_squared, r, X_bar, Y_bar, S_xx, S_yy, S_xy = calculate_regression_values(df, x_axis, y_axis)

        st.markdown("#### Regression Analysis Results:")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            - ###### Mean of X = $$\\bar{{X}} = $$ {X_bar}
            - ###### Mean of Y = $$\\bar{{Y}} = $$ {Y_bar}
            - ###### $$S_{{xx}} = \\sum (x_i-\\bar{{X}})^2 = $$ {round(S_xx,3)}
            - ###### $$S_{{yy}} = \\sum (y_i-\\bar{{Y}})^2 = $$ {round(S_yy,3)}
            - ###### $$S_{{xy}} = \\sum (x_i-\\bar{{X}})(y_i-\\bar{{Y}}) = $$ {round(S_xy,3)}
                    """)
        with col2:
            st.markdown(f"""
            - ###### Slope: $$\quad m = $$  {slope}
            - ###### Y-intercept: $$\quad b = $$ {b}
            - ###### Coef of Determination: $\quad R^2 = $ {R_squared}
            - ###### Linear Correlation Coef: $\quad r = \\sqrt{{R^2}} = $ {r}
                    """)
    

        
        #display the regression equation
        eqn = display_regression_equation(slope, b)
        st.markdown(f""" ##### The equation of the line is:$\quad$ {eqn}""")
        st.markdown('#### Regression Line Plot:')
        if st.button("##### Plot Regression Line"):
            figure = display_regression_line(df, x_axis, y_axis, slope, b)
            st.markdown(figure)



        st.markdown('#### Predicting Values:')
        st.write("You can use the equation of the line to predict values of Y for given values of X.")
        X_new = st.number_input("Enter a value for X:", value=0.0)
        if st.button("Predict"):
            Y_pred = predict_value(slope, b, X_new)
            st.markdown(f"##### The predicted value of Y for X = {X_new} is $\quad $ {Y_pred}")
        


else:
    st.write("Please upload a CSV file")

