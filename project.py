import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objs as go
from scipy.optimize import minimize
from typing import List, Tuple
from functools import cache  
from IPython.display import display
from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sb
from datetime import datetime
from datetime import date
from nsepy import get_history as gh
plt.style.use('fivethirtyeight')
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import  risk_models
from pypfopt import expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pypfopt import plotting
from pypfopt import HRPOpt
from pypfopt.efficient_frontier import EfficientCVaR
import pydot
import matplotlib.colors as mcolors
import graphviz
import hashlib
import sqlite3 
import warnings
warnings.filterwarnings('ignore')

def page_home():
    
    st.subheader("""
    TRENT UNIVERSITY
    
    APPLIED MODELLING AND QUANTITATIVE METHODS
    
    BIG DATA AND FINANCIAL ANALYTICS STREAMS
    """)
    st.write("""


Welcome to the Automation Portfolio Optimization Project!

We are excited to present to you the product of Grace and Shamiul's final work in Applied Modelling and Quantitative Methods, MSc, at Trent University: a thorough portfolio optimization system.
We started our adventure as enthusiastic students of finance and quantitative analysis with the intention of developing a cutting-edge solution that would transform the field of investment management. Our initiative focuses on using automation to expedite portfolio optimization, making it more effective, dependable, and open to everyone.

We have extensively studied the fields of programming, data analytics, and mathematical modelling during our academic careers. We have developed a strong system that makes use of modern algorithms and machine learning techniques to make sure that your assets are carefully allocated for maximum returns and little risk. By combining this expertise with our unwavering excitement for financial markets, we have created this system.

Our Automation Portfolio Optimization System's key characteristics are as follows:
Adaptive asset allocation depending on your risk tolerance and investment objectives.
Integration of real-time market data for accurate and current analysis.
Thorough risk analysis and stress testing to protect your money.
User-friendly interface for simple portfolio modification and seamless navigation.
Functionality that uses past market circumstances to assess performance.

We are eager to share our research with you, our colleagues, our lecturers, and the financial industry. We hope that our Automation Portfolio Optimization Project will serve as a catalyst for more developments in the fields of finance and quantitative analysis since we think it has the potential to have a significant influence.
We appreciate your participation in this thrilling endeavor. We cordially encourage you to learn more about our initiative and see for yourself how automation may maximize the effectiveness of your investment plan.

Grace Faniyi - 0739463

MD Shamiul Islam - 0743469
""")

    st.info("Go to Menu to SignUp/Login")

def page_signup():
    st.subheader("Create New Account")

    new_user = st.text_input("Username")
    new_password = st.text_input("Password", type='password')

    if st.button("Signup"):
        create_usertable()
        add_userdata(new_user, make_hashes(new_password))
        st.success("Thank you for signing up and becoming a part of the success of this project and finance enthusiasts. {}".format(new_user))
        st.info("Please! Go to Menu to login")


def page_login():
    st.subheader("Login Session")

    # Create empty elements
    login_username = st.sidebar.empty()
    login_password = st.sidebar.empty()
    login_button = st.sidebar.empty()

    # Check if login is successful
    if 'login_success' not in st.session_state:
        st.session_state.login_success = False

    if not st.session_state.login_success:
        username = login_username.text_input("User Name")
        password = login_password.text_input("Password", type='password')

        if login_button.button("Login"):
            create_usertable()
            hashed_pswd = make_hashes(password)

            result = login_user(username, check_hashes(password, hashed_pswd))
            if result:
                st.session_state.login_success = True
                st.success(f"Welcome back, {username}! Thank you for signing up and becoming a part of the success of this project and finance enthusiasts.")
            else:
                st.warning("Signup or Incorrect Username/Password")

        
def page2():
    st.title("Page 2")
    st.write("This is Page 2.")
    st.write("You can add content for Page 2 here.")
    
    # Add radio buttons for user to choose options
    option = st.radio("Choose an option:", ("Go to Page 1", "Log out"))

    if option == "Go to Page 1":
        st.experimental_rerun()
    elif option == "Log out":
        st.session_state.login_success = False
        #st.experimental_rerun()

def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password,hashed_text):
    if make_hashes(password) == hashed_text:
            return hashed_text
    return False
# DB Management

conn = sqlite3.connect('data.db')
c = conn.cursor()
# DB  Functions
def create_usertable():
        c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')


def add_userdata(username,password):
        c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',(username,password))
        conn.commit()

def login_user(username,password):
        c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',(username,password))
        data = c.fetchall()
        return data


def view_all_users():
        c.execute('SELECT * FROM userstable')
        data = c.fetchall()
        return data

def fetch_data(tickers):
    try:
            # Fetch historical stock data for the selected tickers
        data = yf.download(tickers, period="3y", interval="1d", progress=False)['Adj Close']
        return data
    except:
        return None

def filter_valid_tickers(tickers):
    valid_tickers = []
    for ticker in tickers:
        try:
            dataa = yf.download(ticker, period='1d')  # Try to download historical data
            if not dataa.empty:  # Check if data was successfully retrieved
                valid_tickers.append(ticker)
            else:
                st.write(f"Delisted or Data not availabe: {ticker}")
        except:
            st.write("Enter valid ticker")
    return valid_tickers


       
        
def process_input(all_tickers):
    # Create a dictionary with variables used in the function (if needed)
    #tickers = {}
    # Evaluate the function using eval
    tickers = filter_valid_tickers(all_tickers)
    return bool(all_tickers.strip())

def get_stock_data(tickers, dataset_period):

    if dataset_period == "Daily":
        data = yf.download(tickers, period='3y', interval='1d')['Adj Close']
    else:
        data = yf.download(tickers, period='3y', interval='1wk')['Adj Close']
    return data
    
# Function to calculate Mean Conditional Value at Risk (mCVAR)
def mCVAR(portfolio_returns, confidence_level=0.95):
    alpha = 1 - confidence_level
    return -portfolio_returns.quantile(alpha)

# Function to calculate the Sharpe ratio
def sharpe_ratio(returns, risk_free_rate):
    excess_returns = returns - risk_free_rate
    return excess_returns.mean() / excess_returns.std()



# Initialize login_success attribute
if "login_success" not in st.session_state:
    st.session_state.login_success = False



def page2():

    # Select dataset period (Daily or Weekly)
    dataset_period = st.sidebar.selectbox("Select Stock Data", ("Daily", "Weekly"))

    total_portfolio_value = st.sidebar.slider('Amount to invest', 5000, 10000)
    operation = st.sidebar.selectbox("Select portfolio objective", ("Optimal Portfolio", "Diversified Portfolio", "Low Risk Portfolio"))
    # Sample list of tickers
    all_tickers = None
    while all_tickers is None:
        all_tickers = st.sidebar.text_input('Enter stock tickers without spaces, e.g. AMZN,JPM,BA,GOOG', '').upper().replace(',', '\n').split('\n')

    # Filter out invalid or delisted tickers
    tickers = filter_valid_tickers(all_tickers)
    data = None  # Initialize the data variable
    if tickers:
        # Display selected tickers
        tickers_df = pd.DataFrame({'Tickers': tickers})

        # Display the title using Markdown formatting
        st.markdown('Selected Tickers')

        # Display the table
        st.table(tickers_df)

        caption_text = ("Table 1: This table shows the selected tickers.")
         
        st.caption(caption_text)
        
        
        # Get stock data
        data = get_stock_data(tickers, dataset_period)
        
        # Display the title using Markdown formatting
        st.markdown('Raw close stock data')
        data_1 = data.tail()
        st.table(data_1.applymap('{:.2f}'.format))

        #ax.annotate(f'{height:.2f}'
        
        caption_text = ("Table 2: This table shows the last 5 Rows of Raw Data.")
         
        st.caption(caption_text)

        descr = data.describe()
        st.table(descr.applymap('{:.2f}'.format))

        # Continue with portfolio optimization or other operations using the selected tickers and data

    
        # Use the 'data' variable for further processing, e.g., portfolio optimization
    
    
        mean = expected_returns.mean_historical_return(data)
        S = risk_models.sample_cov(data)
        st.set_option('deprecation.showPyplotGlobalUse', False)

        plt.style.use('classic')
        fig = plt.figure(figsize=(16, 10))
        for i in range(data.shape[1]):
            plt.plot(data.iloc[:,i], label=data.columns.values[i])
        plt.legend(loc='upper left', fontsize=10)
        plt.title('Stock Close Price Trends', fontsize=18)
        plt.xlabel('Date', fontsize=16)
        plt.ylabel('Price in $', fontsize=16)
        st.pyplot()
        caption_text = ("This chart displays the trend of closing prices of selected stocks over time.\
                        It provides insights into the historical performance of the stocks and helps identify\
                        patterns and potential investment opportunities.")
         
        st.caption(caption_text)

        # Normalized price
        data_nor = data.divide(data.iloc[0] / 100)

        fig = plt.figure(figsize=(16, 10))
        for i in range(data_nor.shape[1]):
            plt.plot(data_nor.iloc[:,i], label=data_nor.columns.values[i])
        plt.legend(loc='upper left', fontsize=10)
        plt.ylabel('Normalized prices($)')
        plt.title('Stock Normalized Prices Trends', fontsize=18)
        plt.xlabel('Date', fontsize=16)
        plt.ylabel('Price in $', fontsize=16)
        st.pyplot()
        caption_text = ("This chart displays the trend of normalized closing prices of selected stocks over time.\
                        It provides insights into the historical performance of the stocks and helps identify\
                        patterns and potential investment opportunities.")
        st.caption(caption_text)

        st.markdown('Covariance of stocks in your portfolio')
        plt.style.use('classic')
        fig = plt.figure(figsize=(16, 10))
        sb.heatmap(S,xticklabels=S.columns, yticklabels=S.columns,
        cmap='Greens', annot=True, linewidth=0.5)
        st.pyplot()
        caption_text = ("This visualization represents a color-coded matrix where each cell's color\
        intensity corresponds to covariance of stocks in your portfolio.")
        st.caption(caption_text)


        ef = EfficientFrontier(mean,S)
        weights = ef.max_sharpe() #for maximizing the Sharpe ratio #Optimization
        cleaned_weights = ef.clean_weights() #to clean the raw weights
    

        rcParams['figure.figsize'] = 16, 10

        TREASURY_BILL_RATE = 0.0528 
        TRADING_DAYS_PER_YEAR = 250


        # Needed for type hinting
        class Asset:
            pass

        def get_log_period_returns(price_history: pd.DataFrame):
            close = price_history['Close'].values
            return np.log(close[1:] / close[:-1]).reshape(-1, 1)


        # daily_price_history has to at least have a column, called 'Close'
        class Asset:
          def __init__(self, name: str, daily_price_history: pd.DataFrame):
            self.name = name
            self.daily_returns = get_log_period_returns(daily_price_history)
            self.expected_daily_return = np.mean(self.daily_returns)

          @property
          def expected_return(self):
            return TRADING_DAYS_PER_YEAR * self.expected_daily_return

          def __repr__(self):
            return f'<Asset name={self.name}, expected return={self.expected_return}>'

          @staticmethod
          @cache
          def covariance_matrix(assets: Tuple[Asset]):  # tuple for hashing in the cache
            product_expectation = np.zeros((len(assets), len(assets)))
            for i in range(len(assets)):
              for j in range(len(assets)):
                if i == j:
                  product_expectation[i][j] = np.mean(assets[i].daily_returns * assets[j].daily_returns)
                else:
                  product_expectation[i][j] = np.mean(assets[i].daily_returns @ assets[j].daily_returns.T)
    
            product_expectation *= (TRADING_DAYS_PER_YEAR - 1) ** 2

            expected_returns = np.array([asset.expected_return for asset in assets]).reshape(-1, 1)
            product_of_expectations = expected_returns @ expected_returns.T

            return product_expectation - product_of_expectations


        def random_weights(weight_count):
            weights = np.random.random((weight_count, 1))
            weights /= np.sum(weights)
            return weights.reshape(-1, 1)


        class Portfolio:
          def __init__(self, assets: Tuple[Asset]):
            self.assets = assets
            self.asset_expected_returns = np.array([asset.expected_return for asset in assets]).reshape(-1, 1)
            self.covariance_matrix = Asset.covariance_matrix(assets)
            self.weights = random_weights(len(assets))
    
          def unsafe_optimize_with_risk_tolerance(self, risk_tolerance: float):
            res = minimize(
              lambda w: self._variance(w) - risk_tolerance * self._expected_return(w),
              random_weights(self.weights.size),
              constraints=[
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.},
              ],
              bounds=[(0., 1.) for i in range(self.weights.size)]
            )

            assert res.success, f'Optimization failed: {res.message}'
            self.weights = res.x.reshape(-1, 1)
  
          def optimize_with_risk_tolerance(self, risk_tolerance: float):
            assert risk_tolerance >= 0.
            return self.unsafe_optimize_with_risk_tolerance(risk_tolerance)
  
          def optimize_with_expected_return(self, expected_portfolio_return: float):
            res = minimize(
              lambda w: self._variance(w),
              random_weights(self.weights.size),
              constraints=[
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.},
                {'type': 'eq', 'fun': lambda w: self._expected_return(w) - expected_portfolio_return},
              ],
              bounds=[(0., 1.) for i in range(self.weights.size)]
            )

            assert res.success, f'Optimization failed: {res.message}'
            self.weights = res.x.reshape(-1, 1)
        

          def optimize_sharpe_ratio(self):
            # Maximize Sharpe ratio = minimize minus Sharpe ratio
            res = minimize(
              lambda w: -(self._expected_return(w) - TREASURY_BILL_RATE / 100) / np.sqrt(self._variance(w)),
              random_weights(self.weights.size),
              constraints=[
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.},
              ],
              bounds=[(0., 1.) for i in range(self.weights.size)]
            )

            assert res.success, f'Optimization failed: {res.message}'
            self.weights = res.x.reshape(-1, 1)

          def _expected_return(self, w):
            return (self.asset_expected_returns.T @ w.reshape(-1, 1))[0][0]
  
          def _variance(self, w):
            return (w.reshape(-1, 1).T @ self.covariance_matrix @ w.reshape(-1, 1))[0][0]

          @property
          def expected_return(self):
            return self._expected_return(self.weights)
  
          @property
          def variance(self):
            return self._variance(self.weights)

          def __repr__(self):
            return f'<Portfolio assets={[asset.name for asset in self.assets]}, expected return={self.expected_return}, variance={self.variance}>'

        def yf_retrieve_data(tickers: List[str]):
          dataframes = []

          for ticker_name in tickers:
            ticker = yf.Ticker(ticker_name)
            history = ticker.history(period='3y')

            if history.isnull().any(axis=1).iloc[0]:  # the first row can have NaNs
              history = history.iloc[1:]
  
            assert not history.isnull().any(axis=None), f'history has NaNs in {ticker_name}'
            dataframes.append(history)
  
          return dataframes


        daily_dataframes = yf_retrieve_data(tickers)
        assets = tuple([Asset(name, daily_df) for name, daily_df in zip(tickers, daily_dataframes)])

        X = []
        y = []

        # Drawing random portfolios
        for i in range(3000):
          portfolio = Portfolio(assets)
          X.append(np.sqrt(portfolio.variance))
          y.append(portfolio.expected_return)

        plt.scatter(X, y, label='Random portfolios')

        # Drawing the efficient frontier
        X = []
        y = []
        for rt in np.linspace(-300, 200, 1000):
          portfolio.unsafe_optimize_with_risk_tolerance(rt)
          X.append(np.sqrt(portfolio.variance))
          y.append(portfolio.expected_return)

        sm = plt.cm.ScalarMappable(cmap='Greens')
        plt.plot(X, y, 'k', linewidth=3, label='Efficient Frontier')

        # Drawing optimized portfolios
        portfolio.optimize_with_risk_tolerance(0)
        plt.plot(np.sqrt(portfolio.variance), portfolio.expected_return, 'm+', markeredgewidth=5, markersize=20, label='optimize_with_risk_tolerance')

        portfolio.optimize_with_risk_tolerance(20)
        plt.plot(np.sqrt(portfolio.variance), portfolio.expected_return, 'r+', markeredgewidth=5, markersize=20, label='optimize_with_risk_tolerance')

        portfolio.optimize_with_expected_return(0.25)
        plt.plot(np.sqrt(portfolio.variance), portfolio.expected_return, 'g+', markeredgewidth=5, markersize=20, label='optimize_with_expected_return')

        portfolio.optimize_sharpe_ratio()
        plt.plot(np.sqrt(portfolio.variance), portfolio.expected_return, 'y+', markeredgewidth=5, markersize=20, label='optimize_sharpe_ratio')

        plt.colorbar(sm, label='Sharpe Ratio')
        plt.xlabel('Portfolio standard deviation', fontsize=16)
        plt.ylabel('Portfolio expected (logarithmic) return', fontsize=16)
        plt.title('Efficient Frontier Graph', fontsize=18)
        plt.legend(title='Indicators', loc='lower right')
        st.pyplot()
        caption_text = ("This graph illustrates the optimal portfolio combinations of assets that offer\
        the highest expected return for a given level of risk. ")
        st.caption(caption_text)


        
        latest_prices = get_latest_prices(data)
        performance_ef= ef.portfolio_performance(verbose=True)
        list_value_ef = list(performance_ef)
        list_value_ef = [x for x in list_value_ef if x != 0]
        df_ef = pd.DataFrame(list_value_ef, columns=["Value"])
        Metric = ["Expected annual return", "Annual volatility", "Sharpe Ratio"]
        df_ef["Metrics"] = Metric
        df_ef = df_ef[["Metrics", "Value"]]
        da_ef = DiscreteAllocation(weights, latest_prices, total_portfolio_value)
        label_ef = list(cleaned_weights.keys())
        value_ef = list(cleaned_weights.values())
    


        returns = data.pct_change().dropna()
        hrp = HRPOpt(returns)
        hrp_weights = hrp.optimize()
    
        performance_hrp = hrp.portfolio_performance(verbose=True)
        list_value_hrp = list(performance_hrp)
        list_value_hrp = [x for x in list_value_hrp if x != 0]
        df_hrp = pd.DataFrame(list_value_hrp, columns=["Value"])
        df_hrp["Metrics"] = Metric
        df_hrp = df_hrp[["Metrics", "Value"]]
        da_hrp = DiscreteAllocation(hrp_weights, latest_prices, total_portfolio_value)
        label_hrp = list(hrp_weights.keys())
        value_hrp = list(hrp_weights.values())
        

        S = data.cov()
        ef_cvar = EfficientCVaR(mean, S)   
        ef_cvar_weights = ef_cvar.min_cvar()    
        ef_cvar_weights = ef_cvar.clean_weights()
    
        da_cvar = DiscreteAllocation(ef_cvar_weights, latest_prices, total_portfolio_value)
        label_cvar = list(ef_cvar_weights.keys())
        value_cvar = list(ef_cvar_weights.values())

        
        
        # Calculate expected return and annual return
        expected_return = returns.mean()
        annual_return = (1 + expected_return) ** 252 - 1  # Assuming daily returns and 252 trading days in a year

        # Calculate Sharpe ratio (considering risk-free rate of 2%)
        risk_free_rate = 0.02
        sharpe = sharpe_ratio(returns, risk_free_rate)

        # Calculate mCVAR (considering 95% confidence level)
        mCVAR_value = mCVAR(returns, confidence_level=0.95)

        # Create DataFrame to display results
        results_df = pd.DataFrame({
            "Expected Return": expected_return,
            "Annual Return": annual_return,
            "Sharpe Ratio": sharpe,
            "mCVAR (95% Confidence Level)": mCVAR_value
        })

        st.markdown('Portfolio performance metrics')
        if operation == "Optimal Portfolio":
            da = da_ef
        elif operation == "Diversified Portfolio":
            da = da_hrp
        else:
            da = da_cvar

        if operation == "Optimal Portfolio":
            da = da_ef
            labels = label_ef
            values = value_ef
            st.write(df_ef)
        
        
        elif operation == "Diversified Portfolio":
            da = da_hrp
            labels = label_hrp
            values = value_hrp
            st.write(df_hrp)
        
    
        else:        
            da = da_cvar
            labels = label_cvar
            values = value_cvar
            st.write(results_df)
        
        
        caption_text = ("Table 3: This displays portfolio performance metrics that is, expected return, annual return,\
        and Sharpe ratio that are essential for risk-adjusted assessment and comparison.")
        st.caption(caption_text)
        
        weight = {'Ticker': labels,'Weight': values}
        
        # Convert the weight to a DataFrame
        filtered_weight = pd.DataFrame(weight)
        st.markdown('Allocated weight to each stock')
        
        st.table(filtered_weight)
        caption_text = ("Tale 4: This table shows allocated weight to each stock.")
        st.caption(caption_text)

        # Remove rows with zero weight
        filtered_weight = filtered_weight[filtered_weight['Weight'] != 0]
        st.table(filtered_weight)
        caption_text = ("Table 5: This table shows the stock(s) that qualified to be in the portfolio.")
        st.caption(caption_text)

        # Convert weights to percentages
        filtered_weight['Weight'] = filtered_weight['Weight'] * 100

        # bar chart of the filtered weights
        fig, ax = plt.subplots(figsize=(16, 10))
        light_colors = mcolors.LinearSegmentedColormap.from_list("", ["#cceeff", "#99ddff", "#66ccff", "#33bbff"])
        bars = ax.bar(filtered_weight['Ticker'], filtered_weight['Weight'], color=light_colors(filtered_weight.index))
        ax.set_xlabel('Ticker', fontsize=16)
        ax.set_ylabel('Weight(%)', fontsize=16)
        ax.set_title('Portfolio Allocation', fontsize=18)
        # Add padding to the x-axis limits
        padding = 0.5  # Adjust the padding value as needed
        xmin, xmax = -padding, len(filtered_weight['Ticker']) - 1 + padding
        ax.set_xlim(xmin, xmax)
        # Add length labels to the bars
        for bar in bars:
            height = bar.get_height()             
            ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom')
        plt.xticks(rotation=45)
        st.pyplot(fig)

        caption_text = ("This visualization presents a breakdown of the portfolio's asset allocation in the form of a bar chart.\
        Each bar represents the weight allocation of different assets or asset classes in the portfolio.")
        st.caption(caption_text)
        
        st.markdown("Discrete allocation")
        allocation, leftover = da.greedy_portfolio()
        allocation = pd.DataFrame(allocation.items(), columns=["Stock ticker", "Allocation(Unit)"])
        st.table(allocation)
        caption_text = ("Table 6: This chart showcases the discretionary allocation of stock within the portfolio.")
        st.caption(caption_text)
        st.write("Funds remaining as of today's price: ${:.2f}".format(leftover))
        caption_text = ("This is the remaining available funds in the portfolio after the asset allocation process.\
                         It shows the portion of funds not allocated to any specific asset or investment.")
        st.caption(caption_text)
        
        
        
    else:
        st.warning('Please enter valid tickers.')

    return data  # Return the data variable

     # Add radio buttons for user to choose options
    option = st.radio("Choose an option:", ("Go to Page 1", "Log out"))

    if option == "Go to Page 1":
        st.experimental_rerun()
    elif option == "Log out":
        st.session_state.login_success = False
        st.experimental_rerun()

def main():
    st.title("PORTFOLIO OPTIMIZATION APP")
    menu = ["Home", "Login", "SignUp"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        page_home()
    elif choice == "SignUp":
        page_signup()
    elif choice == "Login":
        page_login()

    if st.session_state.get('login_success'):
        page2()

    # Logout session
    if st.session_state.login_success:
        logout = st.button("Logout")
        if logout:
            st.session_state.login_success = False
        
if __name__ == "__main__":
    main()






