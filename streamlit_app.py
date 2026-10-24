# Import necessary packages (Make sure to also include in "requirements.txt" in this repository.)
import streamlit as st
import pandas as pd
import statsmodels as sm

# Print Title for the App
st.title("Microsoft Stock Data Dashboard by Michael Romero")

# Read CSV file with stock data (CSV file must be in 'main' folder for your assignment 4 repository) 
df = pd.read_csv("msft.csv")

# Set up the 4 tabs on the Dashboard
tabs = st.tabs(["📋 Raw Data", "📈 Price Chart", "📊 Volume Chart", "📝 Price Regression on Volume"])

# Tab 1: Raw Data
with tabs[0]:
    st.subheader(f"Raw Data for MSFT")
    st.dataframe(df, use_container_width=True)

# Tab 2: Closing Price Chart
with tabs[1]:
    st.subheader(f"Closing Price for MSFT")
    st.line_chart(df['Close'])

# Tab 3: Volume Chart
with tabs[2]:
    st.subheader(f"Volume for MSFT")
    st.bar_chart(df['Volume'])

# Tab 4: Price Regression on Volume
with tabs[3]:
    st.subheader("Price Regression on Volume")
    
    # Prepare clean, numeric data for the regression
    reg_df = df[['Close', 'Volume']].copy()
    reg_df['Close'] = pd.to_numeric(reg_df['Close'], errors='coerce')
    reg_df['Volume'] = pd.to_numeric(reg_df['Volume'], errors='coerce')
    reg_df = reg_df.dropna()
    
    if not reg_df.empty:
        y = reg_df["Close"]
        X = reg_df["Volume"]
        
        # Use statsmodels
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()

        # Render the summary on the Streamlit dashboard as preformatted text
        st.text(str(model.summary()))
    else:
        st.error("Unable to run regression: No valid numeric data found in 'Close' or 'Volume' columns.")
