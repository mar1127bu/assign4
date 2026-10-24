# Import necessary packages (Make sure to also include in "requirements.txt" in this repository.)
import streamlit as st
import pandas as pd
import statsmodels.api as sm

# Print Title for the App
st.title("Microsoft Stock Data Dashboard by Michael R")

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
    st.caption("Ordinary Least Squares (OLS) regression")

    # Prepare clean, numeric data
    reg_df = df[['Close', 'Volume']].copy()
    reg_df['Close'] = pd.to_numeric(reg_df['Close'], errors='coerce')
    reg_df['Volume'] = pd.to_numeric(reg_df['Volume'], errors='coerce')
    reg_df = reg_df.dropna()

    if len(reg_df) > 2:
        y = reg_df["Close"]
        X = sm.add_constant(reg_df[["Volume"]])
        model = sm.OLS(y, X).fit()

        # Key model statistics
        col1, col2 = st.columns(2)
        col1.metric("R-squared", f"{model.rsquared:.3f}")
        col2.metric("Observations", f"{int(model.nobs):,}")

        col3, col4 = st.columns(2)
        col3.metric("F-statistic", f"{model.fvalue:.2f}")
        col4.metric("Model p-value", f"{model.f_pvalue:.2e}")

        st.divider()

        # Regression coefficients
        st.markdown("### Regression Coefficients")

        coef_table = pd.DataFrame({
            "Variable": ["Intercept", "Volume"],
            "Coefficient": [
                model.params["const"],
                model.params["Volume"]
            ],
            "Standard Error": [
                model.bse["const"],
                model.bse["Volume"]
            ],
            "t-statistic": [
                model.tvalues["const"],
                model.tvalues["Volume"]
            ],
            "P-value": [
                model.pvalues["const"],
                model.pvalues["Volume"]
            ]
        })

        st.dataframe(
            coef_table.style.format({
                "Coefficient": "{:.6f}",
                "Standard Error": "{:.6f}",
                "t-statistic": "{:.3f}",
                "P-value": "{:.4g}"
            }),
            hide_index=True,
            use_container_width=True
        )

        # Regression equation
        intercept = model.params["const"]
        slope = model.params["Volume"]

        st.markdown("### Regression Equation")
        st.latex(
            rf"\widehat{{Close}} = {intercept:.2f} "
            rf"{slope:+.3e} \times Volume"
        )

        # Interpretation
        st.markdown("### Interpretation")
        st.write(
            f"The model explains {model.rsquared:.1%} of the "
            "variation in closing prices, based on trading volume."
        )

        if model.f_pvalue < 0.05:
            st.write(
                "The overall regression is statistically significant "
                "at the 5% significance level."
            )
        else:
            st.write(
                "The overall regression is not statistically significant "
                "at the 5% significance level."
            )

        # Optional detailed output
        with st.expander("View Full Statistical Summary"):
            st.text(str(model.summary()))

    else:
        st.error(
            "Unable to run regression: Not enough valid numeric data "
            "in the Close and Volume columns."
        )

