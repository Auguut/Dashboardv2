from openbb_terminal.sdk import openbb
from typing import Union
import streamlit as st
import datetime
import pandas as pd
from st_aggrid import AgGrid
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def lambda_long_number_format(num, round_decimal=3) -> Union[str, int, float]:
    """Format a long number."""
    if num == float("inf"):
        return "inf"

    if isinstance(num, float):
        magnitude = 0
        while abs(num) >= 1000:
            magnitude += 1
            num /= 1000.0

        string_fmt = f".{round_decimal}f"
        num_str = int(num) if num.is_integer() else f"{num:{string_fmt}}"

        return f"{num_str} {' KMBTP'[magnitude]}".strip()

    if isinstance(num, int):
        num = str(num)

    if (
        isinstance(num, str)
        and num.lstrip("-").isdigit()
        and not num.lstrip("-").startswith("0")
    ):
        num = int(num)
        num /= 1.0
        magnitude = 0
        while abs(num) >= 1000:
            magnitude += 1
            num /= 1000.0

        string_fmt = f".{round_decimal}f"
        num_str = int(num) if num.is_integer() else f"{num:{string_fmt}}"

        return f"{num_str} {' KMBTP'[magnitude]}".strip()

    return num

# Candlestick code from 
# https://medium.com/@dannygrovesn7/using-streamlit-and-plotly-to-create-interactive-candlestick-charts-a2a764ad0d8e

ma1=10
ma2=30
days_to_plot=120

def get_candlestick_plot(
        df: pd.DataFrame,
        ma1: int,
        ma2: int,
        ticker: str,
        transaction_dates_sales: list = [],
        transaction_dates_purchases: list = []
):
    
    fig = make_subplots(
        rows = 2,
        cols = 1,
        shared_xaxes = True,
        vertical_spacing = 0.1,
        subplot_titles = (f'{ticker} Stock Price', 'Volume Chart'),
        row_width = [0.3, 0.7],
        row_heights=150
    )
    
    fig.add_trace(
        go.Candlestick(
            x = df['Date'],
            open = df['Open'], 
            high = df['High'],
            low = df['Low'],
            close = df['Close'],
            name = 'Candlestick chart'
        ),
        row = 1,
        col = 1,
    )
    
    fig.add_trace(
        go.Line(x = df['Date'], y = df[f'{ma1}_ma'], name = f'{ma1} SMA'),
        row = 1,
        col = 1,
    )
    
    fig.add_trace(
        go.Line(x = df['Date'], y = df[f'{ma2}_ma'], name = f'{ma2} SMA'),
        row = 1,
        col = 1,
    )

    for data in transaction_dates_sales:
        fig.add_vline(x=data, line_width=1, line_color="red")

    for data in transaction_dates_purchases:
        fig.add_vline(x=data, line_width=1, line_color="green")
    
    fig.add_trace(
        go.Bar(x = df['Date'], y = df['Volume'], name = 'Volume'),
        row = 2,
        col = 1,
    )
    
    fig['layout']['xaxis2']['title'] = 'Date'
    fig['layout']['yaxis']['title'] = 'Price'
    fig['layout']['yaxis2']['title'] = 'Volume'
    
    fig.update_xaxes(
        rangebreaks = [{'bounds': ['sat', 'mon']}],
        rangeslider_visible = False,
    )

    
    return fig
    

st.set_page_config(layout="wide")


homeTab, discoveryTab, Financial_statements, Economies, Government = st.tabs(["Home", "Discovery", "Financial Statements", "Economies", "Government"])


with homeTab:
    ticker = st.text_input('Symbol', key='homeTab_ticker')


    if ticker:
        data = openbb.stocks.load(ticker)

        if len(data) > 0:
            df = data.reset_index()
            df.columns = [x.title() for x in df.columns]
            df[f'{ma1}_ma'] = df['Close'].rolling(ma1).mean()
            df[f'{ma2}_ma'] = df['Close'].rolling(ma2).mean()
            df = df[-days_to_plot:]

            # Display the plotly chart on the dashboard
            st.plotly_chart(
                get_candlestick_plot(df, ma1, ma2, ticker),
                width=0, height=0,
                use_container_width=True,
            )

            col1, col2 = st.columns(2)

            with col1:
                st.subheader('Company supplier')
                st.dataframe(openbb.stocks.fa.supplier(ticker))

            with col2:
                st.subheader('Company customer')
                st.dataframe(openbb.stocks.fa.customer(ticker))

            col1, col2 = st.columns(2)

            with col1:
                st.subheader('Options information')
                st.dataframe(openbb.stocks.options.info(ticker))

            with col2:
                st.subheader('Put/Call Ratio (PCR)')

                def plot_pcr_for_symbol(symbol: str):
                    # Fetch PCR data
                    pcr_data = openbb.stocks.options.pcr(symbol)

                    # Create a plotly graph
                    fig = go.Figure()

                    # Add a line trace for PCR values
                    fig.add_trace(go.Scatter(x=pcr_data.index, y=pcr_data['PCR'], mode='lines', name='PCR'))

                    # Set the title and labels
                    fig.update_layout(title=f'Put/Call Ratio (PCR) for {symbol}', xaxis_title='Date', yaxis_title='PCR')

                    # Return the plot
                    return fig

                # Call the function to plot PCR for the ticker and display it in Streamlit
                st.plotly_chart(plot_pcr_for_symbol(ticker), use_container_width=True)

            st.subheader('Insider trading')
            st.dataframe(openbb.stocks.ins.stats(ticker))

            
with discoveryTab:
    st.subheader('Latest Top Gainers')
    gainers = openbb.stocks.disc.gainers()
    st.dataframe(gainers)

    st.subheader('Latest Top Losers')
    losers = openbb.stocks.disc.losers()
    st.dataframe(losers)

    st.subheader('Stocks with Highest Trade Volumes')
    active = openbb.stocks.disc.active()
    st.dataframe(active)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Hot Penny Stocks')
        hotpenny = openbb.stocks.disc.hotpenny()
        st.dataframe(hotpenny)
        
    with col2:
        st.subheader('Tech Stocks with Earnings Growth Over 25%')
        gtech = openbb.stocks.disc.gtech()
        st.dataframe(gtech)

    st.subheader('Last insider purcharses')
    Purcharses = openbb.stocks.ins.blip().head(25)
    st.dataframe(Purcharses)

    st.subheader('Last insider sales')
    Sales = openbb.stocks.ins.blis().head(25)
    st.dataframe(Sales)

    st.subheader('Whales tracker')
    Whales = openbb.stocks.ins.filter("whales")
    st.dataframe(Whales)

    st.subheader('Unusual options')
    unu_df,unu_ts = openbb.stocks.options.unu(limit = 500)
    unu_df = unu_df.sort_values(by = 'Vol/OI', ascending = False)
    unu_df

    st.subheader('Upcoming Earnings Release Dates')
    upcoming = openbb.stocks.disc.upcoming()
    st.dataframe(upcoming)

    st.subheader('Upcoming Dividend Payment')
    dividends = openbb.stocks.disc.dividends()
    st.dataframe(dividends)

with Financial_statements:
    ticker = st.text_input('Symbol', key='Financial_statements_ticker')

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.subheader('Overview')
        overview = openbb.stocks.fa.overview((ticker))
        
        # Check if overview is a list and convert it to a DataFrame if it is
        if isinstance(overview, list):
            overview = pd.DataFrame(overview)
        
        # Format the numbers in the dataframe
        for col in overview.columns:
            overview[col] = overview[col].apply(lambda x: lambda_long_number_format(x))
        st.dataframe(overview)

    with col2:
        st.subheader('Key ratios')
        Key_ratio = openbb.stocks.fa.key((ticker))
        
        # Check if Key_ratio is a list and convert it to a DataFrame if it is
        if isinstance(Key_ratio, list):
            Key_ratio = pd.DataFrame(Key_ratio)
        
        # Format the numbers in the dataframe
        for col in Key_ratio.columns:
            Key_ratio[col] = Key_ratio[col].apply(lambda x: lambda_long_number_format(x))
        st.dataframe(Key_ratio)

    with col3:
        st.subheader('Key metrics')
        Key_metrics = openbb.stocks.fa.metrics((ticker))
        
        # Check if Key_metrics is a list and convert it to a DataFrame if it is
        if isinstance(Key_metrics, list):
            Key_metrics = pd.DataFrame(Key_metrics)
        
        # Format the numbers in the dataframe
        for col in Key_metrics.columns:
            Key_metrics[col] = Key_metrics[col].apply(lambda x: lambda_long_number_format(x))
        st.dataframe(Key_metrics)


    with col4:
        st.subheader('Analyst price targets')
        Price_targets = openbb.stocks.fa.pt((ticker))
        
        # Check if Price_targets is a list and convert it to a DataFrame if it is
        if isinstance(Price_targets, list):
            Price_targets = pd.DataFrame(Price_targets)
        
        # Format the numbers in the dataframe
        for col in Price_targets.columns:
            Price_targets[col] = Price_targets[col].apply(lambda x: lambda_long_number_format(x))
        st.dataframe(Price_targets)


    if ticker:
        st.subheader('Income statement')
        income_statement = openbb.stocks.fa.income(ticker, source="AlphaVantage", quarterly=True)
        
        # Check if income_statement is a DataFrame before proceeding
        if isinstance(income_statement, pd.DataFrame):
            # Convert numeric columns to float type
            for col in income_statement.columns:
                income_statement[col] = pd.to_numeric(income_statement[col], errors='ignore')
            
            # Format the numbers in the dataframe using the lambda_long_number_format function
            for col in income_statement.columns:
                income_statement[col] = income_statement[col].apply(lambda x: lambda_long_number_format(x))

            st.dataframe(income_statement)

        st.subheader('Balance Sheet')
        balance_sheet = openbb.stocks.fa.balance(ticker, source="AlphaVantage", quarterly=True)

        # Check if balance_sheet is a DataFrame before proceeding
        if isinstance(balance_sheet, pd.DataFrame):
            # Convert numeric columns to float type
            for col in balance_sheet.columns:
                balance_sheet[col] = pd.to_numeric(balance_sheet[col], errors='ignore')

            # Format the numbers in the dataframe using the lambda_long_number_format function
            for col in balance_sheet.columns:
                balance_sheet[col] = balance_sheet[col].apply(lambda x: lambda_long_number_format(x))

            st.dataframe(balance_sheet)

        st.subheader('Cash Flow Statement')
        cash_flow = openbb.stocks.fa.cash(ticker, source="AlphaVantage", quarterly=True)

        # Check if cash_flow is a DataFrame before proceeding
        if isinstance(cash_flow, pd.DataFrame):
            # Convert numeric columns to float type
            for col in cash_flow.columns:
                cash_flow[col] = pd.to_numeric(cash_flow[col], errors='ignore')

            # Format the numbers in the dataframe using the lambda_long_number_format function
            for col in cash_flow.columns:
                cash_flow[col] = cash_flow[col].apply(lambda x: lambda_long_number_format(x))

            st.dataframe(cash_flow)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader('Earnings estimate')
            earnings = openbb.stocks.fa.earnings((ticker))
            st.dataframe(earnings)

        with col2:
            st.subheader('Dividend payment')
            
            # Fetch the dividend data
            divs_data = openbb.stocks.fa.divs(ticker)
            
            # Create a plotly graph
            fig = go.Figure()

            # Add a line trace for dividend values
            fig.add_trace(go.Scatter(x=divs_data.index, y=divs_data.iloc[:, 0], mode='lines', name='Dividend'))

            # Set the title and labels
            fig.update_layout(title=f'Dividend Over Time for {ticker}', xaxis_title='Date', yaxis_title='Dividend Amount')

            # Display the plot in Streamlit
            st.plotly_chart(fig, use_container_width=True)

        # This section will span across both columns since it's outside the 'with' blocks for col1 and col2
        st.subheader('SEC Analysis')
        SEC_analysis = openbb.stocks.fa.analysis((ticker))
        st.dataframe(SEC_analysis)


with Economies:
    st.subheader('International index')
    # Dictionary mapping short index names to their full descriptive names
    index_names = {
    'sp500': 'S&P 500',
    'sp400': 'S&P 400 Mid Cap',
    'sp600': 'S&P 600 Small Cap',
    'sp500tr': 'S&P 500 TR',
    'sp_xsp': 'S&P 500 Mini SPX Options',
    'ca_banks': 'S&P/TSX Composite Banks',
    'ar_mervel': 'S&P MERVAL TR',
    'eu_speup': 'S&P Europe 350',
    'uk_spuk': 'S&P United Kingdom',
    'in_bse': 'S&P Bombay'
    }
    # Fetch the economy index data
    df = openbb.economy.index(indices=list(index_names.keys()))
    df = df.tail(3000)

    # Create a plotly graph
    fig = go.Figure()

    # Iterate over each column in the dataframe and add a trace for each index
    for col in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=index_names[col]))

    # Set the title and labels
    fig.update_layout(title='Economy Indices Over Time', xaxis_title='Date', yaxis_title='Index Value')

    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    st.subheader('Sector performance')
    # Fetch the economy performance data for sectors
    df = openbb.economy.performance(group='sector')

    # Define the function to format values
    def format_value(x):
        try:
            val = float(x)
            if -1 <= val <= 1:
                return f"{val*100:.2f}%"
            return x
        except ValueError:
            return x

    # Apply the formatting function to the dataframe
    df = df.applymap(format_value)

    # Display the formatted dataframe in Streamlit
    st.dataframe(df)

    st.subheader('10 years interest rate')
    # Fetch the macroeconomic data
    data, units, denomination = openbb.economy.macro(parameters=['Y10YD'], countries=['United States', 'China', 'France', 'Italy', 'Spain', 'Germany', 'United Kingdom'])

    # Create a plotly graph
    fig = go.Figure()

    # Iterate over each column in the dataframe and add a trace for each country's Y10YD data
    for country in data.columns:
        # Convert the tuple to a string for the name property
        country_name = ' '.join(country) if isinstance(country, tuple) else country
        fig.add_trace(go.Scatter(x=data.index, y=data[country], mode='lines', name=country_name))

    # Set the title and labels
    fig.update_layout(title='Macroeconomic Data Over Time', xaxis_title='Date', yaxis_title='Y10YD Value')

    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)


    # Display a subheader in your Streamlit app for 3 Month Yield
    st.subheader('3 Month Yield (M3YD)')

    # Fetch the macroeconomic data for M3YD
    data, units, denomination = openbb.economy.macro(parameters=['M3YD'], countries=['United States', 'China', 'France', 'Italy', 'Spain', 'Germany', 'United Kingdom'])

    # Create a plotly graph
    fig = go.Figure()

    # Iterate over each column in the dataframe and add a trace for each country's M3YD data
    for country in data.columns:
        country_name = ' '.join(country) if isinstance(country, tuple) else country
        fig.add_trace(go.Scatter(x=data.index, y=data[country], mode='lines', name=country_name))

    # Set the title and labels
    fig.update_layout(title='3 Month Yield (M3YD) Over Time', xaxis_title='Date', yaxis_title='M3YD Value')

    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)


    # Display a subheader in your Streamlit app
    st.subheader('Real Gross Domestic Product (RGDP)')

    # Fetch the macroeconomic data for RGDP
    data, units, denomination = openbb.economy.macro(parameters=['RGDP'], countries=['United States', 'China', 'France', 'Italy', 'Spain', 'Germany', 'United Kingdom'])

    # Create a plotly graph
    fig = go.Figure()

    # Iterate over each column in the dataframe and add a trace for each country's RGDP data
    for country in data.columns:
        # Convert the tuple to a string for the name property
        country_name = ' '.join(country) if isinstance(country, tuple) else country
        fig.add_trace(go.Scatter(x=data.index, y=data[country], mode='lines', name=country_name))

    # Set the title and labels
    fig.update_layout(title='Real Gross Domestic Product (RGDP) Over Time', xaxis_title='Date', yaxis_title='RGDP Value')

    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)


    # Display a subheader in your Streamlit app
    st.subheader('Consumer Price Index (CPI)')

    # Fetch the macroeconomic data for CPI
    data, units, denomination = openbb.economy.macro(parameters=['CPI'], countries=['United States', 'China', 'France', 'Italy', 'Spain', 'Germany', 'United Kingdom','India', 'Brazil'])

    # Create a plotly graph
    fig = go.Figure()

    # Iterate over each column in the dataframe and add a trace for each country's CPI data
    for country in data.columns:
        # Convert the tuple to a string for the name property
        country_name = ' '.join(country) if isinstance(country, tuple) else country
        fig.add_trace(go.Scatter(x=data.index, y=data[country], mode='lines', name=country_name))

    # Set the title and labels
    fig.update_layout(title='Consumer Price Index (CPI) Over Time', xaxis_title='Date', yaxis_title='CPI Value')

    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # Display a subheader in your Streamlit app
    st.subheader('Core Consumer Price Index (CORE)')

    # Fetch the macroeconomic data for CORE
    data, units, denomination = openbb.economy.macro(parameters=['CORE'], countries=['United States', 'China', 'France', 'Italy', 'Spain', 'Germany', 'United Kingdom'])

    # Create a plotly graph
    fig = go.Figure()

    # Iterate over each column in the dataframe and add a trace for each country's CORE data
    for country in data.columns:
        # Convert the tuple to a string for the name property
        country_name = ' '.join(country) if isinstance(country, tuple) else country
        fig.add_trace(go.Scatter(x=data.index, y=data[country], mode='lines', name=country_name))

    # Set the title and labels
    fig.update_layout(title='Core Consumer Price Index (CORE) Over Time', xaxis_title='Date', yaxis_title='CORE Value')

    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)


    # Display a subheader in your Streamlit app
    st.subheader('Producer Price Index (PPI)')

    # Fetch the macroeconomic data for PPI
    data, units, denomination = openbb.economy.macro(parameters=['PPI'], countries=['United States', 'China', 'France', 'Italy', 'Spain', 'Germany', 'United Kingdom', 'India', 'Brazil'])

    # Create a plotly graph
    fig = go.Figure()

    # Iterate over each column in the dataframe and add a trace for each country's PPI data
    for country in data.columns:
        # Convert the tuple to a string for the name property
        country_name = ' '.join(country) if isinstance(country, tuple) else country
        fig.add_trace(go.Scatter(x=data.index, y=data[country], mode='lines', name=country_name))

    # Set the title and labels
    fig.update_layout(title='Producer Price Index (PPI) Over Time', xaxis_title='Date', yaxis_title='PPI Value')

    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # Display a subheader in your Streamlit app
    st.subheader('Government Debt (GDEBT)')

    # Fetch the macroeconomic data for GDEBT
    data, units, denomination = openbb.economy.macro(parameters=['GDEBT'], countries=['United States', 'China', 'France', 'Italy', 'Spain', 'Germany', 'United Kingdom'])

    # Create a plotly graph
    fig = go.Figure()

    # Iterate over each column in the dataframe and add a trace for each country's GDEBT data
    for country in data.columns:
        # Convert the tuple to a string for the name property
        country_name = ' '.join(country) if isinstance(country, tuple) else country
        fig.add_trace(go.Scatter(x=data.index, y=data[country], mode='lines', name=country_name))

    # Set the title and labels
    fig.update_layout(title='Government Debt (GDEBT) Over Time', xaxis_title='Date', yaxis_title='GDEBT Value')

    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # Display a subheader in your Streamlit app for Current Account Balance
    st.subheader('Current Account Balance (CA)')

    # Fetch the macroeconomic data for CA
    data, units, denomination = openbb.economy.macro(parameters=['CA'], countries=['United States', 'China', 'France', 'Italy', 'Spain', 'Germany', 'United Kingdom'])

    # Create a plotly graph
    fig = go.Figure()

    # Iterate over each column in the dataframe and add a trace for each country's CA data
    for country in data.columns:
        country_name = ' '.join(country) if isinstance(country, tuple) else country
        fig.add_trace(go.Scatter(x=data.index, y=data[country], mode='lines', name=country_name))

    # Set the title and labels
    fig.update_layout(title='Current Account Balance (CA) Over Time', xaxis_title='Date', yaxis_title='CA Value')

    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # Display a subheader in your Streamlit app for Trade Balance
    st.subheader('Trade Balance (TB)')

    # Fetch the macroeconomic data for TB
    data, units, denomination = openbb.economy.macro(parameters=['TB'], countries=['United States', 'China', 'France', 'Italy', 'Spain', 'Germany', 'United Kingdom'])

    # Create a plotly graph
    fig = go.Figure()

    # Iterate over each column in the dataframe and add a trace for each country's TB data
    for country in data.columns:
        country_name = ' '.join(country) if isinstance(country, tuple) else country
        fig.add_trace(go.Scatter(x=data.index, y=data[country], mode='lines', name=country_name))

    # Set the title and labels
    fig.update_layout(title='Trade Balance (TB) Over Time', xaxis_title='Date', yaxis_title='TB Value')

    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # Display a subheader in your Streamlit app for House Price Index
    st.subheader('House Price Index (HOU)')

    # Fetch the macroeconomic data for HOU
    data, units, denomination = openbb.economy.macro(parameters=['HOU'], countries=['United States', 'China', 'France', 'Italy', 'Spain', 'Germany', 'United Kingdom'])

    # Create a plotly graph
    fig = go.Figure()

    # Iterate over each column in the dataframe and add a trace for each country's HOU data
    for country in data.columns:
        country_name = ' '.join(country) if isinstance(country, tuple) else country
        fig.add_trace(go.Scatter(x=data.index, y=data[country], mode='lines', name=country_name))

    # Set the title and labels
    fig.update_layout(title='House Price Index (HOU) Over Time', xaxis_title='Date', yaxis_title='HOU Value')

    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # Display a subheader in your Streamlit app for Government Balance
    st.subheader('Government Balance (GBAL)')

    # Fetch the macroeconomic data for GBAL
    data, units, denomination = openbb.economy.macro(parameters=['GBAL'], countries=['United States', 'China', 'France', 'Italy', 'Spain', 'Germany', 'United Kingdom'])

    # Create a plotly graph
    fig = go.Figure()

    # Iterate over each column in the dataframe and add a trace for each country's GBAL data
    for country in data.columns:
        country_name = ' '.join(country) if isinstance(country, tuple) else country
        fig.add_trace(go.Scatter(x=data.index, y=data[country], mode='lines', name=country_name))

    # Set the title and labels
    fig.update_layout(title='Government Balance (GBAL) Over Time', xaxis_title='Date', yaxis_title='GBAL Value')

    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)


    st.subheader('Upcoming calendar')
    upcoming_calendar = openbb.economy.events().head(100)
    st.dataframe(upcoming_calendar)

with Government:
    # Ticker input
    ticker = st.text_input('Enter Ticker:', value='AAPL').upper()

    # Display contracts awarded to the company
    st.subheader(f'Contracts Awarded to {ticker}')
    contracts = openbb.stocks.gov.contracts(ticker)
    st.write(contracts)

    # Display reported trades in the company's stock by members of the US Congress and Senate
    st.subheader(f'Reported Trades in {ticker} by Members of the US Congress and Senate')
    gtrades = openbb.stocks.gov.gtrades(ticker)
    st.write(gtrades)

    st.subheader('Latest Reported Trades Made by Members of the US Congress and Senate')
    lasttrades = openbb.stocks.gov.lasttrades()
    st.write(lasttrades)

    st.subheader('The Top Buyers in Office')
    topbuys = openbb.stocks.gov.topbuys()
    st.write(topbuys)

    st.subheader('The Top Sellers in Office')
    topsellers = openbb.stocks.gov.topsells()
    st.write(topsellers)
        
    st.subheader('Corporate Lobbyist Activity')
    toplobbying = openbb.stocks.gov.toplobbying()
    st.write(toplobbying)
