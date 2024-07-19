import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Load the data
data_path = 'Data/LymanDataForPred.csv'
data = pd.read_csv(data_path)
data['Date'] = pd.to_datetime(data['Date'])

# Fill missing 'ItemSold' and sort the data
data['ItemSold'] = data.groupby('Category Size')['ItemSold'].transform(lambda x: x.fillna(x.mean()))
data_sorted = data.sort_values(by=['Category Size', 'Date'])

# Resample to monthly data and interpolate missing values
data_monthly = data_sorted.groupby('Category Size').resample('M', on='Date')['ItemSold'].median().reset_index()
data_monthly['ItemSold'] = data_monthly.groupby('Category Size')['ItemSold'].transform(lambda x: x.interpolate())

# Forecast using Prophet for each Category Size and calculate accuracy
forecasts = pd.DataFrame()

# Forecast using Prophet for each Category Size
for CategorySize, group in data_monthly.groupby('Category Size'):
    if group['ItemSold'].notnull().sum() < 2:
        st.write(f"Skipping {CategorySize} due to insufficient data.")
        continue  # Skip to the next iteration

    # Prepare the data for Prophet
    df_prophet = pd.DataFrame({
        'ds': group['Date'],
        'y': group['ItemSold']
    })

    # Create and fit the Prophet model
    model = Prophet(yearly_seasonality=True,
                    weekly_seasonality=False,
                    daily_seasonality=False)
    model.fit(df_prophet)

    # Make a future dataframe for April to September 2024
    future = pd.date_range(start='2024-06-01', end='2024-08-01', freq='M')
    future = pd.DataFrame({'ds': future})

    # Forecast
    forecast = model.predict(future)

    # Add Category Size column to forecast dataframe
    forecast['Category Size'] = CategorySize

    # Append the results
    forecasts = pd.concat([forecasts, forecast], axis=0)

# Create an empty list to store the forecast data
forecast_output = []

# Iterate through each group of forecasts
for CategorySize, group in forecasts.groupby('Category Size'):
    # Iterate through each row in the group
    for index, row in group.iterrows():
        # Calculate the accuracy
        accuracy = row['yhat_upper'] - row['yhat_lower']
        # Calculate Accuracy Score
        accuracy_score = (accuracy / row['yhat']) * 100
        # Format accuracy score as percentage string
        accuracy_score_str = "{:.2f}%".format(accuracy_score)
        # Append the forecast data to the list
        forecast_output.append({
            'Category Size': CategorySize,
            'Date': row['ds'].strftime('%Y-%m-%d'),
            'Forecast': row['yhat'],
            'Lower': row['yhat_upper'],
            'Upper': row['yhat_lower'],
            'Confidence Interval': accuracy,
            'Accuracy Score': accuracy_score_str
        })

# Convert the list of dictionaries to a DataFrame
forecast_df = pd.DataFrame(forecast_output)

# Save the DataFrame to a CSV file
#forecast_df.to_csv('Data/ProphetForecastLymanCI.csv', index=False)

# Streamlit app
st.title('Prophet Forecast for Lyman Data')

# Sidebar for category selection
category_size = st.sidebar.selectbox('Select Category Size:', forecast_df['Category Size'].unique())

# Function to plot forecast for a selected category size
def plot_forecast(category_size):
    # Filter the DataFrame for the selected category size
    filtered_df = forecast_df[forecast_df['Category Size'] == category_size]

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(filtered_df['Date'], filtered_df['Forecast'], marker='o', linestyle='-')
    plt.fill_between(filtered_df['Date'], filtered_df['Lower'], filtered_df['Upper'], color='gray', alpha=0.2)
    plt.title(f'Forecast for {category_size}')
    plt.xlabel('Date')
    plt.ylabel('Forecast')
    plt.grid(True)
    st.pyplot(plt)

# Display the plot
plot_forecast(category_size)

# Display the forecast table
st.write(forecast_df[forecast_df['Category Size'] == category_size])
