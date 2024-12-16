import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from torch.amp import GradScaler, autocast
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load foreign indices data
indices_data = pd.read_csv(r'X:\foreign_indices_filtered.csv', parse_dates=['datadate'])

# Load the selected stock data with daily returns for 10 years
stocks_data = pd.read_csv(r'X:\services_stocks.csv', parse_dates=['date'])

# Define selected stock IDs
selected_permnos = stocks_data['permno'].unique()

# Create feature and target columns for 8 weeks of lagged returns if needed
features = [f'lag_{lag}w_return' for lag in range(1, 9)]
if not all(col in stocks_data.columns for col in features):
    stocks_data.sort_values(['permno', 'date'], inplace=True)
    for lag in range(1, 9):
        stocks_data[f'lag_{lag}w_return'] = stocks_data.groupby('permno')['ret'].shift(lag)

# Ensure we have the dependent variable `next_day_return`
stocks_data['next_day_return'] = stocks_data.groupby('permno')['ret'].shift(-1)

# Drop rows with NaN after shifting
stocks_data = stocks_data.dropna(subset=['next_day_return'] + features)

# Merge with indices data on dates
merged_data = pd.merge(stocks_data, indices_data, left_on='date', right_on='datadate', how='inner')

# Add a 'quarter' column for segmentation
merged_data['quarter'] = merged_data['date'].dt.to_period('Q')

# Set up the device for GPU computation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the neural network model optimized for GPU
class StudyNN(nn.Module):
    def __init__(self, input_size):
        super(StudyNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x

# Initialize scaler and mixed precision
scaler_features = MinMaxScaler()
scaler = GradScaler()

# Standardize lagged features in chunks to avoid memory issues
valid_features = [col for col in features if col in merged_data.columns]

if valid_features and not merged_data[valid_features].isnull().all().all():
    for start in range(0, len(merged_data), 50000):
        end = min(start + 50000, len(merged_data))
        if len(merged_data.loc[start:end, valid_features]) > 0:
            merged_data.loc[start:end, valid_features] = scaler_features.fit_transform(
                merged_data.loc[start:end, valid_features]
            )
else:
    print("No valid features or empty data found for scaling.")

# List to store performance metrics
performance_metrics = []

# Get unique quarters for out-of-sample training/testing
quarters = merged_data['quarter'].unique()

def train_and_predict(stock_id, train_data, test_data, test_quarter):
    stock_train_data = train_data[train_data['permno'] == stock_id]
    stock_test_data = test_data[test_data['permno'] == stock_id]

    if len(stock_train_data) < 2 or len(stock_test_data) < 2:
        return None

    # Prepare data as PyTorch tensors for DataLoader
    X_train = torch.tensor(stock_train_data[valid_features].values, dtype=torch.float32)  # Keep on CPU
    y_train = torch.tensor(stock_train_data['next_day_return'].values, dtype=torch.float32).view(-1, 1)  # Keep on CPU
    X_test = torch.tensor(stock_test_data[valid_features].values, dtype=torch.float32).to(device)
    y_test = stock_test_data['next_day_return'].values  # For performance metric calculations

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True, num_workers=0, pin_memory=True)

    # Define model, loss function, and optimizer
    model = StudyNN(input_size=len(valid_features)).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # Train the model for 100 epochs
    model.train()
    epochs = 100
    for epoch in range(epochs):
        for X_batch, y_batch in train_loader:
            # Move each batch to GPU
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    # Predict next-day returns using the trained model
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test).cpu().numpy().flatten()

    # Calculate out-of-sample R^2 (R2_oo)
    r2_oo = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    bias = np.mean(y_pred - y_test)

    # Store metrics for this stock and quarter
    performance_metrics.append({
        'permno': stock_id, 
        'quarter': test_quarter, 
        'R2_oo': r2_oo,
        'MAE': mae,
        'RMSE': rmse,
        'Bias': bias
    })

# Main training loop across quarters
for i in range(3, len(quarters) - 1):
    train_quarters = quarters[:i + 1]
    test_quarter = quarters[i + 1]

    train_data = merged_data[merged_data['quarter'].isin(train_quarters)]
    test_data = merged_data[merged_data['quarter'] == test_quarter]

    if train_data.empty or test_data.empty:
        continue

    # Only process the selected stocks for each quarter
    for stock_id in selected_permnos:
        train_and_predict(stock_id, train_data, test_data, test_quarter)

# Convert metrics to DataFrame and calculate mean statistics per stock
performance_df = pd.DataFrame(performance_metrics)
performance_summary = performance_df.groupby('permno').agg({
    'R2_oo': ['mean', lambda x: (x > 0).mean()],
    'MAE': 'mean',
    'RMSE': 'mean',
    'Bias': 'mean'
})
performance_summary.columns = ['Mean_R2_oo', 'Fraction_Positive_R2', 'Mean_MAE', 'Mean_RMSE', 'Mean_Bias']

print("Performance Summary for Selected Stocks:")
print(performance_summary)

# Save the summary to a CSV file if needed
performance_summary.to_csv(r'X:\performance_summary_selected_stocks.csv')



