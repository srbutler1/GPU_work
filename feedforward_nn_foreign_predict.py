import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from torch.amp import GradScaler, autocast
import logging
import time
import optuna

# Load data
data_path = r'X:\World Indices\ret_lag_merge.csv'
merged_data = pd.read_csv(data_path, parse_dates=["date"])

# Create a 'quarter' column based on the 'date' column
merged_data['quarter'] = merged_data['date'].dt.to_period('Q')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Prepare data: Sort by 'permno' and 'date' for shifting operations
merged_data = merged_data.sort_values(by=['permno', 'date'])

# Use 'ret' to create the target variable 'same_day_return'
merged_data['same_day_return'] = merged_data['ret']

# Define foreign signal columns (refining for lagged return patterns)
foreign_signal_features = [
    col for col in merged_data.columns if "Lag" in col
]

# Debugging step: Print the list of detected features
logging.info(f"Detected foreign signal features: {foreign_signal_features}")

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the neural network model
class StudyNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, dropout_rate):
        super(StudyNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, 64)
        self.fc4 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(dropout_rate)
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

# Define the Optuna objective function
def objective(trial):
    hidden_size1 = trial.suggest_int('hidden_size1', 64, 256)
    hidden_size2 = trial.suggest_int('hidden_size2', 64, 256)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)

    # Filter the last 252 days for training data for this stock
    stock_train_data = train_data[train_data['permno'] == stock_id]
    stock_test_data = test_data[test_data['permno'] == stock_id]

    if len(stock_train_data) < 252:
        return None

    # Prepare features (foreign signals) and target (same day's return)
    X_train = torch.tensor(stock_train_data[foreign_signal_features].values, dtype=torch.float32)
    y_train = torch.tensor(stock_train_data['same_day_return'].values, dtype=torch.float32).view(-1, 1)
    X_test = torch.tensor(stock_test_data[foreign_signal_features].values, dtype=torch.float32).to(device)
    y_test = torch.tensor(stock_test_data['same_day_return'].values, dtype=torch.float32).to(device)

    # DataLoader setup
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=16384, shuffle=True, pin_memory=True)

    # Initialize the model
    model = StudyNN(len(foreign_signal_features), hidden_size1, hidden_size2, dropout_rate).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scaler = GradScaler()

    # Training loop
    model.train()
    epochs = 75
    l1_lambda = 0.001

    for epoch in range(epochs):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)
            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                outputs = model(X_batch)
                mse_loss = criterion(outputs, y_batch)
                l1_loss = sum(p.abs().sum() for p in model.parameters())
                loss = mse_loss + l1_lambda * l1_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    # Evaluation on test data
    model.eval()
    with torch.no_grad():
        with autocast(device_type='cuda'):
            y_pred = model(X_test).flatten()

    # Calculate performance metrics
    r2_oo = 1 - torch.sum((y_test - y_pred) ** 2) / torch.sum((y_test - y_test.mean()) ** 2)
    return r2_oo.item()

# Main loop for training/testing across stocks and quarters
performance_metrics = []
quarters = merged_data['quarter'].unique()

for stock_id in merged_data['permno'].unique():
    for i in range(3, len(quarters) - 1):  # Start from the 4th quarter
        train_quarters = quarters[i - 3:i + 1]  # Last 4 quarters (~252 trading days)
        test_quarter = quarters[i + 1]  # Next quarter

        # Extract training data and check its completeness
        train_data = merged_data[(merged_data['permno'] == stock_id) & (merged_data['quarter'].isin(train_quarters))]
        if len(train_data) < 252:  # Skip if less than 252 days of training data
            continue

        # Extract testing data
        test_data = merged_data[(merged_data['permno'] == stock_id) & (merged_data['quarter'] == test_quarter)]

        # Ensure features are scaled properly
        scaler = StandardScaler()
        train_data.loc[:, foreign_signal_features] = scaler.fit_transform(train_data[foreign_signal_features])
        test_data.loc[:, foreign_signal_features] = scaler.transform(test_data[foreign_signal_features])

        # Run Optuna optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)

        # Best trial metrics
        best_trial = study.best_trial
        logging.info(f"Best trial for stock {stock_id} in quarter {test_quarter}: {best_trial.params}")

# Convert metrics to DataFrame and save
performance_df = pd.DataFrame(performance_metrics)
performance_df.to_csv(r'X:\performance_metrics.csv', index=False)
print("Performance metrics saved to performance_metrics.csv")

# Print the number of positive out-of-sample R^2 values
print(performance_df[performance_df['R2_oo'] > 0].shape[0])

# Print the percentage of positive out-of-sample R^2 values per stock 
positive_r2 = performance_df[performance_df['R2_oo'] > 0].groupby('permno').size()
print(positive_r2 / performance_df.groupby('permno').size())
