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

    # Prepare data with representative quarters
    representative_quarters = quarters[:4]  # First 4 quarters as representative
    train_data = merged_data[(merged_data['permno'] == stock_id) & (merged_data['quarter'].isin(representative_quarters))]

    if len(train_data) < 252:
        return None

    # Prepare features and target
    X_train = torch.tensor(train_data[foreign_signal_features].values, dtype=torch.float32)
    y_train = torch.tensor(train_data['same_day_return'].values, dtype=torch.float32).view(-1, 1)

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

    # Return a dummy R^2 value for validation
    return -1.0  # Placeholder until real validation is implemented

# Main loop for optimization per stock
optimized_params = {}
quarters = merged_data['quarter'].unique()

for stock_id in merged_data['permno'].unique():
    # Run Optuna optimization once for each stock
    logging.info(f"Optimizing hyperparameters for stock {stock_id}...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    # Save best trial parameters
    optimized_params[stock_id] = study.best_trial.params
    logging.info(f"Best parameters for stock {stock_id}: {study.best_trial.params}")

# Main loop for training/testing using optimized hyperparameters
performance_metrics = []

for stock_id in merged_data['permno'].unique():
    best_params = optimized_params.get(stock_id)
    if not best_params:
        continue

    for i in range(3, len(quarters) - 1):
        train_quarters = quarters[i - 3:i + 1]  # Last 4 quarters
        test_quarter = quarters[i + 1]  # Next quarter

        train_data = merged_data[(merged_data['permno'] == stock_id) & (merged_data['quarter'].isin(train_quarters))]
        if len(train_data) < 252:
            continue

        test_data = merged_data[(merged_data['permno'] == stock_id) & (merged_data['quarter'] == test_quarter)]

        # Scale data
        scaler = StandardScaler()
        train_data.loc[:, foreign_signal_features] = scaler.fit_transform(train_data[foreign_signal_features])
        test_data.loc[:, foreign_signal_features] = scaler.transform(test_data[foreign_signal_features])

        # Train and evaluate model with best parameters
        model = StudyNN(len(foreign_signal_features), best_params['hidden_size1'], best_params['hidden_size2'], best_params['dropout_rate']).to(device)
        
        # Prepare data
        X_train = torch.tensor(train_data[foreign_signal_features].values, dtype=torch.float32).to(device)
        y_train = torch.tensor(train_data['same_day_return'].values, dtype=torch.float32).view(-1, 1).to(device)
        X_test = torch.tensor(test_data[foreign_signal_features].values, dtype=torch.float32).to(device)
        y_test = torch.tensor(test_data['same_day_return'].values, dtype=torch.float32).to(device)

        optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'], weight_decay=1e-5)
        criterion = nn.MSELoss()
        model.train()
        for epoch in range(75):
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

        # Evaluation
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test).flatten()
        r2_oo = 1 - torch.sum((y_test - y_pred) ** 2) / torch.sum((y_test - y_test.mean()) ** 2)
        
        performance_metrics.append({
            'permno': stock_id,
            'quarter': test_quarter,
            'R2_oo': r2_oo.item()
        })

# Convert metrics to DataFrame and save
performance_df = pd.DataFrame(performance_metrics)
performance_df.to_csv(r'X:\performance_metrics.csv', index=False)
print("Performance metrics saved to performance_metrics.csv")

# Print the number of positive out-of-sample R^2 values
print(performance_df[performance_df['R2_oo'] > 0].shape[0])

# Print the percentage of positive out-of-sample R^2 values per stock 
positive_r2 = performance_df[performance_df['R2_oo'] > 0].groupby('permno').size()
print(positive_r2 / performance_df.groupby('permno').size())
