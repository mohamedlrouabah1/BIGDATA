import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load the data
data_path = 'path/to/task1_data_directory'
external_data = pd.read_csv(f'{data_path}/external.csv')
blockchain_global_data = pd.read_csv(f'{data_path}/blockchain_global.csv')
blockchain_by_actor_data = pd.read_csv(f'{data_path}/blockchain_by_actor.csv')

def feature_engineering(data):
    # Add feature engineering steps here
    # For simplicity, let's use lag features
    for i in range(1, 6):
        data[f'PriceUSD_lag_{i}'] = data['PriceUSD'].shift(i)
    return data

merged_data = pd.merge(external_data, blockchain_global_data, on=['week', 'weekday'])
merged_data = pd.merge(merged_data, blockchain_by_actor_data, on=['week', 'weekday'])

merged_data = feature_engineering(merged_data)

features = merged_data.drop(['week', 'weekday', 'PriceUSD'], axis=1)
target = (merged_data['PriceUSD'].shift(-6) > merged_data['PriceUSD']).astype(int)

X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

predictions = model.predict(X_val_scaled)

accuracy = accuracy_score(y_val, predictions)
print(f'Accuracy on the validation set: {accuracy}')

# Now,  use the trained model to make predictions on the test set and save the results in the required format.

