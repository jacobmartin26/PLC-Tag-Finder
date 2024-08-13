import sqlite3
import pandas as pd
from pycomm3 import LogixDriver, PycommError
from datetime import datetime
#from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import time
#import seaborn as sns
import matplotlib.pyplot as plt

def read_and_log_data():
    try:
        # Connect to the PLC
        with LogixDriver('192.168.0.10') as plc:
            if not plc.connected:
                raise ConnectionError("Failed to connect to the PLC")
            
            tags = plc.get_tag_list()
            
            with sqlite3.connect('plc_tags.db') as conn:
                cur = conn.cursor()
                
                # Create table if not exists (now with TEXT for tag_value)
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS tags (
                        timestamp TEXT,
                        tag_name TEXT,
                        tag_value TEXT
                    )
                ''')
                
                for tag in tags:
                    tag_name = tag['tag_name']
                    tag_value = str(plc.read(tag_name).value)  # Convert to string
                    cur.execute('''
                        INSERT INTO tags (timestamp, tag_name, tag_value) 
                        VALUES (?, ?, ?)
                    ''', (datetime.now().isoformat(), tag_name, tag_value))
                
                conn.commit()

        print("Data successfully read and logged.")

    except ConnectionError as e:
        print(f"Connection Error: {e}")
    except Exception as e:
        print(f"Failed to read and log data: {e}")

# Function to preprocess data
def preprocess_data():
    conn = sqlite3.connect('plc_tags.db')
    df = pd.read_sql_query("SELECT * FROM tags", conn)
    conn.close()

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Convert tag_value back to numeric, coercing errors to NaN
    df['tag_value'] = pd.to_numeric(df['tag_value'], errors='coerce')
    
    df['is_integer'] = df['tag_value'].apply(lambda x: float(x).is_integer() if pd.notnull(x) else False)
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['minute_of_hour'] = df['timestamp'].dt.minute
    df['hourly_change'] = df.groupby(['tag_name', 'hour_of_day'])['tag_value'].diff().rolling(window=60).sum()
    df['hourly_reset'] = df.groupby(['tag_name', 'hour_of_day'])['tag_value'].diff().rolling(window=60).apply(lambda x: 1 if np.sum(x) < 0 else 0)
    df['shift_reset'] = df.groupby('tag_name')['tag_value'].diff().rolling(window=480).apply(lambda x: 1 if np.sum(x) < 0 else 0)
    df['is_percentage'] = df['tag_value'].apply(lambda x: 1 if 0 <= x <= 1 else 0 if pd.notnull(x) else np.nan)
    df['value_range'] = df.groupby('tag_name')['tag_value'].transform(lambda x: x.max() - x.min())

    return df

# Function to train model
def train_model(df):
    features = ['is_integer', 'hourly_change', 'hourly_reset', 'shift_reset', 'is_percentage', 'value_range']
    X = df[features]
    y = df['tag_name']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, X_test, y_test

# Function to predict tags
def predict_and_visualize(model, X_test, y_test):
    # Get feature importances
    importances = model.feature_importances_
    feature_names = X_test.columns
    
    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    
    # Plot feature importances
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(X_test.shape[1]), importances[indices])
    plt.xticks(range(X_test.shape[1]), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.show()

    # Get probabilities for each class
    probabilities = model.predict_proba(X_test)
    
    # Get top 3 most likely classes for each sample
    top_3_classes = np.argsort(probabilities, axis=1)[:, -3:]
    
    # Print the top 3 most likely registers for each sample
    for i, (true_class, top_3) in enumerate(zip(y_test, top_3_classes)):
        print(f"Sample {i}:")
        print(f"True register: {true_class}")
        print("Top 3 predicted registers:")
        for j, class_idx in enumerate(top_3[::-1], 1):
            print(f"  {j}. {model.classes_[class_idx]} (Probability: {probabilities[i, class_idx]:.2f})")
        print()

def main():
    while True:
        try:
            # Read and log data
            read_and_log_data()

            # Preprocess data
            df = preprocess_data()

            # Train model
            model, X_test, y_test = train_model(df)

            # Predict and visualize results
            predict_and_visualize(model, X_test, y_test)

        except Exception as e:
            print(f"An error occurred in the main loop: {e}")

        # Wait for 30 minutes before the next iteration
        time.sleep(1800)

if __name__ == "__main__":
    main()