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
        
def get_user_input():
    tags_to_predict = []
    while True:
        tag_label = input("Enter a label for the tag you want to predict (or 'done' to finish): ")
        if tag_label.lower() == 'done':
            break
        
        tag_info = {
            'label': tag_label,
            'is_integer': input("Is this tag an integer? (Y/N): ").lower() == 'y',
            'is_percentage': input("Is this tag a percentage? (Y/N): ").lower() == 'y',
            'hourly_reset': input("Does this value reset each hour? (Y/N): ").lower() == 'y',
            'shift_reset': input("Does this value reset each shift? (Y/N): ").lower() == 'y',
            'min_value': float(input("What is around the minimum value this tag can have?: ")),
            'max_value': float(input("What is around the maximum value this tag can have?: "))
        }
        tags_to_predict.append(tag_info)
    
    return tags_to_predict

# Function to preprocess data
def preprocess_data():
    conn = sqlite3.connect('plc_tags.db')
    df = pd.read_sql_query("SELECT * FROM tags", conn)
    conn.close()

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['tag_value'] = pd.to_numeric(df['tag_value'], errors='coerce')
    
    # Create features
    df['is_integer'] = df['tag_value'].apply(lambda x: float(x).is_integer() if pd.notnull(x) else False)
    df['is_percentage'] = df['tag_value'].apply(lambda x: 0 <= x <= 1 if pd.notnull(x) else False)
    df['hour_of_day'] = df['timestamp'].dt.hour  
    df['minute_of_hour'] = df['timestamp'].dt.minute
    df['hourly_change'] = df.groupby(['tag_name', 'hour_of_day'])['tag_value'].diff().rolling(window=60).sum()
    df['hourly_reset'] = df.groupby(['tag_name', 'hour_of_day'])['tag_value'].diff().rolling(window=60).apply(lambda x: 1 if np.sum(x) < 0 else 0)
    df['shift_reset'] = df.groupby('tag_name')['tag_value'].diff().rolling(window=480).apply(lambda x: 1 if np.sum(x) < 0 else 0)
    df['value_range'] = df.groupby('tag_name')['tag_value'].transform(lambda x: x.max() - x.min())

    return df

# Function to train model
def train_model(df):
    features = ['is_integer', 'is_percentage', 'hourly_reset', 'shift_reset', 'value_range', 'hourly_change']
    X = df[features]
    y = df['tag_name']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, X_test, y_test

# Function to predict tags 
def predict_tags(model, tags_to_predict):
    # Create a DataFrame with one row for each user-defined tag
    prediction_df = pd.DataFrame([
        {
            'is_integer': tag['is_integer'],
            'is_percentage': tag['is_percentage'],
            'hourly_reset': tag['hourly_reset'],
            'shift_reset': tag['shift_reset'],
            'value_range': tag['max_value'] - tag['min_value'],
            'hourly_change': 0  # We don't have this info from user input, so we use a default value
        }
        for tag in tags_to_predict
    ])
    
    # Get probabilities for each class
    probabilities = model.predict_proba(prediction_df)
    
    # Get top 3 most likely classes for each prediction
    top_3_classes = np.argsort(probabilities, axis=1)[:, -3:]
    
    predictions = []
    for i, tag_info in enumerate(tags_to_predict):
        top_3 = top_3_classes[i]
        predictions.append({
            'label': tag_info['label'],
            'predictions': [
                (model.classes_[class_idx], probabilities[i, class_idx])
                for class_idx in top_3[::-1]
            ]
        })
    
    return predictions

def visualize_feature_importance(model, X_test):
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

def main():
    tags_to_predict = get_user_input()
    
    while True:
        try:
            read_and_log_data()
            df = preprocess_data()
            model, X_test, y_test = train_model(df)
            predictions = predict_tags(model, tags_to_predict)
            
            # Print predictions
            for pred in predictions:
                print(f"\nPredictions for {pred['label']}:")
                for i, (tag, prob) in enumerate(pred['predictions'], 1):
                    print(f"  {i}. {tag} (Probability: {prob:.2f})")
            
            visualize_feature_importance(model, X_test)
            
        except Exception as e:
            print(f"An error occurred in the main loop: {e}")
          
        time.sleep(900)
         
if __name__ == "__main__":
    main()