import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# The list of all unique teams you provided
teams = [
    'Fulham', 'Crystal Palace', 'Liverpool', 'West Ham', 'West Brom', 'Tottenham',
    'Brighton', 'Sheffield United', 'Everton', 'Leeds', 'Man United', 'Arsenal',
    'Southampton', 'Newcastle', 'Chelsea', 'Leicester', 'Aston Villa', 'Wolves',
    'Burnley', 'Man City', 'Brentford', 'Watford', 'Norwich', 'Bournemouth',
    'Nott\'m Forest', 'Luton', 'Ipswich'
]

def create_features(df):
    """
    This function engineers new features based on the provided data.
    """
    # Calculate rolling averages for goals scored and conceded for each team
    # 'shift(1)' is used to ensure we're using past data, not data from the current match
    df['HomeTeam_Goals_Scored_Avg'] = df.groupby('HomeTeam')['FTHG'].transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
    df['HomeTeam_Goals_Conceded_Avg'] = df.groupby('HomeTeam')['FTAG'].transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
    
    df['AwayTeam_Goals_Scored_Avg'] = df.groupby('AwayTeam')['FTAG'].transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
    df['AwayTeam_Goals_Conceded_Avg'] = df.groupby('AwayTeam')['FTHG'].transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
    
    return df

try:
    # Step 1: Load the data from your CSV file
    df = pd.read_csv('football_data.csv')

    # Step 2: Clean the data by dropping unnecessary columns and rows with missing data
    df = df.copy() 
    df = df.dropna(subset=['FTR', 'HTR', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HTHG', 'HTAG'])

    # Step 3: Engineer new features (form, etc.)
    df = create_features(df)
    df = df.dropna()
    
    # Step 4: Convert team names into numerical data
    df = pd.get_dummies(df, columns=['HomeTeam', 'AwayTeam'])

    # Step 5: Split the data into features (X) and targets (y)
    home_team_cols = [f'HomeTeam_{team}' for team in teams]
    away_team_cols = [f'AwayTeam_{team}' for team in teams]
    
    features_to_use = [
        'HomeTeam_Goals_Scored_Avg', 'HomeTeam_Goals_Conceded_Avg',
        'AwayTeam_Goals_Scored_Avg', 'AwayTeam_Goals_Conceded_Avg'
    ] + home_team_cols + away_team_cols
    
    X_cols = [col for col in features_to_use if col in df.columns]
    X = df[X_cols]
    
    y_full = df['FTR']
    y_half = df['HTR']

    # Step 6: Train the Machine Learning Models
    model_full = RandomForestClassifier(n_estimators=100, random_state=42)
    model_full.fit(X, y_full)
    
    model_half = RandomForestClassifier(n_estimators=100, random_state=42)
    model_half.fit(X, y_half)

    # Step 7: Save the trained models and the list of features
    joblib.dump(model_full, 'model_full.pkl')
    joblib.dump(model_half, 'model_half.pkl')
    joblib.dump(X_cols, 'features.pkl')
    
    print("Models and feature list saved successfully! You are now ready for the next step.")

except FileNotFoundError:
    print("Error: 'football_data.csv' not found. Please make sure the file is in the same folder as this script.")
except Exception as e:
    print(f"An error occurred: {e}")
    