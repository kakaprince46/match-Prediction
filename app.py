import pandas as pd
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained models and feature list
model_full = joblib.load('model_full.pkl')
model_half = joblib.load('model_half.pkl')
features = joblib.load('features.pkl')

teams = [
    'Fulham', 'Crystal Palace', 'Liverpool', 'West Ham', 'West Brom', 'Tottenham',
    'Brighton', 'Sheffield United', 'Everton', 'Leeds', 'Man United', 'Arsenal',
    'Southampton', 'Newcastle', 'Chelsea', 'Leicester', 'Aston Villa', 'Wolves',
    'Burnley', 'Man City', 'Brentford', 'Watford', 'Norwich', 'Bournemouth',
    'Nott\'m Forest', 'Luton', 'Ipswich'
]

@app.route('/')
def index():
    return render_template('index.html', teams=teams)

@app.route('/predict', methods=['POST'])
def predict():
    home_team = request.form['home_team']
    away_team = request.form['away_team']

    # Create a new dataframe with zeros for all teams
    input_data = pd.DataFrame(0, index=[0], columns=features)

    # Set the teams to 1 for the selected match
    input_data[f'HomeTeam_{home_team}'] = 1
    input_data[f'AwayTeam_{away_team}'] = 1

    # Get probability predictions
    prob_full = model_full.predict_proba(input_data)[0]
    prob_half = model_half.predict_proba(input_data)[0]

    # Map probabilities to a dictionary
    prediction_full = {
        'Home Win': round(prob_full[2] * 100, 2), # FTR 'H'
        'Away Win': round(prob_full[0] * 100, 2), # FTR 'A'
        'Draw': round(prob_full[1] * 100, 2)      # FTR 'D'
    }
    prediction_half = {
        'Home Win': round(prob_half[2] * 100, 2), # HTR 'H'
        'Away Win': round(prob_half[0] * 100, 2), # HTR 'A'
        'Draw': round(prob_half[1] * 100, 2)      # HTR 'D'
    }

    return render_template('result.html', 
                           home_team=home_team, 
                           away_team=away_team, 
                           prediction_full=prediction_full, 
                           prediction_half=prediction_half)

if __name__ == '__main__':
    app.run(debug=True)
    