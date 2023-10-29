
# Load packages
from flask import Flask, request, jsonify
from catboost import CatBoostClassifier

###############################################################################

app = Flask(__name)

# Dictionary to hold pre-trained Classifiers
models = {}

# Load pre-trained Classifiers for each team
teams = ['ARI','ATL','BAL','BUF','CAR','CHI','CIN','CLE','DAL','DEN',
         'DET','GB','HOU','IND','JAX','KC','LA','LAC','LV','MIA','MIN',
         'NE','NO', 'NYG','NYJ','PHI','PIT','SEA','SF','TB','TEN','WAS']

for team in teams:
    model = CatBoostClassifier()
    model.load_model('{team}_classifier.cbm')
    models[team] = model

@app.route('/predict', methods=['Post'])
def predict():
    data = request.get_json()
    team = data.get('posteam', 'SF')
    model = models.get(team, None)
    
    if model is None:
        return jsonify({'error': f'Model for team {team} not found.'})
    
    # Extract data for prediction
    input_data = data.get('input_data', {})
    
    # Process the input data and make predictions
    playcall_labels = model.classes_
    playcall_probs = model.predict_proba(input_data)
    
    return jsonify({'predicted_plays': playcall_labels.tolist(), 'predicted_probs': playcall_probs.tolist()})

if __name__ == '__main':
    app.run(debug=True)                 # For local testing
    #app.run(host='0.0.0.0', port=7400) # For running online