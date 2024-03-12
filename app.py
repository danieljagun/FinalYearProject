from flask import Flask, jsonify, request
from flask_cors import CORS
import subprocess

app = Flask(__name__)
CORS(app, origins="http://localhost:3000")


@app.route('/')
def hello_world():
    return jsonify({'message': 'Hello, Test!'})


@app.route('/predict', methods=['POST'])
def predict():
    coin_name = request.json.get('coin_name', None)
    if coin_name is None:
        return jsonify({'success': False, 'error': 'No coin name provided'})

    try:
        # Run the TrainModelPrediction.py script for the specified coin
        result = subprocess.check_output(['python', 'TrainModelPrediction.py', coin_name]).decode('utf-8')

        # Return the result as JSON
        return jsonify({'success': True, 'output': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
