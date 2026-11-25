from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__, template_folder='templates')

# Load saved models

with open('rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

with open('scalar.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('imputer.pkl', 'rb') as f:
    fill_zeros = pickle.load(f)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        field_names = [
            'num_preg', 'glucose_conc', 'diastolic_bp',
            'thickness', 'insulin', 'bmi', 'diab_pred', 'age'
        ]

        features = [float(request.form[field]) for field in field_names]

        input_array = np.array(features).reshape(1, -1)
        input_array = fill_zeros.transform(input_array)
        input_array = scaler.transform(input_array)

        prediction = int(rf_model.predict(input_array)[0])

        return render_template(
            'index.html',
            prediction=prediction
        )

    except Exception as e:
        return f"Prediction error: {str(e)}"

        # Get values from the HTML form
        form_data = {
            field: request.form[field]
            for field in field_names
        }

        # Convert values to numbers
        features = [
            float(form_data[field])
            for field in field_names
        ]

        # Convert into a 2D array
        input_array = np.array(features).reshape(1, -1)

        # Preprocess input
        input_array = fill_zeros.transform(input_array)
        input_array = scaler.transform(input_array)

        # Predict
        prediction = rf_model.predict(input_array)

        result = (
            "Diabetic" if prediction[0] == 1
            else "Not Diabetic"
        )

        return render_template(
            'index.html',
            prediction_text=f'Prediction: {result}',
            form_data=form_data
        )

    except Exception as e:
        return render_template(
            'index.html',
            prediction_text=f'Error: {str(e)}'
        )


if __name__ == '__main__':
    app.run(debug=True)
