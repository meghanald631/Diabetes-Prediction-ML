# Diabetes Prediction Using Machine Learning

## Project Overview
This project predicts whether a person is likely to have diabetes based on medical parameters using a Machine Learning model.

The application uses a trained Random Forest Classifier to generate predictions through a simple Flask web interface.

## Technologies Used
- Python
- Flask
- Pandas
- NumPy
- Scikit-learn
- HTML and CSS
- Machine Learning (Random Forest)

## Machine Learning Workflow
1. Data preprocessing and cleaning
2. Handling missing values using an imputer
3. Feature scaling
4. Model training using Random Forest Classifier
5. Saving the trained model and preprocessing objects using Pickle
6. Integrating the model with Flask

## Project Structure
```text
ProjectDiabetes/
│
├── app.py
├── rf_model.pkl
├── imputer.pkl
├── scalar.pkl
├── templates/
│   └── index.html
├── .gitignore
└── README.md
```

## How to Run the Project

1. Clone the repository:
   `git clone https://github.com/meghanald631/Diabetes-Prediction-ML.git`

2. Navigate to the project folder:
   `cd Diabetes-Prediction-ML`

3. Install dependencies:
   `pip install flask pandas numpy scikit-learn`

4. Run the application:
   `python app.py`

5. Open the application in your browser using the local URL displayed in the terminal.

## Model
The project uses a Random Forest Classifier with saved preprocessing components to process input data and generate predictions.

## Disclaimer
This project is developed for educational purposes and is not intended to replace professional medical diagnosis.