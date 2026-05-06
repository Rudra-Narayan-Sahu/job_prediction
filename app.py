from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load models
classifier = joblib.load('classifier.pkl')
regressor = joblib.load('regressor.pkl')

# Load label encoders
le_branch = joblib.load('le_branch.pkl')
le_job_role = joblib.load('le_job_role.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    cgpa = float(request.form['cgpa'])
    branch = le_branch.transform([request.form['branch']])[0]
    college_tier = int(request.form['college_tier'])
    python_skill = int(request.form['python_skill'])
    dsa_skill = int(request.form['dsa_skill'])
    ml_skill = int(request.form['ml_skill'])
    web_dev_skill = int(request.form['web_dev_skill'])
    coding_score = float(request.form['coding_score'])
    communication_score = float(request.form['communication_score'])
    aptitude_score = float(request.form['aptitude_score'])
    internships = int(request.form['internships'])
    projects = int(request.form['projects'])
    backlogs = int(request.form['backlogs'])
    resume_score = float(request.form['resume_score'])
    skill_score = int(request.form['skill_score'])

    # Prepare input
    input_data = pd.DataFrame({
        'cgpa': [cgpa],
        'branch': [branch],
        'college_tier': [college_tier],
        'python_skill': [python_skill],
        'dsa_skill': [dsa_skill],
        'ml_skill': [ml_skill],
        'web_dev_skill': [web_dev_skill],
        'coding_score': [coding_score],
        'communication_score': [communication_score],
        'aptitude_score': [aptitude_score],
        'internships': [internships],
        'projects': [projects],
        'backlogs': [backlogs],
        'resume_score': [resume_score],
        'skill_score': [skill_score],
        'company_type': [0]  # Placeholder
    })

    # Predict
    predicted_job_role_encoded = classifier.predict(input_data)[0]
    predicted_salary = regressor.predict(input_data)[0]

    # Decode job_role
    predicted_job_role = le_job_role.inverse_transform([predicted_job_role_encoded])[0]

    return render_template('index.html', prediction=f'Predicted Job Role: {predicted_job_role}, Predicted Salary: {predicted_salary:.2f} LPA')

if __name__ == '__main__':
    app.run(debug=True)