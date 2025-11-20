from flask import Flask, render_template, request, send_file, redirect, url_for, session
from flask_mysqldb import MySQL
from flask_mail import Mail, Message
import MySQLdb.cursors
import joblib
import numpy as np
import pandas as pd
import json
from fpdf import FPDF
from datetime import datetime
import os
import random
import string

# -------------------- App Setup --------------------
app = Flask(__name__)
app.secret_key = 'your_secret_key'

# -------------------- MySQL Configuration --------------------
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'chronic_db'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'
mysql = MySQL(app)

# -------------------- Mail Configuration --------------------
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'yuvashree28405@gmail.com'
app.config['MAIL_PASSWORD'] = 'vkadbxswangajgep'
app.config['MAIL_DEFAULT_SENDER'] = ('Chronic Disease Prediction', 'yuvashree28405@gmail.com')
mail = Mail(app)

# -------------------- Home --------------------
@app.route('/')
def home():
    return render_template('home.html')

# -------------------- Login --------------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    msg = ''
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM users WHERE username = %s AND password = %s', (username, password))
        account = cursor.fetchone()

        if account:
            session['loggedin'] = True
            session['id'] = account['id']
            session['username'] = account['username']
            return redirect(url_for('predict'))
        else:
            msg = 'Incorrect username or password!'
    return render_template('login.html', msg=msg)

# -------------------- Register --------------------
@app.route('/register', methods=['GET', 'POST'])
def register():
    msg = ''
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']

        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM users WHERE username = %s', (username,))
        account = cursor.fetchone()

        if account:
            msg = 'Account already exists!'
        else:
            cursor.execute('INSERT INTO users (username, password, email) VALUES (%s, %s, %s)',
                           (username, password, email))
            mysql.connection.commit()
            msg = 'You have successfully registered!'
    return render_template('register.html', msg=msg)

# -------------------- Forgot Password --------------------
@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    msg = ''
    if request.method == 'POST':
        email = request.form['email']

        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM users WHERE email = %s', (email,))
        account = cursor.fetchone()

        if account:
            token = ''.join(random.choices(string.ascii_letters + string.digits, k=20))
            reset_link = f"http://127.0.0.1:5000/reset_password/{token}"
            try:
                message = Message('Password Reset Request', recipients=[email])
                message.body = f"Hello {account['username']},\n\nClick the link below to reset your password:\n{reset_link}\n\nIf you didn’t request this, ignore this message."
                mail.send(message)
                msg = "✅ Password reset link has been sent to your registered email."
            except Exception as e:
                msg = f"⚠️ Error sending email: {str(e)}"
        else:
            msg = "❌ No account found with this email address."

        cursor.close()
    return render_template('forgot_password.html', msg=msg)

# -------------------- Prediction --------------------
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # Always clear old data when user opens prediction form
    if request.method == 'GET':
        session.pop('result', None)
        session.pop('user_inputs', None)
        return render_template('predict.html')

    if request.method == 'POST':
        try:
            # ---- Collect User Input ----
            age = float(request.form['age'])
            gender = 1 if request.form['gender'] == 'Male' else 0
            bmi = float(request.form['bmi'])
            bp = float(request.form['bp'])
            cholesterol = float(request.form['cholesterol'])
            glucose = float(request.form['glucose'])
            physical_activity = float(request.form['physical_activity'])
            alcohol = float(request.form['alcohol'])
            smoking_map = {'Never': 0, 'Former': 1, 'Current': 2}
            smoking = smoking_map.get(request.form['smoking'], 0)
            family_history = 1 if request.form['family_history'] == 'Yes' else 0

            user_inputs = {
                "Age": age,
                "Gender": request.form['gender'],
                "BMI": bmi,
                "Systolic BP": bp,
                "Cholesterol": cholesterol,
                "Glucose": glucose,
                "Physical Activity": physical_activity,
                "Smoking Status": request.form['smoking'],
                "Alcohol Intake": alcohol,
                "Family History": request.form['family_history']
            }

            # ---- Prepare DataFrame ----
            features = pd.DataFrame({
                'age': [age],
                'gender': [gender],
                'bmi': [bmi],
                'blood_pressure': [bp],
                'cholesterol_level': [cholesterol],
                'glucose_level': [glucose],
                'physical_activity': [physical_activity],
                'smoking_status': [smoking],
                'alcohol_intake': [alcohol],
                'family_history': [family_history]
            })

            # ---- Load Models & Predict ----
            model_paths = {
                "Hypertension": "models/hypertension_model.joblib",
                "Diabetes": "models/diabetes_model.joblib",
                "Heart Disease": "models/heart_model.joblib"
            }

            predictions = {}
            for disease, path in model_paths.items():
                if os.path.exists(path):
                    model = joblib.load(path)
                    if hasattr(model, 'feature_names_in_'):
                        missing = [c for c in model.feature_names_in_ if c not in features.columns]
                        for c in missing:
                            features[c] = 0
                        features = features[model.feature_names_in_]
                    prob = model.predict_proba(features)[0][1] * 100 if hasattr(model, 'predict_proba') else model.predict(features)[0] * 100
                    predictions[disease] = round(float(prob), 2)
                else:
                    predictions[disease] = "Model missing"

            formatted_result = {k: f"{v}%" for k, v in predictions.items()}

            # ---- Save temporary report ----
            with open("temp_report.json", "w") as f:
                json.dump({"inputs": user_inputs, "predictions": predictions}, f)

            # ✅ Store fresh result in session (clears old data automatically)
            session['result'] = formatted_result
            session['user_inputs'] = user_inputs

            return redirect(url_for('show_result'))

        except Exception as e:
            return render_template('predict.html', error=f"Error: {str(e)}")

# -------------------- Show Result --------------------
@app.route('/show_result')
def show_result():
    result = session.get('result')
    user_inputs = session.get('user_inputs')

    # If no result yet, just show empty form
    if not result or not user_inputs:
        return redirect(url_for('predict'))

    return render_template('predict.html', result=result, user_inputs=user_inputs, show_download=True)

# -------------------- Download Report --------------------
@app.route('/download_report')
def download_report():
    try:
        with open("temp_report.json", "r") as f:
            data = json.load(f)

        user_inputs = data["inputs"]
        predictions = data["predictions"]

        os.makedirs("reports", exist_ok=True)
        filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        filepath = os.path.join("reports", filename)

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(200, 10, "Chronic Disease Prediction Report", ln=True, align="C")
        pdf.ln(10)

        pdf.set_font("Arial", "B", 14)
        pdf.cell(200, 10, "User Entered Details:", ln=True)
        pdf.set_font("Arial", "", 12)
        for key, value in user_inputs.items():
            pdf.cell(200, 8, f"{key}: {value}", ln=True)

        pdf.ln(8)
        pdf.set_font("Arial", "B", 14)
        pdf.cell(200, 10, "Predicted Results:", ln=True)
        pdf.set_font("Arial", "", 12)
        for disease, value in predictions.items():
            pdf.cell(200, 8, f"{disease}: {value}%", ln=True)

        pdf.output(filepath)
        return send_file(filepath, as_attachment=True)

    except Exception as e:
        return f"Error generating report: {str(e)}"

# -------------------- Run App --------------------
if __name__ == '__main__':
    app.run(debug=True)
