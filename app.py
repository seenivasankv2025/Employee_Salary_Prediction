from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained ML pipeline
with open("salary_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load dataset
df = pd.read_csv("Salary Data.csv")
df = df.dropna()

# Dropdown values
genders = sorted(df["Gender"].unique())
education_levels = sorted(df["Education Level"].unique())
job_titles = sorted(df["Job Title"].unique())

# Graph data
experience_data = df["Years of Experience"].tolist()
salary_data = df["Salary"].tolist()


@app.route("/")
def home():
    return render_template(
        "index.html",
        genders=genders,
        education_levels=education_levels,
        job_titles=job_titles,
        experience_data=experience_data,
        salary_data=salary_data,
        prediction=None,
        predicted_experience=None
    )


@app.route("/predict", methods=["POST"])
def predict():
    age = float(request.form["age"])
    gender = request.form["gender"]
    education = request.form["education"]
    job_title = request.form["job_title"]
    experience = float(request.form["experience"])

    employee = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "Education Level": [education],
        "Job Title": [job_title],
        "Years of Experience": [experience]
    })

    prediction = model.predict(employee)[0]

    return render_template(
        "index.html",
        genders=genders,
        education_levels=education_levels,
        job_titles=job_titles,
        experience_data=experience_data,
        salary_data=salary_data,

        prediction=round(prediction, 2),
        predicted_experience=experience,

        age=age,
        gender=gender,
        education=education,
        job_title=job_title,
        experience=experience
    )


if __name__ == "__main__":
    app.run(debug=True)