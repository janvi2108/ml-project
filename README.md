🎓 **Student Math Score Predictor**

A Machine Learning web application that predicts a student's **Math Score** based on demographic and academic features such as gender, parental education, lunch type, and reading/writing scores.

The model is trained using multiple regression algorithms and deployed using **Flask** on **Render**.

---

## 🚀 Live Demo

 **Deployed App:**
(https://studentperformace-xo1y.onrender.com/)

---

## 📌 Features

* Predicts Math Score using ML models
* User-friendly web interface (Flask)
* Preprocessing with Scikit-learn pipelines
* Model persistence using Pickle
* Cloud deployment on Render

---

## 🧠 Machine Learning Models Used

* Random Forest Regressor
* Gradient Boosting Regressor
* Linear Regression
* XGBoost
* CatBoost

The best-performing model is selected based on **R² Score**.

---

## 🗂️ Project Structure

```
ml project/
│
├── application.py
├── requirements.txt
├── artifacts/
│   ├── model.pkl
│   └── preprocessor.pkl
│
├── notebook/
│   └── src/
│       ├── components/
│       ├── pipeline/
│       ├── exception.py
│       └── utils.py
│
├── templates/
│   ├── index.html
│   └── home.html
│
└── README.md
```

---

## ⚙️ Installation & Setup (Local)

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2️⃣ Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the app

```bash
python application.py
```

Open in browser:

```
http://127.0.0.1:5000
```

---

## 🌐 Deployment (Render)

**Build Command**

```
pip install -r requirements.txt
```

**Start Command**

```
gunicorn application:application
```

---

## 📊 Input Features

| Feature            | Description      |
| ------------------ | ---------------- |
| Gender             | Male / Female    |
| Race/Ethnicity     | Group A–E        |
| Parental Education | Education level  |
| Lunch              | Standard / Free  |
| Test Prep          | Completed / None |
| Reading Score      | 0–100            |
| Writing Score      | 0–100            |

---

## 🎯 Output

The model predicts:

```
Predicted Math Score
```

---

## 🛠️ Tech Stack

* Python
* Flask
* Scikit-learn
* Pandas
* NumPy
* XGBoost
* CatBoost
* HTML/CSS
* Render (Cloud Hosting)

---

## ⭐ Future Improvements

* Add confidence interval
* Model explainability (SHAP)
* UI improvements
* REST API
* Mobile responsiveness

---



