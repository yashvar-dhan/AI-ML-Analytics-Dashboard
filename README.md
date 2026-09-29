# AI/ML Analytics Dashboard

An end-to-end machine-learning application by Yashvardhan Chaudhary demonstrating data generation, preprocessing, model training, evaluation, model persistence, REST inference and a responsive web dashboard.

## Pipeline
CSV data → Pandas → median imputation → StandardScaler → Logistic Regression → evaluation → Flask API → web UI

## Technical stack
Python, Pandas, NumPy, Scikit-learn, Flask, Joblib, HTML, CSS, JavaScript, Gunicorn.

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/train.py
python -m app.main
```
Open http://localhost:5000.

## API
POST /api/predict with JSON fields: age, annual_income, monthly_visits, avg_order_value, days_since_last_visit, threshold.

## Deployment
Use a Python host such as Render or Railway with start command: `gunicorn app.main:app`.

## Author
Yashvardhan Chaudhary — B.Tech CSE (AI/ML)
GitHub: https://github.com/yashvar-dhan
