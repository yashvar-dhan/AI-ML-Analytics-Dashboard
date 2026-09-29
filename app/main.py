from pathlib import Path
import json, subprocess, sys, joblib, pandas as pd
from flask import Flask, jsonify, render_template, request

ROOT=Path(__file__).resolve().parents[1]; MODEL_DIR=ROOT/'model'
MODEL_FILE=MODEL_DIR/'customer_purchase_pipeline.joblib'; METRIC_FILE=MODEL_DIR/'metrics.json'
if not MODEL_FILE.exists() or not METRIC_FILE.exists():
    subprocess.run([sys.executable,str(ROOT/'src'/'train.py')],check=True,cwd=str(ROOT/'src'))
MODEL=joblib.load(MODEL_FILE); METRICS=json.loads(METRIC_FILE.read_text())
app=Flask(__name__); FEATURES=['age','annual_income','monthly_visits','avg_order_value','days_since_last_visit']
@app.get('/')
def index(): return render_template('index.html',metrics=METRICS)
@app.get('/api/metrics')
def metrics(): return jsonify(METRICS)
@app.post('/api/predict')
def predict():
    p=request.get_json(force=True)
    try: row={k:float(p[k]) for k in FEATURES}
    except (KeyError,TypeError,ValueError): return jsonify({'error':'Provide all five numeric features.'}),400
    threshold=float(p.get('threshold',.5)); probability=float(MODEL.predict_proba(pd.DataFrame([row]))[0,1])
    return jsonify({'probability':round(probability,4),'prediction':int(probability>=threshold),'threshold':threshold})
if __name__=='__main__': app.run(host='0.0.0.0',port=5000)
