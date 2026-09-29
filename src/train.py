from pathlib import Path
import json, joblib, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from generate_data import generate

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'/'customer_purchase.csv'; MODEL=ROOT/'model'
MODEL.mkdir(exist_ok=True); DATA.parent.mkdir(exist_ok=True)
if not DATA.exists(): generate().to_csv(DATA,index=False)
df=pd.read_csv(DATA); X=df.drop(columns='purchased'); y=df['purchased']
prep=ColumnTransformer([('num',Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler())]),X.columns.tolist())])
pipe=Pipeline([('preprocess',prep),('classifier',LogisticRegression(max_iter=1000,random_state=42))])
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
pipe.fit(Xtr,ytr); pred=pipe.predict(Xte); prob=pipe.predict_proba(Xte)[:,1]
metrics={'accuracy':round(accuracy_score(yte,pred),4),'precision':round(precision_score(yte,pred),4),'recall':round(recall_score(yte,pred),4),'f1':round(f1_score(yte,pred),4),'roc_auc':round(roc_auc_score(yte,prob),4),'train_samples':len(Xtr),'test_samples':len(Xte),'features':len(X.columns)}
joblib.dump(pipe,MODEL/'customer_purchase_pipeline.joblib'); (MODEL/'metrics.json').write_text(json.dumps(metrics,indent=2)); print(metrics)
