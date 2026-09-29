import numpy as np
import pandas as pd
from pathlib import Path

def generate(n=2500, seed=42):
    rng=np.random.default_rng(seed)
    age=rng.normal(34,10,n).clip(18,70)
    income=rng.lognormal(np.log(60000),.45,n).clip(18000,250000)
    visits=rng.poisson(5,n).clip(0,20)
    order=rng.normal(75,25,n).clip(10,200)
    recency=rng.exponential(18,n).clip(1,120)
    score=.35*visits+.012*order-.018*recency+.000004*income+.015*age+rng.normal(0,1.2,n)
    y=(score>np.median(score)).astype(int)
    df=pd.DataFrame({'age':age.round(1),'annual_income':income.round(2),'monthly_visits':visits,'avg_order_value':order.round(2),'days_since_last_visit':recency.round(1),'purchased':y})
    for col in ['annual_income','avg_order_value']:
        idx=rng.choice(n,size=int(n*.03),replace=False); df.loc[idx,col]=np.nan
    return df

if __name__=='__main__':
    out=Path(__file__).resolve().parents[1]/'data'/'customer_purchase.csv'
    out.parent.mkdir(exist_ok=True)
    generate().to_csv(out,index=False)
    print(out)
