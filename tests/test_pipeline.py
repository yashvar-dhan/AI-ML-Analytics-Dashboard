from app.main import app

def test_home():
    r=app.test_client().get('/')
    assert r.status_code==200

def test_prediction():
    r=app.test_client().post('/api/predict',json={'age':32,'annual_income':65000,'monthly_visits':6,'avg_order_value':80,'days_since_last_visit':10})
    assert r.status_code==200
    assert 0 <= r.json['probability'] <= 1
