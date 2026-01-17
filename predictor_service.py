import json
import joblib
import pandas as pd
import redis
from fastapi import FastAPI
from confluent_kafka import Consumer, KafkaError
import logging
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Fleet Anomaly Predictor")


model = joblib.load("telematics_xgb_model.joblib")
features = joblib.load("model_features.joblib")
logger.info("Model loaded successfully – ready for real-time predictions")


r = redis.Redis(host='redis', port=6379, db=0)


conf = {
    'bootstrap.servers': 'kafka:29092',   # ← THIS IS CORRECT
    'group.id': 'predictor-group',
    'auto.offset.reset': 'earliest'
}
consumer = Consumer(conf)
consumer.subscribe(['telematics_stream'])

def predict_and_store():
    while True:
        msg = consumer.poll(1.0)
        if msg is None: continue
        if msg.error(): 
            if msg.error().code() != KafkaError._PARTITION_EOF:
                logger.error(msg.error())
            continue

        data = json.loads(msg.value().decode('utf-8'))
        df = pd.DataFrame([data])[features]

        pred = int(model.predict(df)[0])
        prob = float(model.predict_proba(df)[0][1])

        
        key = f"vehicle:{data['vehicle_id']}"
        data.update({
            "anomaly_score": prob,
            "is_anomaly": bool(pred)
        })
        r.hmset(key, {k: str(v) for k, v in data.items()})
        r.expire(key, 300)

        if pred:
            logger.info(f"ANOMALY → {data['vehicle_id']} | Risk: {prob:.1%} | Speed: {data['speed_kmh']} km/h")

threading.Thread(target=predict_and_store, daemon=True).start()

@app.get("/")
def home():
    return {"message": "Real-time anomaly detection ACTIVE", "vehicles_monitored": 40}