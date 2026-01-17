
import pandas as pd
import time
import json
from confluent_kafka import Producer
import os  

KAFKA_BROKER = 'kafka:29092'
TOPIC_NAME = 'telematics_stream'
CSV_FILE = 'synthetic_telematics.csv'
TIME_INTERVAL = 0.1  

def delivery_report(err, msg):
    """Called once for each message produced to indicate delivery result."""
    if err is not None:
        
        pass
   
def run_producer():
    
    if not os.path.exists(CSV_FILE):
        print(f"FATAL ERROR: The file '{CSV_FILE}' was not found in the container at /app.")
        print("Please ensure your CSV file is in the same directory as your docker-compose.yml.")
        time.sleep(10)  
        return
    
    
    conf = {'bootstrap.servers': KAFKA_BROKER}
   
    producer = Producer(conf)
    
    
    try:
        df = pd.read_csv(CSV_FILE)
        
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].astype('float32')
        print(f"Dataset loaded. Total records: {len(df)}. Total vehicles: {df['vehicle_id'].nunique()}.")
    except Exception as e:
        print(f"Error loading or processing CSV: {e}")
        return
    
    print(f"Starting continuous telematics stream simulation on topic: {TOPIC_NAME}...")
    
    
    vehicle_groups = df.groupby('vehicle_id')
    
    
    while True:
        for vehicle_id, group in vehicle_groups:
            for index, row in group.iterrows():
                
                message = row.to_dict()
                
                
                key = str(message['vehicle_id']).encode('utf-8')
                try:
                    producer.produce(
                        TOPIC_NAME,
                        key=key,
                        
                        value=json.dumps(message).encode('utf-8'),
                        callback=delivery_report
                    )
                    
                    
                    producer.poll(0)
                except Exception as e:
                    print(f"An error occurred during message production: {e}")
                
                
                time.sleep(TIME_INTERVAL)
        
        print("One full dataset cycle completed for all vehicles. Restarting stream...")

if __name__ == '__main__':
    
    print("Waiting 15 seconds for Kafka broker to be ready...")
    time.sleep(15)
    run_producer()