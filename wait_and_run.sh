
echo "Waiting 30 seconds for Kafka to initialize before starting predictor..."
sleep 30 
# Then, start the FastAPI/Uvicorn server
exec uvicorn predictor_service:app --host 0.0.0.0 --port 8000