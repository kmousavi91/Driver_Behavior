# 🚗 Driver Behavior Classification API

This project uses sensor data to classify driver behavior as one of several categories like `Normal`, `Aggressive`, or `Risky`. It consists of a machine learning model trained on processed driving telemetry and served via a FastAPI endpoint.

---

## 🧠 Model Details

* **Type**: Random Forest Classifier
* **Features Used**: 16 engineered windowed features
* **Training Data**: `windowed_features.csv`
* **Labels**:

  * `0`: Normal
  * `1`: Aggressive
  * `2`: Risky
  * `3`: Drowsy
  * `4`: Dangerous

---

## 📦 Project Structure

```
.
├── app.py                  # FastAPI backend for prediction
├── ingest.py              # Cleans and prepares sensor data
├── model_training.py      # Trains and saves the model
├── driver_behavior_model.pkl  # Trained model file
├── data/
        windowed_features.csv
        cleaned_sensor_data.csv
        sensor_raw.csv
├── logs/                  # Logs of predictions
├── plots/
├── requirements.txt
└── README.md
```

---

## 🚀 Run Locally

### 1. Install requirements

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python model_training.py
```

### 3. Start the API

```bash
uvicorn app:app --host 0.0.0.0 --port 8080
```

### 4. Test the API

Open in browser: [http://localhost:8080/docs](http://localhost:8080/docs)

Send a POST request to `/predict` with this example:

```json
{
  "features": [
    0.05, -0.02, 0.08, 0.002, -0.001, 0.005, 0.3, -0.2, 0.1,
    0.8, -0.5, 0.7, 5.1, 4.9, 5.3, -0.8
  ]
}
```

---

## 📟 Prediction Response

```json
{
  "predicted_class": 4,
  "label": "Dangerous"
}
```

---

## 📊 Logs

* All prediction logs are saved in `logs/predictions_YYYY-MM-DD.jsonl`
* Contains timestamp, input features, and predicted class.

---

## 📌 Notes

* Model must receive **exactly 16 float values** in the same order as training.
* All logging and behavior classification happen locally.
* Designed to run on EC2, but works fully on local machines too.

---

## 🔒 Optional

You can integrate with:

* **AWS S3** for log storage
* **CloudWatch** for live logs
* **Streamlit dashboard** for visualization
