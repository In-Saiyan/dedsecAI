import torch
import torch.nn as nn
import numpy as np
import json
import xgboost as xgb
import win32evtlog
import win32evtlogutil
import win32con
import time
import os
from datetime import datetime, timezone
from sklearn.preprocessing import StandardScaler
from collections import deque

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Autoencoder Model
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Load trained Autoencoder model
input_dim = 3  # Timestamp, Event ID, Frequency
model = Autoencoder(input_dim).to(device)
model.load_state_dict(torch.load("cbs_anomaly_detector.pth", map_location=device))
model.eval()

# Load trained XGBoost model
xgb_model = xgb.XGBClassifier()
xgb_model.load_model("xgb_anomaly_detector.json")

# StandardScaler (Ensure consistency)
scaler = StandardScaler()

# Event Mapping
event_id_map = {}
event_queue = deque(maxlen=1000)  # Store recent events

# Read Windows Event Log
LOG_TYPE = "System"  # Change to "Application" or another log source as needed

def fetch_windows_events():
    logs = []
    try:
        hand = win32evtlog.OpenEventLog(None, LOG_TYPE)
        flags = win32evtlog.EVENTLOG_BACKWARDS_READ | win32evtlog.EVENTLOG_SEQUENTIAL_READ
        total = win32evtlog.GetNumberOfEventLogRecords(hand)
        records = win32evtlog.ReadEventLog(hand, flags, 0)
        for event in records:
            timestamp = event.TimeGenerated.timestamp()
            event_id = event.EventID & 0xFFFF  # Extract low 16 bits
            message = win32evtlogutil.SafeFormatMessage(event, LOG_TYPE)
            
            if event_id not in event_id_map:
                event_id_map[event_id] = len(event_id_map) + 1
            
            logs.append((timestamp, event_id_map[event_id], message))
        win32evtlog.CloseEventLog(hand)
    except Exception as e:
        print(f"Error reading event log: {e}")
    return logs

# Process logs and detect anomalies
def detect_anomalies():
    events = fetch_windows_events()
    if not events:
        return

    timestamps, event_ids = zip(*[(e[0], e[1]) for e in events])

    # Compute frequencies of event IDs
    frequencies = {eid: event_ids.count(eid) for eid in set(event_ids)}
    
    # Prepare Data
    data = np.array([[timestamps[i], event_ids[i], frequencies[event_ids[i]]] for i in range(len(events))])

    # Normalize Data
    if len(event_queue) == 0:
        scaler.fit(data)  # Fit only once
    data = scaler.transform(data)

    data_tensor = torch.tensor(data, dtype=torch.float32).to(device)

    with torch.no_grad():
        reconstructions = model(data_tensor).cpu().numpy()

    # Compute Reconstruction Errors
    errors = np.mean((data - reconstructions) ** 2, axis=1)
    threshold = np.percentile(errors, 95)
    anomalies = (errors > threshold).astype(int)

    # Use XGBoost to further classify anomalies
    anomaly_predictions = xgb_model.predict(data)

    anomaly_logs = []
    for i, event in enumerate(events):
        if anomalies[i] or anomaly_predictions[i]:
            anomaly_logs.append({
                "timestamp": datetime.fromtimestamp(event[0], datetime.timezone.utc).isoformat(),
                "event_id": event[1],
                "message": event[2]
            })

    # Save anomalies to JSON (Append instead of Overwrite)
    if anomaly_logs:
        save_anomalies_to_json(anomaly_logs)

def save_anomalies_to_json(new_anomalies, filename="anomalies.json"):
    # Load existing anomalies if file exists
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        with open(filename, "r", encoding="utf-8") as f:
            try:
                anomalies = json.load(f)
            except json.JSONDecodeError:
                anomalies = []
    else:
        anomalies = []

    # Append new anomalies
    anomalies.extend(new_anomalies)

    # Write back to file
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(anomalies, f, indent=4)

    print(f"{len(new_anomalies)} anomalies detected and saved.")

# Monitor logs in real-time
def monitor_logs():
    print("Monitoring Windows Event Logs for anomalies...")
    while True:
        detect_anomalies()
        time.sleep(5)  # Wait for 5 seconds before checking again

if __name__ == "__main__":
    monitor_logs()
