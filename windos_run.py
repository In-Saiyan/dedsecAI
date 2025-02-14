import torch
import torch.nn as nn
import numpy as np
import json
import win32evtlog
import win32evtlogutil
import time
from datetime import datetime, timezone
import os
import logging
import hashlib
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("cbs_monitor.log")
    ]
)

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

# Define Anomaly Classifier Model
class AnomalyClassifier(nn.Module):
    def __init__(self, input_dim):
        super(AnomalyClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

# Load trained models
input_dim = 3  # Timestamp, Event ID, Frequency
autoencoder = Autoencoder(input_dim).to(device)
autoencoder.load_state_dict(torch.load("./models/cbs_anomaly_detector.pth", map_location=device))
autoencoder.eval()

classifier = AnomalyClassifier(input_dim).to(device)
classifier.load_state_dict(torch.load("./models/anomaly_classifier.pth", map_location=device))
classifier.eval()

# StandardScaler (Ensure consistency)
scaler = StandardScaler()

# Event Mapping
event_id_map = {}

# Helper function to compute the hash of a message
def compute_hash(message):
    return hashlib.sha256(message.encode("utf-8")).hexdigest()

# Keep track of seen event hashes
SEEN_EVENTS_FILE = "seen_event_hashes.json"
seen_event_hashes = set()
if os.path.exists(SEEN_EVENTS_FILE):
    with open(SEEN_EVENTS_FILE, "r", encoding="utf-8") as f:
        try:
            seen_event_hashes = set(json.load(f))
        except json.JSONDecodeError:
            seen_event_hashes = set()

# Read Windows CBS Event Log
LOG_TYPE = "Microsoft-Windows-CBS/Operational"

def fetch_cbs_events():
    logs = []
    hand = None
    try:
        logging.info("Fetching CBS Event Logs...")
        # Open the event log
        hand = win32evtlog.OpenEventLog(None, LOG_TYPE)
        if not hand:
            raise Exception("Failed to open event log. Handle is None or invalid.")

        flags = win32evtlog.EVENTLOG_BACKWARDS_READ | win32evtlog.EVENTLOG_SEQUENTIAL_READ
        records = win32evtlog.ReadEventLog(hand, flags, 0)

        for event in records:
            timestamp = event.TimeGenerated.timestamp()
            event_id = event.EventID & 0xFFFF
            message = win32evtlogutil.SafeFormatMessage(event, LOG_TYPE)

            if event_id not in event_id_map:
                event_id_map[event_id] = len(event_id_map) + 1

            # Compute hash of the message
            message_hash = compute_hash(message)

            if message_hash in seen_event_hashes:
                continue

            seen_event_hashes.add(message_hash)
            logs.append((timestamp, event_id_map[event_id], message))

        logging.info(f"Fetched {len(logs)} new events.")
    except Exception as e:
        logging.error(f"Error reading event log: {e}")
    finally:
        if hand is not None:
            try:
                # Explicitly check the validity of the handle before closing
                win32evtlog.CloseEventLog(hand)
                logging.info("Event log handle closed successfully.")
            except Exception as close_error:
                logging.error(f"Error closing event log handle: {close_error}")
                # Reset handle to None to avoid re-closing
                hand = None
    return logs



# Detect anomalies
def detect_anomalies():
    events = fetch_cbs_events()
    if not events:
        logging.info("No new events to process.")
        return

    timestamps, event_ids = zip(*[(e[0], e[1]) for e in events])
    frequencies = {eid: event_ids.count(eid) for eid in set(event_ids)}

    data = np.array([[timestamps[i], event_ids[i], frequencies[event_ids[i]]] for i in range(len(events))])
    scaler.fit(data)  # Fit only once
    data = scaler.transform(data)

    data_tensor = torch.tensor(data, dtype=torch.float32).to(device)

    with torch.no_grad():
        reconstructions = autoencoder(data_tensor).cpu().numpy()

    errors = np.mean((data - reconstructions) ** 2, axis=1)
    threshold = np.percentile(errors, 95)
    anomalies = (errors > threshold).astype(int)

    with torch.no_grad():
        classifier_predictions = classifier(data_tensor).cpu().detach().numpy().flatten()
        classifier_predictions = (classifier_predictions > 0.5).astype(int)

    anomaly_logs = []
    for i, event in enumerate(events):
        if anomalies[i] or classifier_predictions[i]:
            anomaly_logs.append({
                "timestamp": datetime.fromtimestamp(event[0], timezone.utc).isoformat(),
                "event_id": event[1],
                "message": event[2]
            })

    logging.info(f"Detected {len(anomaly_logs)} anomalies.")

    if anomaly_logs:
        save_anomalies_to_json(anomaly_logs)

def save_anomalies_to_json(new_anomalies, filename="anomalies.json"):
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        with open(filename, "r", encoding="utf-8") as f:
            try:
                anomalies = json.load(f)
            except json.JSONDecodeError:
                anomalies = []
    else:
        anomalies = []

    anomalies.extend(new_anomalies)

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(anomalies, f, indent=4)

    logging.info(f"Saved {len(new_anomalies)} anomalies to {filename}.")

# Save seen event hashes
def save_seen_event_hashes():
    with open(SEEN_EVENTS_FILE, "w", encoding="utf-8") as f:
        json.dump(list(seen_event_hashes), f)
    logging.info("Seen event hashes saved.")

# Monitor logs in real-time
def monitor_logs():
    logging.info("Starting CBS Event Log monitoring...")
    try:
        while True:
            detect_anomalies()
            save_seen_event_hashes()
            time.sleep(3)
    except KeyboardInterrupt:
        logging.info("Stopping monitoring. Saving seen event hashes.")
        save_seen_event_hashes()

if __name__ == "__main__":
    monitor_logs()
