import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import re
import xgboost as xgb
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define Autoencoder model
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

# Parse CBS log lines
def parse_log_line(line):
    match = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?(\bHRESULT = 0x[0-9a-fA-F]+|\bError\b|\bWarning\b)", line)
    if match:
        timestamp = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
        error_code = match.group(2)
        return timestamp, error_code
    return None, None

# Load CBS logs efficiently
def load_cbs_log(file_path, max_lines=5000000000000000000000000):
    timestamps, error_codes = [], []
    error_map, frequencies = {}, {}

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(tqdm(f, desc="Reading logs")):
            if i >= max_lines:
                break
            timestamp, error_code = parse_log_line(line)
            if timestamp and error_code:
                timestamps.append(timestamp.timestamp())
                if error_code not in error_map:
                    error_map[error_code] = len(error_map) + 1
                error_id = error_map[error_code]
                error_codes.append(error_id)
                frequencies[error_id] = frequencies.get(error_id, 0) + 1

    log_frequencies = np.array([frequencies[error_code] for error_code in error_codes])
    return np.array(timestamps), np.array(error_codes), log_frequencies

# Load data
log_file = "cbs.log"
timestamps, error_codes, log_frequencies = load_cbs_log(log_file)

# Normalize and prepare dataset
scaler = StandardScaler()
X = scaler.fit_transform(np.column_stack((timestamps, error_codes, log_frequencies)))

# Train-test split
X_train, X_test = train_test_split(X, test_size=0.1, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

# Use DataLoader for batch processing
dataset = TensorDataset(X_train_tensor)
dataloader = DataLoader(dataset, batch_size=8192, shuffle=True)

# Initialize Autoencoder model
input_dim = X_train.shape[1]
model = Autoencoder(input_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
scaler_amp = torch.cuda.amp.GradScaler()

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output = model(batch[0])
            loss = criterion(output, batch[0])
        scaler_amp.scale(loss).backward()
        scaler_amp.step(optimizer)
        scaler_amp.update()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss / len(dataloader):.6f}")

# Save trained model
torch.save(model.state_dict(), "./models/cbs_anomaly_detector.pth")
print("Training complete! Model saved.")

# XGBoost for Anomaly Detection
model.eval()
with torch.no_grad():
    X_train_reconstructed = model(X_train_tensor).cpu().numpy()
    X_test_reconstructed = model(X_test_tensor).cpu().numpy()

train_errors = np.mean((X_train - X_train_reconstructed) ** 2, axis=1)
test_errors = np.mean((X_test - X_test_reconstructed) ** 2, axis=1)

threshold = np.percentile(train_errors, 95)
y_train = (train_errors > threshold).astype(int)
y_test = (test_errors > threshold).astype(int)

xgb_model = xgb.XGBClassifier(
    n_estimators=100, 
    max_depth=6, 
    learning_rate=0.1, 
    tree_method='gpu_hist' if torch.cuda.is_available() else 'hist',
    use_label_encoder=False, 
    eval_metric='logloss'
)
xgb_model.fit(X_train, y_train)

xgb_model.save_model("./models/xgb_anomaly_detector.json")
print("XGBoost model trained and saved!")