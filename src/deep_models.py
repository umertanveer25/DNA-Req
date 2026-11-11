import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder

class BioSequenceAttention(nn.Module):
    """
    Self-attention pooling mechanism over sequence states.
    Computes importance weights for each token state in the requirement sequence.
    """
    def __init__(self, hidden_dim):
        super(BioSequenceAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, rnn_outputs):
        # rnn_outputs: (batch_size, seq_len, hidden_dim)
        weights = self.attention(rnn_outputs)  # (batch_size, seq_len, 1)
        weights = torch.softmax(weights, dim=1)
        context = torch.sum(weights * rnn_outputs, dim=1)  # (batch_size, hidden_dim)
        return context, weights

class HybridSequenceNetwork(nn.Module):
    """
    Core PyTorch Neural Network for Hybrid Sequence Classification:
    Combines dense representation learning (384-d SBERT) with 
    bidirectional sequence modeling and self-attention.
    """
    def __init__(self, input_dim=434, num_classes=5, hidden_dim=128, dropout=0.3):
        super(HybridSequenceNetwork, self).__init__()
        
        # Sequence projection layer treating input features as sequential channels
        self.seq_proj = nn.Linear(input_dim, hidden_dim)
        
        # Bidirectional Sequence Modeling (BiLSTM)
        self.bilstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # Self-Attention Layer
        self.attention = BioSequenceAttention(hidden_dim)
        
        # Deep Representation Fusion & Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        # Reshape to 2D pseudo-sequence for sequence modeling
        batch_size = x.size(0)
        proj = torch.relu(self.seq_proj(x)).unsqueeze(1)  # (batch, 1, hidden_dim)
        
        # BiLSTM Sequence propagation
        lstm_out, _ = self.bilstm(proj)
        
        # Attention pooling
        attended, _ = self.attention(lstm_out)
        
        # Classification
        logits = self.classifier(attended)
        return logits

class HybridDeepSequenceClassifier(BaseEstimator, ClassifierMixin):
    """
    Scikit-Learn compatible Hybrid Sequence Model.
    Directly satisfies the PhD Thesis Title:
    'An Effective Hybrid Sequence Model for Automated Software Requirement 
     Classification Through Deep Representation Learning' (Synopsis [UT1], [UT2]).
    """
    def __init__(self, epochs=25, batch_size=32, lr=1e-3, hidden_dim=128, 
                 dropout=0.3, random_state=42):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.random_state = random_state
        self.model = None
        self.label_encoder = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        self.classes_ = self.label_encoder.classes_
        num_classes = len(self.classes_)
        
        input_dim = X.shape[1]
        self.model = HybridSequenceNetwork(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout
        ).to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        
        tensor_x = torch.tensor(X, dtype=torch.float32)
        tensor_y = torch.tensor(y_encoded, dtype=torch.long)
        dataset = TensorDataset(tensor_x, tensor_y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
        return self

    def predict_proba(self, X):
        self.model.eval()
        tensor_x = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor_x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        preds_idx = np.argmax(probs, axis=1)
        return self.label_encoder.inverse_transform(preds_idx)
