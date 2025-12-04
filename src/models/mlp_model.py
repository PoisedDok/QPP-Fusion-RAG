"""
MLP (Neural Network) fusion weight model.
"""

import os
import numpy as np
from typing import Dict, List, Optional

# Fix for M4 OpenMP threading issue
os.environ.setdefault('OMP_NUM_THREADS', '1')

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .base import BaseFusionModel


class FusionMLP(BaseFusionModel):
    """
    Multi-layer Perceptron for predicting fusion weights.
    
    Architecture:
        Input (65) -> Hidden (128) -> ReLU -> Dropout
        -> Hidden (64) -> ReLU -> Output (5) -> Softmax
    
    The softmax ensures weights sum to 1.
    """
    
    def __init__(
        self,
        retrievers: List[str],
        n_qpp: int = 13,
        hidden_sizes: List[int] = None,
        dropout: float = 0.2,
        device: str = None
    ):
        super().__init__(retrievers, n_qpp)
        
        if not HAS_TORCH:
            raise ImportError("PyTorch not installed. Run: pip install torch")
        
        self.hidden_sizes = hidden_sizes or [128, 64]
        self.dropout = dropout
        self.device = device or ('mps' if torch.backends.mps.is_available() 
                                  else 'cuda' if torch.cuda.is_available() 
                                  else 'cpu')
        
        self.model = self._build_model()
        self.model.to(self.device)
    
    def _build_model(self) -> nn.Module:
        """Build the MLP architecture."""
        layers = []
        in_features = self.n_features
        
        for hidden_size in self.hidden_sizes:
            layers.extend([
                nn.Linear(in_features, hidden_size),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            ])
            in_features = hidden_size
        
        # Output layer (no activation, softmax applied in forward)
        layers.append(nn.Linear(in_features, self.n_retrievers))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with softmax normalization."""
        logits = self.model(x)
        return torch.softmax(logits, dim=1)
    
    def train(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        Y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        patience: int = 10
    ) -> Dict:
        """Train the MLP model."""
        print(f"\n=== Training FusionMLP on {self.device} ===")
        
        # Normalize Y to sum to 1
        Y_train_sum = Y_train.sum(axis=1, keepdims=True)
        Y_train_sum[Y_train_sum == 0] = 1
        Y_train = Y_train / Y_train_sum
        
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        Y_train_t = torch.FloatTensor(Y_train).to(self.device)
        
        train_dataset = TensorDataset(X_train_t, Y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Validation data
        if X_val is not None and Y_val is not None:
            Y_val_sum = Y_val.sum(axis=1, keepdims=True)
            Y_val_sum[Y_val_sum == 0] = 1
            Y_val = Y_val / Y_val_sum
            
            X_val_t = torch.FloatTensor(X_val).to(self.device)
            Y_val_t = torch.FloatTensor(Y_val).to(self.device)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        history = {'train_loss': [], 'val_loss': []}
        
        self.model.train()
        
        for epoch in range(epochs):
            # Training
            train_loss = 0.0
            for X_batch, Y_batch in train_loader:
                optimizer.zero_grad()
                Y_pred = self.forward(X_batch)
                loss = criterion(Y_pred, Y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)
            
            # Validation
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    Y_val_pred = self.forward(X_val_t)
                    val_loss = criterion(Y_val_pred, Y_val_t).item()
                self.model.train()
                
                history['val_loss'].append(val_loss)
                scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
                
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            else:
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}")
        
        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
        
        self.model.eval()
        self.is_trained = True
        
        print(f"Training complete. Best val_loss: {best_val_loss:.4f}")
        
        return {
            'final_train_loss': history['train_loss'][-1],
            'best_val_loss': best_val_loss,
            'epochs_trained': len(history['train_loss']),
            'history': history
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict weights."""
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        
        # Ensure model is on correct device
        self.model.to(self.device)
        self.model.eval()
        
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            weights = self.forward(X_t).cpu().numpy()
        
        return weights
    
    def save(self, path: str):
        """Save model including PyTorch state."""
        import pickle
        from pathlib import Path
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Move model to CPU for saving
        self.model.cpu()
        
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self,  # Save full object for easy loading
                'model_state': self.model.state_dict(),
                'retrievers': self.retrievers,
                'n_qpp': self.n_qpp,
                'hidden_sizes': self.hidden_sizes,
                'dropout': self.dropout,
                'model_type': 'FusionMLP',
                'is_trained': self.is_trained
            }, f)
        
        # Move back to device
        self.model.to(self.device)
        print(f"Saved FusionMLP to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'FusionMLP':
        """Load model from file."""
        import pickle
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        model = cls(
            retrievers=data['retrievers'],
            n_qpp=data['n_qpp'],
            hidden_sizes=data['hidden_sizes'],
            dropout=data['dropout']
        )
        
        model.model.load_state_dict(data['model_state'])
        model.model.to(model.device)
        model.model.eval()
        model.is_trained = data['is_trained']
        
        return model

