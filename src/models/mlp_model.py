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
        Input (n_retrievers * n_qpp_used) -> Hidden -> ReLU -> Dropout -> Output (n_retrievers)
    
    Uses CrossEntropyLoss with target distribution (proper for probability outputs).
    Post-prediction normalization like LightGBM for consistency.
    
    Args:
        qpp_indices: Which QPP methods to use. Default [5] = RSD only.
                    Options: 0=SMV, 1=Sigma_max, 2=Sigma(%), 3=NQC, 4=UEF, 
                            5=RSD, 6=QPP-PRP, 7=WIG, 8=SCNQC, 9=QV-NQC,
                            10=DM, 11=NQA-QPP, 12=BERTQPP
    """
    
    # QPP method index mapping
    QPP_NAMES = {
        0: "SMV", 1: "Sigma_max", 2: "Sigma(%)", 3: "NQC", 4: "UEF", 5: "RSD",
        6: "QPP-PRP", 7: "WIG", 8: "SCNQC", 9: "QV-NQC", 10: "DM", 
        11: "NQA-QPP", 12: "BERTQPP"
    }
    
    def __init__(
        self,
        retrievers: List[str],
        n_qpp: int = 13,
        qpp_indices: List[int] = None,
        hidden_sizes: List[int] = None,
        dropout: float = 0.2,
        device: str = None
    ):
        # Default to RSD only (index 5) - best single predictor
        self.qpp_indices = qpp_indices if qpp_indices is not None else [5]
        self.n_qpp_used = len(self.qpp_indices)
        
        super().__init__(retrievers, n_qpp)
        
        # Override n_features to use only selected QPP methods
        self.n_features = self.n_qpp_used * len(retrievers)
        self.feature_names = [f"{r}_{self.QPP_NAMES.get(i, i)}" 
                              for r in retrievers for i in self.qpp_indices]
        
        if not HAS_TORCH:
            raise ImportError("PyTorch not installed. Run: pip install torch")
        
        # Scale hidden sizes based on input size
        if hidden_sizes is None:
            if self.n_qpp_used == 1:
                hidden_sizes = [32, 16]  # Smaller network for single QPP
            else:
                hidden_sizes = [128, 64]
        
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.device = device or ('mps' if torch.backends.mps.is_available() 
                                  else 'cuda' if torch.cuda.is_available() 
                                  else 'cpu')
        
        self.model = self._build_model()
        self.model.to(self.device)
        
        qpp_names = [self.QPP_NAMES.get(i, str(i)) for i in self.qpp_indices]
        print(f"[FusionMLP] Using QPP methods: {qpp_names} ({self.n_features} features)")
    
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
    
    def _filter_features(self, X: np.ndarray) -> np.ndarray:
        """
        Filter input features to only use selected QPP indices.
        
        Input X has shape (n_samples, n_retrievers * 13)
        Output has shape (n_samples, n_retrievers * n_qpp_used)
        """
        n_samples = X.shape[0]
        X_filtered = np.zeros((n_samples, self.n_features))
        
        for j in range(self.n_retrievers):
            for k, qpp_idx in enumerate(self.qpp_indices):
                # Source: retriever j, QPP index qpp_idx (out of 13)
                src_idx = j * self.n_qpp + qpp_idx
                # Dest: retriever j, QPP position k (out of n_qpp_used)
                dst_idx = j * self.n_qpp_used + k
                X_filtered[:, dst_idx] = X[:, src_idx]
        
        return X_filtered
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass - raw logits (no softmax, normalized at predict time)."""
        return self.model(x)
    
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
        """Train the MLP model using CrossEntropyLoss."""
        qpp_names = [self.QPP_NAMES.get(i, str(i)) for i in self.qpp_indices]
        print(f"\n=== Training FusionMLP on {self.device} (CrossEntropyLoss) ===")
        print(f"    QPP methods: {qpp_names}")
        print(f"    Input features: {self.n_features} ({self.n_retrievers} retrievers × {self.n_qpp_used} QPP)")
        
        # Filter features to only use selected QPP indices
        X_train_filtered = self._filter_features(X_train)
        
        # Normalize Y to sum to 1 (target distribution)
        Y_train_sum = Y_train.sum(axis=1, keepdims=True)
        Y_train_sum[Y_train_sum == 0] = 1
        Y_train_norm = Y_train / Y_train_sum
        
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train_filtered).to(self.device)
        Y_train_t = torch.FloatTensor(Y_train_norm).to(self.device)
        
        train_dataset = TensorDataset(X_train_t, Y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Validation data
        if X_val is not None and Y_val is not None:
            X_val_filtered = self._filter_features(X_val)
            
            Y_val_sum = Y_val.sum(axis=1, keepdims=True)
            Y_val_sum[Y_val_sum == 0] = 1
            Y_val_norm = Y_val / Y_val_sum
            
            X_val_t = torch.FloatTensor(X_val_filtered).to(self.device)
            Y_val_t = torch.FloatTensor(Y_val_norm).to(self.device)
        
        # CrossEntropyLoss: expects raw logits, applies log_softmax internally
        # We use soft labels (probability distribution), so we use KLDivLoss equivalent
        # CrossEntropy with soft labels: -sum(y * log_softmax(pred))
        def soft_cross_entropy(pred_logits, target_probs):
            log_probs = torch.log_softmax(pred_logits, dim=1)
            return -(target_probs * log_probs).sum(dim=1).mean()
        
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
                logits = self.forward(X_batch)
                loss = soft_cross_entropy(logits, Y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)
            
            # Validation
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_logits = self.forward(X_val_t)
                    val_loss = soft_cross_entropy(val_logits, Y_val_t).item()
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
        """Predict weights (softmax at inference, then normalize like LightGBM)."""
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        
        # Filter features to only use selected QPP indices
        X_filtered = self._filter_features(X)
        
        # Ensure model is on correct device
        self.model.to(self.device)
        self.model.eval()
        
        with torch.no_grad():
            X_t = torch.FloatTensor(X_filtered).to(self.device)
            logits = self.forward(X_t)
            # Apply softmax at inference to get probabilities
            weights = torch.softmax(logits, dim=1).cpu().numpy()
        
        # Additional normalization for consistency (softmax already sums to 1, but clip negatives)
        return self._normalize_weights(weights)
    
    def save(self, path: str):
        """Save model including PyTorch state."""
        import pickle
        from pathlib import Path
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Move model to CPU for saving
        self.model.cpu()
        
        qpp_names = [self.QPP_NAMES.get(i, str(i)) for i in self.qpp_indices]
        
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self,  # Save full object for easy loading
                'model_state': self.model.state_dict(),
                'retrievers': self.retrievers,
                'n_qpp': self.n_qpp,
                'qpp_indices': self.qpp_indices,
                'hidden_sizes': self.hidden_sizes,
                'dropout': self.dropout,
                'model_type': 'FusionMLP',
                'is_trained': self.is_trained
            }, f)
        
        # Move back to device
        self.model.to(self.device)
        print(f"Saved FusionMLP to {path} (QPP: {qpp_names})")
    
    @classmethod
    def load(cls, path: str) -> 'FusionMLP':
        """Load model from file."""
        import pickle
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        # Handle legacy models without qpp_indices
        qpp_indices = data.get('qpp_indices', list(range(13)))
        
        model = cls(
            retrievers=data['retrievers'],
            n_qpp=data['n_qpp'],
            qpp_indices=qpp_indices,
            hidden_sizes=data['hidden_sizes'],
            dropout=data['dropout']
        )
        
        model.model.load_state_dict(data['model_state'])
        model.model.to(model.device)
        model.model.eval()
        model.is_trained = data['is_trained']
        
        return model

