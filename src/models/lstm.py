""""""
import torch
import numpy as np
from src.models.base_model import BaseModel
from src.edo_data_structures import normalisation, denormalisation


class LSTM(BaseModel):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int) -> None:
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass du LSTM.
        
        Args:
            x: Tensor de shape (batch, seq_len, input_size)
        Returns:
            Prédictions de shape (batch, output_size) pour le dernier pas
        """
        out, _ = self.lstm(x)                 # (batch, seq_len, hidden)
        last = out[:, -1, :]                  # (batch, hidden)
        return self.fc(last)                  # (batch, output_size)

    def predict(self, start_sequence: np.ndarray, n_steps: int) -> np.ndarray:
        """Prédiction autonome itérative à partir d'une séquence initiale.

        Args:
            start_sequence: array de shape (seq_len, F) ou (F,) pour une séquence initiale.
                           Si shape (F,), sera traité comme une séquence de longueur 1.
            n_steps: nombre de pas à prédire.
        Returns:
            np.ndarray de shape (n_steps, F)
        """
        self.eval()
        preds = []
        
        # Gérer le cas où start_sequence est 1D
        if start_sequence.ndim == 1:
            start_sequence = start_sequence.reshape(1, -1)
        
        with torch.no_grad():
            # Initialiser avec la séquence complète
            x = torch.tensor(start_sequence, dtype=torch.float32).unsqueeze(0).to(self.device)  # (1, seq_len, F)
            
            for _ in range(n_steps):
                out, _ = self.lstm(x)                  # (1, seq_len, hidden)
                y = self.fc(out[:, -1, :])             # (1, F) - prédiction du prochain pas
                preds.append(y.squeeze(0).cpu().numpy())
                
                # Glisser la fenêtre : enlever le premier élément et ajouter la prédiction
                x = torch.cat([x[:, 1:, :], y.unsqueeze(1)], dim=1)  # (1, seq_len, F)
        
        return np.vstack(preds)
