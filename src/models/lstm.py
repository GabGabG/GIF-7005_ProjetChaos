""""""
import torch
import numpy as np
from src.models.base_model import BaseModel


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

    def predict(self, start_state: np.ndarray, n_steps: int) -> np.ndarray:
        """Prédiction autonome itérative à partir d'un état initial.

        Args:
            start_state: array de shape (F,) représentant l'état initial.
            n_steps: nombre de pas à prédire.
        Returns:
            np.ndarray de shape (n_steps, F)
        """
        self.eval()
        preds = []
        with torch.no_grad():
            x = torch.tensor(start_state, dtype=torch.float32).view(1, 1, -1).to(self.device)
            h_c = None
            for _ in range(n_steps):
                out, h_c = self.lstm(x, h_c)          # (1, 1, hidden)
                y = self.fc(out[:, -1, :])            # (1, F)
                preds.append(y.squeeze(0).cpu().numpy())
                x = y.view(1, 1, -1)                  # réinjecte la sortie comme prochaine entrée
        return np.vstack(preds)
