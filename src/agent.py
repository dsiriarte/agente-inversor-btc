"""Agente DQN con Stable-Baselines3.

MlpPolicy 64x64 con ReLU, replay buffer, red objetivo y exploración
epsilon-greedy lineal.
"""
import random

import numpy as np
import torch
from stable_baselines3 import DQN

PARAMS_SB3 = ["learning_rate", "buffer_size", "learning_starts", "batch_size", "gamma",
              "train_freq", "gradient_steps", "target_update_interval", "exploration_fraction",
              "exploration_initial_eps", "exploration_final_eps"]


def fijar_semillas(semilla: int) -> None:
    """Semillas de Python, NumPy y PyTorch. Un hilo para que el cálculo sea repetible."""
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    torch.set_num_threads(1)


def crear_agente(env, cfg_agente: dict, semilla: int) -> DQN:
    fijar_semillas(semilla)
    params = {k: cfg_agente[k] for k in PARAMS_SB3}
    return DQN("MlpPolicy", env, seed=semilla, verbose=0, device="cpu",
               policy_kwargs={"net_arch": list(cfg_agente["net_arch"])}, **params)


def cargar_agente(ruta) -> DQN:
    return DQN.load(str(ruta), device="cpu")


class PoliticaDQN:
    """Envuelve el modelo para usarlo igual que los baselines: act(obs) -> acción."""

    def __init__(self, modelo: DQN, nombre: str = "DQN"):
        self.modelo, self.nombre = modelo, nombre

    def reiniciar(self) -> None:
        pass

    def act(self, obs) -> int:
        accion, _ = self.modelo.predict(obs, deterministic=True)
        return int(accion)

    def valores_q(self, obs) -> np.ndarray:
        with torch.no_grad():
            t = torch.as_tensor(np.asarray(obs, dtype=np.float32)).reshape(1, -1)
            return self.modelo.q_net(t).numpy()[0]
