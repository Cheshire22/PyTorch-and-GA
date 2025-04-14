import torch
import torch.nn as nn
import numpy as np


class NetControl:
    def __init__(self, input_size, hidden_size, output_size):
        """
        :param input_size: размер входа (4 для CartPole)
        :param hidden_size: размер скрытого слоя
        :param output_size: размер выхода (1 для CartPole)
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.model = self._build_model()

    @staticmethod
    def getTotalWeights(input_size, hidden_size, output_size):
        """Вычисляет общее количество весов (включая смещения)"""
        return (input_size + 1) * hidden_size + (hidden_size + 1) * output_size

    def _build_model(self):
        """Создаёт модель с одним скрытым слоем"""
        return nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size),
            nn.Sigmoid()
        )

    def get_weights(self):
        """Возвращает все веса как единый плоский вектор"""
        weights = []
        for param in self.model.parameters():
            weights.append(param.data.cpu().numpy().flatten())
        return np.concatenate(weights)

    def set_weights(self, weights):
        """Устанавливает веса из плоского вектора"""
        ptr = 0
        with torch.no_grad():
            for param in self.model.parameters():
                # Получаем размер текущего параметра
                param_size = param.data.numel()

                # Извлекаем соответствующую часть весов
                weights_part = weights[ptr:ptr + param_size]

                # Преобразуем и устанавливаем веса
                param.data = torch.FloatTensor(weights_part).reshape(param.data.shape)
                ptr += param_size

    def predict(self, x):
        """Предсказание для CartPole (возвращает 0 или 1)"""
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
        with torch.no_grad():
            output = self.model(x)
        return 1 if output.item() > 0.5 else 0