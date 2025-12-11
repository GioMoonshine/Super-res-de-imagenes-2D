import numpy as np
from abc import ABC, abstractmethod
from .operators import compute_gradient_x, compute_gradient_y, compute_divergence

class Regularizer(ABC):
    @abstractmethod
    def compute_value(self, x):
        pass
    
    @abstractmethod
    def compute_gradient(self, x):
        pass

class L2GradientRegularizer(Regularizer):
    #Regularización Tikhonov (Suavizado).
    
    def compute_value(self, x):
        dx = compute_gradient_x(x)
        dy = compute_gradient_y(x)
        
        # Sumatoria manual de los cuadrados
        suma_cuadrados = 0.0
        filas, cols = x.shape
        
        for i in range(filas):
            for j in range(cols):
                suma_cuadrados += dx[i, j]**2 + dy[i, j]**2
                
        return suma_cuadrados
    
    def compute_gradient(self, x):
        dx = compute_gradient_x(x)
        dy = compute_gradient_y(x)
        
        # El gradiente es -2 * divergencia (Laplaciano)
        div = compute_divergence(dx, dy)
        return -2.0 * div

class HuberGradientRegularizer(Regularizer):
    #Regularización robusta que preserva bordes.
    
    def __init__(self, delta=0.1):
        self.delta = delta
        
    def compute_value(self, x):
        dx = compute_gradient_x(x)
        dy = compute_gradient_y(x)
        
        total_cost = 0.0
        filas, cols = x.shape
        
        # Evaluamos la función de Huber pixel a pixel
        for i in range(filas):
            for j in range(cols):
                # Eje X
                val_x = abs(dx[i, j])
                if val_x <= self.delta:
                    total_cost += (val_x**2) / (2 * self.delta)
                else:
                    total_cost += val_x - (self.delta / 2)
                
                # Eje Y
                val_y = abs(dy[i, j])
                if val_y <= self.delta:
                    total_cost += (val_y**2) / (2 * self.delta)
                else:
                    total_cost += val_y - (self.delta / 2)
                    
        return total_cost
    
    def compute_gradient(self, x):
        dx = compute_gradient_x(x)
        dy = compute_gradient_y(x)
        
        # Calculamos los pesos w_grad manualmente
        w_dx = np.zeros_like(dx)
        w_dy = np.zeros_like(dy)
        filas, cols = x.shape
        
        for i in range(filas):
            for j in range(cols):
                # Derivada Huber para X
                if abs(dx[i, j]) <= self.delta:
                    w_dx[i, j] = dx[i, j] / self.delta
                else:
                    w_dx[i, j] = np.sign(dx[i, j])
                
                # Derivada Huber para Y
                if abs(dy[i, j]) <= self.delta:
                    w_dy[i, j] = dy[i, j] / self.delta
                else:
                    w_dy[i, j] = np.sign(dy[i, j])

        return -compute_divergence(w_dx, w_dy)