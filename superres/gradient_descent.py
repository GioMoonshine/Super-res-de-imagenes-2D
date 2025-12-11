import numpy as np
from .operators import DegradationOperator
from .regularizers import L2GradientRegularizer, HuberGradientRegularizer

class SuperResolutionSolver:
    """
    Solver principal usando Descenso de Gradiente.
    """

    def __init__(self, degradation_op, regularizer, lambda_reg=0.01,
                 tau=0.001, max_iter=100, tolerance=1e-6, verbose=True):
        
        self.A = degradation_op
        self.regularizer = regularizer
        self.lambda_reg = lambda_reg
        self.tau = tau
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.verbose = verbose

        # Listas para guardar el historial y graficar después
        self.cost_history = []
        self.data_term_history = []
        self.reg_term_history = []
        self.gradient_norm_history = []

    def compute_cost(self, x, b):
        # 1. Error de fidelidad (qué tanto se parece a la imagen borrosa)
        # Ax - b
        residuo = self.A.apply(x) - b
        
        # Norma L2 al cuadrado manual
        data_cost = 0.5 * np.sum(residuo**2)

        # 2. Costo de regularización (qué tan "fea" o ruidosa es la imagen)
        reg_val = self.regularizer.compute_value(x)
        reg_cost = self.lambda_reg * reg_val

        return data_cost + reg_cost, data_cost, reg_cost

    def compute_gradient(self, x, b):
        # Gradiente de datos: A.T * (Ax - b)
        residuo = self.A.apply(x) - b
        grad_datos = self.A.apply_adjoint(residuo)

        # Gradiente de regularización
        grad_reg = self.lambda_reg * self.regularizer.compute_gradient(x)

        return grad_datos + grad_reg

    def solve(self, b):
        # Inicialización simple: escalar la imagen pequeña
        x = self.A.upsample_image(b)
        
        if self.verbose:
            print(f"Iniciando optimización por {self.max_iter} iteraciones...")

        converged = False

        # Bucle principal de optimización
        for i in range(self.max_iter):
            
            # 1. Calcular costos actuales para monitoreo
            costo_total, costo_datos, costo_reg = self.compute_cost(x, b)
            
            # 2. Calcular la dirección hacia donde moverse (gradiente)
            grad = self.compute_gradient(x, b)
            magnitud_grad = np.linalg.norm(grad)

            # Guardamos datos
            self.cost_history.append(costo_total)
            self.data_term_history.append(costo_datos)
            self.reg_term_history.append(costo_reg)
            self.gradient_norm_history.append(magnitud_grad)

            # Imprimir info cada 10 iteraciones
            if self.verbose and i % 10 == 0:
                print(f"Iter {i}: Costo={costo_total:.4f} (Datos={costo_datos:.4f}, Reg={costo_reg:.4f})")

            # 3. Paso de descenso (Actualización de la imagen)
            # x_nuevo = x_viejo - paso * gradiente
            x = x - self.tau * grad
            
            # Chequeo de seguridad: evitar valores infinitos
            if np.any(np.isnan(x)):
                print("¡Error! La optimización divergió (valores NaN).")
                break

        return x, {
            'cost_history': self.cost_history,
            'data_term_history': self.data_term_history,
            'reg_term_history': self.reg_term_history,
            'gradient_norm_history': self.gradient_norm_history
        }