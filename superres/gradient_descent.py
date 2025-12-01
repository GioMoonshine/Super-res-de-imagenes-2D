import numpy as np
from scipy.ndimage import gaussian_filter, convolve
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

# ============================================================================
# Herramientas de Cálculo Diferencial Discreto
# ============================================================================

def compute_gradient_x(image):
    """
    Obtiene la derivada parcial discreta respecto al eje horizontal (x).
    
    Usa el método de diferencias finitas adelantadas (forward difference).

    Entrada:
    --------
    image : ndarray
        Matriz de la imagen original.

    Salida:
    -------
    grad_x : ndarray
        Matriz con los cambios de intensidad horizontal.
    """
    grad_x = np.zeros_like(image)
    # Cálculo: f(x+1) - f(x)
    grad_x[:, :-1] = image[:, 1:] - image[:, :-1]
    # Frontera: se asume derivada cero en el borde derecho
    return grad_x


def compute_gradient_y(image):
    """
    Obtiene la derivada parcial discreta respecto al eje vertical (y).
    
    Usa el método de diferencias finitas adelantadas.

    Entrada:
    --------
    image : ndarray
        Matriz de la imagen.

    Salida:
    -------
    grad_y : ndarray
        Matriz con los cambios de intensidad vertical.
    """
    grad_y = np.zeros_like(image)
    # Cálculo: f(y+1) - f(y)
    grad_y[:-1, :] = image[1:, :] - image[:-1, :]
    # Frontera: se asume derivada cero en el borde inferior
    return grad_y


def compute_divergence(grad_x, grad_y):
    """
    Calcula la divergencia del campo vectorial formado por los gradientes.
    
    Operación: div(F) = dFx/dx + dFy/dy
    Nota: Se utilizan diferencias hacia atrás (backward) para ser adjunto
    al operador gradiente (forward).

    Entrada:
    --------
    grad_x, grad_y : ndarray
        Componentes del campo vectorial.

    Salida:
    -------
    div : ndarray
        Mapa de divergencia resultante (escalar).
    """
    div = np.zeros_like(grad_x)

    # Aporte del componente X (diferencia backward)
    div[:, 1:] += grad_x[:, 1:] - grad_x[:, :-1]
    div[:, 0] += grad_x[:, 0]

    # Aporte del componente Y (diferencia backward)
    div[1:, :] += grad_y[1:, :] - grad_y[:-1, :]
    div[0, :] += grad_y[0, :]

    return div


# ============================================================================
# Modelo Físico de Degradación (Clase DegradationOperator)
# ============================================================================

class DegradationOperator:
    """
    Simula el proceso de pérdida de calidad de la imagen.
    
    Representa el operador lineal 'A' tal que: LR = A(HR).
    El proceso consta de:
    1. Convolución con filtro de desenfoque (Gaussian Blur).
    2. Submuestreo (Downsampling) reduciendo dimensiones.
    """

    def __init__(self, scale_factor=2, sigma=1.0, kernel_size=None):
        """
        Configura las propiedades del operador de degradación.

        Args:
            scale_factor (int): Cuántas veces se reduce la imagen (ej. 2x).
            sigma (float): Intensidad del desenfoque gaussiano.
            kernel_size (int, opcional): Tamaño de la matriz de convolución.
        """
        self.scale_factor = scale_factor
        self.sigma = sigma

        # Definición automática del tamaño del filtro si no se especifica
        if kernel_size is None:
            # Fórmula estándar para cubrir +/- 3 desviaciones estándar
            kernel_size = 2 * int(np.ceil(3 * sigma)) + 1

        # Asegurar tamaño impar para tener un píxel central
        if kernel_size % 2 == 0:
            kernel_size += 1

        self.kernel_size = kernel_size
        
        # Generación del filtro
        self.kernel = self._create_gaussian_kernel()

    def _create_gaussian_kernel(self):
        """
        Genera la matriz del filtro Gaussiano normalizado.
        """
        # Ejes de coordenadas centrados en 0
        ax = np.arange(-self.kernel_size // 2 + 1, self.kernel_size // 2 + 1)
        xx, yy = np.meshgrid(ax, ax)

        # Ecuación de la campana de Gauss
        kernel = np.exp(-(xx**2 + yy**2) / (2 * self.sigma**2))

        # Normalización (la suma debe ser 1 para no alterar el brillo global)
        kernel = kernel / np.sum(kernel)

        return kernel

    def apply(self, x):
        """
        Ejecuta la degradación directa: HR -> LR.
        
        Operación: Submuestreo( Convolución(x, kernel) )
        """
        # 1. Aplicar borrosidad
        blurred = convolve(x, self.kernel, mode='reflect')

        # 2. Reducir resolución (saltar píxeles)
        downsampled = blurred[::self.scale_factor, ::self.scale_factor]

        return downsampled

    def apply_adjoint(self, y):
        """
        Ejecuta el operador transpuesto (Adjunto): LR -> HR.
        
        Necesario para el cálculo del gradiente en la optimización.
        Operación: Convolución( Upsample(y), kernel_transpuesto )
        """
        # 1. Aumentar tamaño rellenando con ceros
        upsampled = self._upsample_with_zeros(y)

        # 2. Convolución (el kernel gaussiano es simétrico, su transpuesta es igual)
        result = convolve(upsampled, self.kernel, mode='reflect')

        return result

    def _upsample_with_zeros(self, y):
        """
        Expande la imagen intercalando ceros (Zero-padding grid).
        """
        h_lr, w_lr = y.shape
        h_hr = h_lr * self.scale_factor
        w_hr = w_lr * self.scale_factor

        # Lienzo negro
        upsampled = np.zeros((h_hr, w_hr), dtype=y.dtype)

        # Inyectar píxeles LR en la rejilla
        upsampled[::self.scale_factor, ::self.scale_factor] = y

        return upsampled

    def upsample_image(self, y):
        """
        Método simple de interpolación por vecino más cercano.
        Útil para generar una estimación inicial rápida.
        """
        # Repite cada valor 's' veces en ambos ejes
        x = np.repeat(np.repeat(y, self.scale_factor, axis=0),
                      self.scale_factor, axis=1)
        return x


# ============================================================================
# Módulos de Regularización (Priors)
# ============================================================================

class Regularizer(ABC):
    """
    Interfaz base para términos de penalización.
    Define el contrato que deben cumplir los regularizadores.
    """

    @abstractmethod
    def compute_value(self, x):
        """Retorna el valor escalar del costo de regularización."""
        pass

    @abstractmethod
    def compute_gradient(self, x):
        """Retorna el gradiente de la función de regularización (matriz)."""
        pass


class L2GradientRegularizer(Regularizer):
    """
    Regularización de Tikhonov (Norma L2 al cuadrado).
    
    Favorece soluciones suaves penalizando cambios bruscos (bordes fuertes).
    Matemáticamente: Suma de los cuadrados de las derivadas.
    Gradiente asociado: Proporcional al Laplaciano negativo.
    """

    def __init__(self):
        pass

    def compute_value(self, x):
        """
        Calcula: ||Dx||^2 + ||Dy||^2
        """
        grad_x = compute_gradient_x(x)
        grad_y = compute_gradient_y(x)

        value = np.sum(grad_x**2) + np.sum(grad_y**2)
        return value

    def compute_gradient(self, x):
        """
        Calcula el gradiente analítico de la regularización.
        Resultado: -2 * laplaciano(x)
        """
        grad_x = compute_gradient_x(x)
        grad_y = compute_gradient_y(x)

        # La divergencia de los gradientes es el Laplaciano
        div = compute_divergence(grad_x, grad_y)
        
        # Derivada de x^2 es 2x, y por integración por partes aparece el menos
        gradient = -2.0 * div

        return gradient


class HuberGradientRegularizer(Regularizer):
    """
    Regularización Variación Total con suavizado Huber.
    
    Combina lo mejor de dos mundos:
    - Comportamiento L2 (cuadrático) cerca de cero para evitar ruido 'staircasing'.
    - Comportamiento L1 (lineal) para valores grandes para preservar bordes.
    """

    def __init__(self, delta=0.1):
        """
        Args:
            delta (float): Umbral de transición entre comportamiento cuadrático y lineal.
        """
        self.delta = delta

    def _huber_function(self, z):
        """Evalúa la función de costo Huber elemento a elemento."""
        abs_z = np.abs(z)
        
        # Máscara booleana para la zona suave
        is_quadratic = abs_z <= self.delta
        
        values = np.where(
            is_quadratic,
            z**2 / (2 * self.delta),       # Parábola
            abs_z - self.delta / 2         # Línea recta
        )
        return values

    def _huber_derivative(self, z):
        """Evalúa la primera derivada de la función Huber."""
        abs_z = np.abs(z)
        is_quadratic = abs_z <= self.delta
        
        derivs = np.where(
            is_quadratic,
            z / self.delta,    # Derivada de parábola
            np.sign(z)         # Derivada de valor absoluto
        )
        return derivs

    def compute_value(self, x):
        """Suma total de los costos Huber sobre los gradientes de la imagen."""
        grad_x = compute_gradient_x(x)
        grad_y = compute_gradient_y(x)

        huber_x = self._huber_function(grad_x)
        huber_y = self._huber_function(grad_y)

        return np.sum(huber_x) + np.sum(huber_y)

    def compute_gradient(self, x):
        """
        Calcula el gradiente variacional.
        Gradiente = -Divergencia( DerivadaHuber(Gradiente) )
        """
        grad_x = compute_gradient_x(x)
        grad_y = compute_gradient_y(x)

        # Ponderación no lineal de los gradientes
        w_grad_x = self._huber_derivative(grad_x)
        w_grad_y = self._huber_derivative(grad_y)

        div = compute_divergence(w_grad_x, w_grad_y)

        return -div


# ============================================================================
# Motor de Optimización (Solver)
# ============================================================================

class SuperResolutionSolver:
    """
    Clase orquestadora para resolver el problema inverso.
    
    Minimiza la función de energía: E(x) = Fidelidad_Datos + lambda * Regularización
    Algoritmo: Descenso de Gradiente (Gradient Descent).
    """

    def __init__(self, degradation_op, regularizer, lambda_reg=0.01,
                 tau=0.001, max_iter=100, tolerance=1e-6, verbose=True):
        """
        Configuración del solver.

        Args:
            degradation_op: Instancia del operador A.
            regularizer: Instancia del prior R(x).
            lambda_reg: Peso de la regularización (balance suavidad vs fidelidad).
            tau: Tasa de aprendizaje (step size).
            max_iter: Límite de iteraciones.
            tolerance: Criterio de parada por norma del gradiente.
            verbose: Control de mensajes en consola.
        """
        self.A = degradation_op
        self.regularizer = regularizer
        self.lambda_reg = lambda_reg
        self.tau = tau
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.verbose = verbose

        # Almacén de métricas
        self.cost_history = []
        self.data_term_history = []
        self.reg_term_history = []
        self.gradient_norm_history = []

    def compute_cost(self, x, b):
        """Calcula el valor actual de la función objetivo J(x)."""
        # 1. Error de ajuste a los datos (Residuo)
        residual = self.A.apply(x) - b
        data_cost = 0.5 * np.sum(residual**2)

        # 2. Costo del prior
        reg_val = self.regularizer.compute_value(x)
        reg_cost = self.lambda_reg * reg_val

        return data_cost + reg_cost, data_cost, reg_cost

    def compute_gradient(self, x, b):
        """
        Calcula la dirección de máxima pendiente.
        Gradiente Total = Gradiente_Datos + lambda * Gradiente_Reg
        """
        # Gradiente del término ||Ax - b||^2  ->  A.T * (Ax - b)
        residual = self.A.apply(x) - b
        data_grad = self.A.apply_adjoint(residual)

        # Gradiente del prior
        reg_grad = self.lambda_reg * self.regularizer.compute_gradient(x)

        return data_grad + reg_grad

    def solve(self, b, x_init=None):
        """
        Ejecuta el bucle principal de optimización.

        Entrada:
            b: Imagen borrosa/baja resolución (observación).
            x_init: Estimación inicial (opcional).
        """
        # Limpieza de historiales previos
        self.cost_history = []
        self.data_term_history = []
        self.reg_term_history = []
        self.gradient_norm_history = []

        # Punto de partida
        if x_init is None:
            # Si no hay inicialización, escalar la imagen LR simplemente
            x = self.A.upsample_image(b)
        else:
            x = x_init.copy()

        if self.verbose:
            print("#" * 70)
            print(" INICIANDO PROCESO DE RECONSTRUCCIÓN")
            print("#" * 70)
            print(f" Input LR shape: {b.shape}")
            print(f" Parámetros: lambda={self.lambda_reg}, LR={self.tau}")
            print(f" Tipo de Regularización: {type(self.regularizer).__name__}")
            print("-" * 70)

        converged = False

        for i in range(self.max_iter):
            # 1. Evaluar estado actual
            cost, d_term, r_term = self.compute_cost(x, b)

            # 2. Calcular dirección de mejora
            grad = self.compute_gradient(x, b)
            grad_magnitude = np.linalg.norm(grad)

            # 3. Registrar telemetría
            self.cost_history.append(cost)
            self.data_term_history.append(d_term)
            self.reg_term_history.append(r_term)
            self.gradient_norm_history.append(grad_magnitude)

            # Reporte periódico
            if self.verbose and (i % 10 == 0 or i == self.max_iter - 1):
                print(f" Ciclo {i:3d} | Costo Total: {cost:.5e} | "
                      f"Datos: {d_term:.5e} | Reg: {r_term:.5e} | "
                      f"Mag. Grad: {grad_magnitude:.5e}")

            # Chequeo de convergencia
            if grad_magnitude < self.tolerance:
                converged = True
                if self.verbose:
                    print(f"\n>>> Convergencia prematura en iteración {i}")
                break

            # 4. Actualización de parámetros (Paso de descenso)
            x = x - self.tau * grad

        if self.verbose:
            print("-" * 70)
            status = "FINALIZADO (Convergencia)" if converged else "FINALIZADO (Máx Iteraciones)"
            print(f" Estado: {status}")
            print(f" Error Final: {self.cost_history[-1]:.5e}")
            print("#" * 70)

        stats = {
            'cost_history': np.array(self.cost_history),
            'data_term_history': np.array(self.data_term_history),
            'reg_term_history': np.array(self.reg_term_history),
            'gradient_norm_history': np.array(self.gradient_norm_history),
            'iterations': len(self.cost_history),
            'converged': converged
        }

        return x, stats


# ============================================================================
# Bloque Principal de Ejecución (Demo)
# ============================================================================

if __name__ == "__main__":

    print("\n" + "*" * 70)
    print(" EJECUCIÓN DEL SCRIPT DE DEMOSTRACIÓN DE SUPER-RESOLUCIÓN")
    print("*" * 70)

    # ------------------------------------------------------------------------
    # 1. Generación de datos sintéticos (Ground Truth)
    # ------------------------------------------------------------------------
    print("\n[1/6] Generando imagen original de alta calidad...")

    size = 64
    x_true = np.zeros((size, size))

    # Dibujando patrones geométricos
    x_true[10:30, 10:30] = 0.8  # Bloque claro
    x_true[35:55, 35:55] = 0.6  # Bloque medio
    x_true[15:25, 40:50] = 1.0  # Bloque brillante

    # Añadiendo imperfecciones (ruido)
    np.random.seed(42)
    x_true += 0.05 * np.random.randn(size, size)
    x_true = np.clip(x_true, 0, 1)

    print(f"   -> Dimensiones: {x_true.shape}")

    # ------------------------------------------------------------------------
    # 2. Simulación de la captura (Degradación)
    # ------------------------------------------------------------------------
    print("\n[2/6] Simulando captura en baja resolución (LR)...")

    s_factor = 2
    op_degradation = DegradationOperator(scale_factor=s_factor, sigma=1.5)

    b_obs = op_degradation.apply(x_true)
    print(f"   -> Imagen observada: {b_obs.shape}")
    print(f"   -> Reducción aplicada: {s_factor}x")

    # ------------------------------------------------------------------------
    # 3. Reconstrucción con L2
    # ------------------------------------------------------------------------
    print("\n[3/6] Restaurando con regularizador L2 (Suavizado)...")

    reg_l2_inst = L2GradientRegularizer()
    solver_l2 = SuperResolutionSolver(
        degradation_op=op_degradation,
        regularizer=reg_l2_inst,
        lambda_reg=0.01,
        tau=0.0005,
        max_iter=100,
        verbose=True
    )

    rec_l2, info_l2 = solver_l2.solve(b_obs)

    # ------------------------------------------------------------------------
    # 4. Reconstrucción con Huber
    # ------------------------------------------------------------------------
    print("\n[4/6] Restaurando con regularizador Huber (Preservación de bordes)...")

    reg_huber_inst = HuberGradientRegularizer(delta=0.1)
    solver_huber = SuperResolutionSolver(
        degradation_op=op_degradation,
        regularizer=reg_huber_inst,
        lambda_reg=0.01,
        tau=0.0005,
        max_iter=100,
        verbose=True
    )

    rec_huber, info_huber = solver_huber.solve(b_obs)

    # ------------------------------------------------------------------------
    # 5. Análisis Cuantitativo
    # ------------------------------------------------------------------------
    print("\n[5/6] Evaluación de Resultados:")
    print("-" * 40)

    # Cálculo de error relativo normalizado
    err_l2 = np.linalg.norm(rec_l2 - x_true) / np.linalg.norm(x_true)
    err_huber = np.linalg.norm(rec_huber - x_true) / np.linalg.norm(x_true)

    print(f"Error Relativo (L2):    {err_l2:.6f}")
    print(f"Error Relativo (Huber): {err_huber:.6f}")
    print(f"Iteraciones totales:    L2={info_l2['iterations']}, Huber={info_huber['iterations']}")

    # ------------------------------------------------------------------------
    # 6. Salida Gráfica
    # ------------------------------------------------------------------------
    print("\n[6/6] Generando reportes visuales...")

    try:
        # Gráfica de curvas de aprendizaje
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # J(x)
        axes[0, 0].semilogy(info_l2['cost_history'], 'b-', label='L2')
        axes[0, 0].semilogy(info_huber['cost_history'], 'r-', label='Huber')
        axes[0, 0].set_title('Evolución del Costo Total')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

        # Término de Fidelidad
        axes[0, 1].semilogy(info_l2['data_term_history'], 'b-', label='L2')
        axes[0, 1].semilogy(info_huber['data_term_history'], 'r-', label='Huber')
        axes[0, 1].set_title('Error de Datos (Fidelidad)')
        axes[0, 1].grid(alpha=0.3)

        # Término de Regularización
        axes[1, 0].semilogy(info_l2['reg_term_history'], 'b-', label='L2')
        axes[1, 0].semilogy(info_huber['reg_term_history'], 'r-', label='Huber')
        axes[1, 0].set_title('Valor de Regularización')
        axes[1, 0].grid(alpha=0.3)

        # Convergencia del Gradiente
        axes[1, 1].semilogy(info_l2['gradient_norm_history'], 'b-', label='L2')
        axes[1, 1].semilogy(info_huber['gradient_norm_history'], 'r-', label='Huber')
        axes[1, 1].set_title('Magnitud del Gradiente')
        axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig('convergence_comparison.png')
        print("   -> Guardado: 'convergence_comparison.png'")

        # Comparativa Visual de Imágenes
        fig2, axes2 = plt.subplots(2, 3, figsize=(12, 8))

        # Fila 1: Referencias
        axes2[0, 0].imshow(x_true, cmap='gray', vmin=0, vmax=1)
        axes2[0, 0].set_title('Original (HR)')
        axes2[0, 0].axis('off')

        axes2[0, 1].imshow(b_obs, cmap='gray', vmin=0, vmax=1)
        axes2[0, 1].set_title('Entrada (LR)')
        axes2[0, 1].axis('off')

        axes2[0, 2].imshow(op_degradation.upsample_image(b_obs), cmap='gray', vmin=0, vmax=1)
        axes2[0, 2].set_title('Escalado Simple (Nearest)')
        axes2[0, 2].axis('off')

        # Fila 2: Resultados
        axes2[1, 0].imshow(rec_l2, cmap='gray', vmin=0, vmax=1)
        axes2[1, 0].set_title(f'Resultado L2\nErr: {err_l2:.4f}')
        axes2[1, 0].axis('off')

        axes2[1, 1].imshow(rec_huber, cmap='gray', vmin=0, vmax=1)
        axes2[1, 1].set_title(f'Resultado Huber\nErr: {err_huber:.4f}')
        axes2[1, 1].axis('off')

        # Mapa de discrepancia
        diff_img = np.abs(rec_l2 - rec_huber)
        axes2[1, 2].imshow(diff_img, cmap='inferno', vmin=0, vmax=0.1)
        axes2[1, 2].set_title('Diferencia L2 vs Huber')
        axes2[1, 2].axis('off')

        plt.tight_layout()
        plt.savefig('reconstruction_comparison.png')
        print("   -> Guardado: 'reconstruction_comparison.png'")

    except ImportError:
        print("   [!] Matplotlib no está instalado, omitiendo gráficas.")

    print("\n" + "*" * 70)
    print(" PROGRAMA FINALIZADO CORRECTAMENTE")
    print("*" * 70)
