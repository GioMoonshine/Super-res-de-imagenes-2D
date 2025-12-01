import numpy as np
from scipy.ndimage import gaussian_filter, convolve

class DegradationOperator:
    """
    Modelo computacional de pérdida de resolución (Operador Lineal A).
    
    Simula el proceso físico de captura:
    A(x) = Decimación( Desenfoque(x) )
    
    Matemáticamente:
    - G: Matriz de convolución (Filtro Paso Bajo Gaussiano).
    - ↓s: Operador de diezmado (Downsampling).
    
    También provee el operador transpuesto A.T, vital para algoritmos iterativos.
    """
    
    def __init__(self, scale_factor=2, sigma=1.0, kernel_size=None):
        """
        Configura los parámetros del sistema de degradación.
        
        Entradas:
        ---------
        scale_factor : int
            Ratio de reducción de tamaño (s). Por defecto: 2.
        sigma : float
            Amplitud del desenfoque (ancho de la campana de Gauss).
        kernel_size : int, opcional
            Dimensión de la matriz del filtro. Se autocalcula si se omite.
        """
        self.scale_factor = scale_factor
        self.sigma = sigma
        
        # Determinación automática del soporte del filtro
        if kernel_size is None:
            # Estimación estándar: cubrir ±3 desviaciones estándar
            kernel_size = 2 * int(np.ceil(3 * sigma)) + 1
        
        # Garantizar simetría central (tamaño impar)
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        self.kernel_size = kernel_size
        
        # Generación de la máscara de convolución
        self.kernel = self._create_gaussian_kernel()
        
    def _create_gaussian_kernel(self):
        """
        Genera la máscara de convolución basada en la distribución Normal 2D.
        
        Salida:
        -------
        kernel : ndarray
            Matriz cuadrada normalizada (suma = 1).
        """
        # Sistema de coordenadas centrado
        ax = np.arange(-self.kernel_size // 2 + 1, self.kernel_size // 2 + 1)
        xx, yy = np.meshgrid(ax, ax)
        
        # Ecuación de la función Gaussiana
        kernel = np.exp(-(xx**2 + yy**2) / (2 * self.sigma**2))
        
        # Normalización de energía
        kernel = kernel / np.sum(kernel)
        
        return kernel
    
    def apply(self, x):
        """
        Ejecuta la simulación de baja resolución (Modelo Directo).
        
        Transformación: Alta Resolución -> Baja Resolución
        
        Entradas:
        ---------
        x : ndarray
            Imagen original nítida.
        
        Salida:
        -------
        y : ndarray
            Imagen resultante borrosa y pequeña.
        """
        # Fase 1: Filtrado espacial (Blur)
        blurred = convolve(x, self.kernel, mode='reflect')
        
        # Fase 2: Decimación (Saltar píxeles según el factor de escala)
        downsampled = blurred[::self.scale_factor, ::self.scale_factor]
        
        return downsampled
    
    def apply_adjoint(self, y):
        """
        Calcula la operación transpuesta (A.T).
        
        Nota: Esto NO es la inversa exacta, sino el adjunto matemático usado
        en el cálculo de gradientes (Backprojection).
        
        Entradas:
        ---------
        y : ndarray
            Imagen de baja resolución.
        
        Salida:
        -------
        x : ndarray
            Proyección en el espacio de alta resolución.
        """
        # Fase 1: Expansión con ceros (Zero-padding grid)
        upsampled = self._upsample_with_zeros(y)
        
        # Fase 2: Convolución correlacionada
        # (Se usa el mismo kernel dado que la Gaussiana es simétrica)
        result = convolve(upsampled, self.kernel, mode='reflect')
        
        return result
    
    def _upsample_with_zeros(self, y):
        """
        Expande la matriz rellenando los espacios nuevos con valores nulos.
        
        Entradas:
        ---------
        y : ndarray
            Imagen pequeña.
        
        Salida:
        -------
        upsampled : ndarray
            Imagen grande esparcida.
        """
        h_lr, w_lr = y.shape
        h_hr = h_lr * self.scale_factor
        w_hr = w_lr * self.scale_factor
        
        # Lienzo vacío
        upsampled = np.zeros((h_hr, w_hr), dtype=y.dtype)
        
        # Inyección de valores conocidos en la rejilla
        upsampled[::self.scale_factor, ::self.scale_factor] = y
        
        return upsampled
    
    def upsample_image(self, y):
        """
        Escalado básico por repetición de vecinos (Nearest Neighbor).
        Útil para generar un punto de partida para la optimización.
        
        Salida:
        -------
        x : ndarray
            Imagen escalada "pixelada".
        """
        # Duplicación de filas y columnas
        x = np.repeat(np.repeat(y, self.scale_factor, axis=0), 
                      self.scale_factor, axis=1)
        return x


# ============================================================================
# Herramientas de Cálculo Diferencial sobre Imágenes
# ============================================================================

def compute_gradient_x(image):
    """
    Obtiene la derivada discreta en el eje de las abscisas (horizontal).
    Utiliza diferencias finitas progresivas (Forward Difference).
    
    Salida:
    -------
    grad_x : ndarray
        Mapa de variaciones horizontales.
    """
    grad_x = np.zeros_like(image)
    # Cálculo: Pixel[siguiente] - Pixel[actual]
    grad_x[:, :-1] = image[:, 1:] - image[:, :-1]
    # Frontera derecha se asume constante (derivada 0)
    return grad_x


def compute_gradient_y(image):
    """
    Obtiene la derivada discreta en el eje de las ordenadas (vertical).
    Utiliza diferencias finitas progresivas.
    
    Salida:
    -------
    grad_y : ndarray
        Mapa de variaciones verticales.
    """
    grad_y = np.zeros_like(image)
    # Cálculo: Pixel[inferior] - Pixel[actual]
    grad_y[:-1, :] = image[1:, :] - image[:-1, :]
    # Frontera inferior se asume constante
    return grad_y


def compute_divergence(grad_x, grad_y):
    """
    Calcula el flujo saliente del campo vectorial.
    Matemáticamente equivale a menos el Laplaciano si se combina con gradientes.
    
    IMPORTANTE: Usa diferencias regresivas (Backward Difference) para asegurar
    que este operador sea el adjunto negativo del gradiente.
    
    Entradas:
    ---------
    grad_x, grad_y : ndarray
        Componentes del campo vectorial.
    
    Salida:
    -------
    div : ndarray
        Escalar de divergencia por pixel.
    """
    div = np.zeros_like(grad_x)
    
    # Acumulación eje X (Backward)
    div[:, 1:] += grad_x[:, 1:] - grad_x[:, :-1]
    div[:, 0] += grad_x[:, 0]
    
    # Acumulación eje Y (Backward)
    div[1:, :] += grad_y[1:, :] - grad_y[:-1, :]
    div[0, :] += grad_y[0, :]
    
    return div


# ============================================================================
# Rutina de Verificación
# ============================================================================

if __name__ == "__main__":
    print("-" * 60)
    print(" EJECUTANDO VALIDACIÓN UNITARIA DEL OPERADOR")
    print("-" * 60)
    
    # Generar ruido aleatorio como imagen base
    image_hr = np.random.rand(64, 64)
    print(f"\n[1] Entrada de Alta Resolución: dim = {image_hr.shape}")
    
    # Instanciación del modelo
    A = DegradationOperator(scale_factor=2, sigma=1.5)
    print(f"    Parámetros: Kernel {A.kernel_size}x{A.kernel_size}, Sigma={A.sigma}")
    
    # Prueba del modelo directo
    image_lr = A.apply(image_hr)
    print(f"\n[2] Resultado Baja Resolución (Ax): dim = {image_lr.shape}")
    print(f"    Dimensiones predichas: {image_hr.shape[0]//2} × {image_hr.shape[1]//2}")
    
    # Prueba del modelo transpuesto
    reconstructed = A.apply_adjoint(image_lr)
    print(f"\n[3] Resultado Transpuesto (A.T*y): dim = {reconstructed.shape}")
    print(f"    ¿Coincide con origen? {reconstructed.shape == image_hr.shape}")
    
    # Prueba de interpolación simple
    upsampled = A.upsample_image(image_lr)
    print(f"\n[4] Interpolación Vecino Cercano: dim = {upsampled.shape}")
    
    # Prueba de cálculo vectorial
    print(f"\n[5] Verificando Operadores Diferenciales:")
    grad_x = compute_gradient_x(image_hr)
    grad_y = compute_gradient_y(image_hr)
    print(f"    Derivada X: dim = {grad_x.shape}")
    print(f"    Derivada Y: dim = {grad_y.shape}")
    
    # Prueba de divergencia
    div = compute_divergence(grad_x, grad_y)
    print(f"    Divergencia (Adjunto): dim = {div.shape}")
    
    # Chequeo de energía (sanity check)
    print(f"\n[6] Métricas de Energía:")
    print(f"    Energía LR (||Ax||²): {np.sum(image_lr**2):.4f}")
    print(f"    Energía Reconstruida: {np.sum(reconstructed**2):.4f}")
    print(f"    Norma Gradiente X:    {np.sum(grad_x**2):.4f}")
    print(f"    Norma Gradiente Y:    {np.sum(grad_y**2):.4f}")
    
    print("\n" + "-" * 60)
    print(">>> TEST COMPLETADO SIN ERRORES")
    print("-" * 60)
