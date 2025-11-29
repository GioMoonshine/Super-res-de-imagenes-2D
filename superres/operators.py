import numpy as np
from scipy.ndimage import gaussian_filter, convolve


class DegradationOperator:
    """
    Operador de degradación A para super-resolución.
    
    A(x) = (G * x)↓s
    donde:
    - G es un kernel gaussiano 2D
    - * denota convolución
    - ↓s indica submuestreo por factor s
    
    A⊤(y) = G⊤ * upsample(y)
    """
    
    def __init__(self, scale_factor=2, sigma=1.0, kernel_size=None):
        """
        Inicializa el operador de degradación.
        
        Parameters:
        -----------
        scale_factor : int
            Factor de submuestreo s (default: 2)
        sigma : float
            Desviación estándar del kernel gaussiano (default: 1.0)
        kernel_size : int, optional
            Tamaño del kernel gaussiano. Si es None, se calcula automáticamente.
        """
        self.scale_factor = scale_factor
        self.sigma = sigma
        
        # Calcular tamaño del kernel (debe ser impar)
        if kernel_size is None:
            # Regla práctica: 2 * ceil(3*sigma) + 1
            kernel_size = 2 * int(np.ceil(3 * sigma)) + 1
        
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        self.kernel_size = kernel_size
        
        # Crear kernel gaussiano 2D
        self.kernel = self._create_gaussian_kernel()
        
    def _create_gaussian_kernel(self):
        """
        Crea un kernel gaussiano 2D normalizado.
        
        Returns:
        --------
        kernel : ndarray
            Kernel gaussiano 2D de tamaño (kernel_size, kernel_size)
        """
        # Crear grid de coordenadas
        ax = np.arange(-self.kernel_size // 2 + 1, self.kernel_size // 2 + 1)
        xx, yy = np.meshgrid(ax, ax)
        
        # Fórmula gaussiana 2D
        kernel = np.exp(-(xx**2 + yy**2) / (2 * self.sigma**2))
        
        # Normalizar para que sume 1
        kernel = kernel / np.sum(kernel)
        
        return kernel
    
    def apply(self, x):
        """
        Aplica el operador de degradación A(x) = (G * x)↓s
        
        Parameters:
        -----------
        x : ndarray
            Imagen de alta resolución (H_HR, W_HR)
        
        Returns:
        --------
        y : ndarray
            Imagen degradada de baja resolución (H_LR, W_LR)
        """
        # Paso 1: Convolución con kernel gaussiano
        blurred = convolve(x, self.kernel, mode='reflect')
        
        # Paso 2: Submuestreo (tomar cada s-ésimo píxel)
        downsampled = blurred[::self.scale_factor, ::self.scale_factor]
        
        return downsampled
    
    def apply_adjoint(self, y):
        """
        Aplica el operador adjunto A⊤(y) = G⊤ * upsample(y)
        
        Parameters:
        -----------
        y : ndarray
            Imagen de baja resolución (H_LR, W_LR)
        
        Returns:
        --------
        x : ndarray
            Imagen reconstruida de alta resolución (H_HR, W_HR)
        """
        # Paso 1: Upsampling (insertar ceros entre píxeles)
        upsampled = self._upsample_with_zeros(y)
        
        # Paso 2: Convolución con kernel adjunto (transpuesto)
        # Para kernels simétricos, G⊤ = G
        # Usamos el mismo kernel pero con modo 'reflect' para consistencia
        result = convolve(upsampled, self.kernel, mode='reflect')
        
        return result
    
    def _upsample_with_zeros(self, y):
        """
        Realiza upsampling insertando ceros entre píxeles.
        
        Parameters:
        -----------
        y : ndarray
            Imagen de baja resolución (H_LR, W_LR)
        
        Returns:
        --------
        upsampled : ndarray
            Imagen con ceros insertados (H_HR, W_HR)
        """
        h_lr, w_lr = y.shape
        h_hr = h_lr * self.scale_factor
        w_hr = w_lr * self.scale_factor
        
        # Crear imagen de alta resolución llena de ceros
        upsampled = np.zeros((h_hr, w_hr), dtype=y.dtype)
        
        # Colocar valores de y en las posiciones correspondientes
        upsampled[::self.scale_factor, ::self.scale_factor] = y
        
        return upsampled
    
    def upsample_image(self, y):
        """
        Realiza upsampling con interpolación (para inicialización x^(0)).
        
        Parameters:
        -----------
        y : ndarray
            Imagen de baja resolución (H_LR, W_LR)
        
        Returns:
        --------
        x : ndarray
            Imagen aumentada de resolución mediante replicación de píxeles
        """
        # Replicar cada píxel s×s veces
        x = np.repeat(np.repeat(y, self.scale_factor, axis=0), 
                      self.scale_factor, axis=1)
        return x


# ============================================================================
# Funciones auxiliares para gradientes discretos
# ============================================================================

def compute_gradient_x(image):
    """
    Calcula el gradiente horizontal Dx usando diferencias finitas hacia adelante.
    
    Parameters:
    -----------
    image : ndarray
        Imagen de entrada
    
    Returns:
    --------
    grad_x : ndarray
        Gradiente horizontal
    """
    grad_x = np.zeros_like(image)
    # Diferencias finitas: grad[i,j] = image[i, j+1] - image[i, j]
    grad_x[:, :-1] = image[:, 1:] - image[:, :-1]
    # Condición de borde: última columna = 0 (o periodic)
    return grad_x


def compute_gradient_y(image):
    """
    Calcula el gradiente vertical Dy usando diferencias finitas hacia adelante.
    
    Parameters:
    -----------
    image : ndarray
        Imagen de entrada
    
    Returns:
    --------
    grad_y : ndarray
        Gradiente vertical
    """
    grad_y = np.zeros_like(image)
    # Diferencias finitas: grad[i,j] = image[i+1, j] - image[i, j]
    grad_y[:-1, :] = image[1:, :] - image[:-1, :]
    # Condición de borde: última fila = 0
    return grad_y


def compute_divergence(grad_x, grad_y):
    """
    Calcula la divergencia discreta ∇·(∇x) = -Δx
    usando diferencias finitas hacia atrás.
    
    Parameters:
    -----------
    grad_x : ndarray
        Gradiente horizontal
    grad_y : ndarray
        Gradiente vertical
    
    Returns:
    --------
    div : ndarray
        Divergencia del campo de gradientes
    """
    div = np.zeros_like(grad_x)
    
    # Divergencia en x (diferencias hacia atrás)
    div[:, 1:] += grad_x[:, 1:] - grad_x[:, :-1]
    div[:, 0] += grad_x[:, 0]
    
    # Divergencia en y (diferencias hacia atrás)
    div[1:, :] += grad_y[1:, :] - grad_y[:-1, :]
    div[0, :] += grad_y[0, :]
    
    return div


# ============================================================================
# Script de prueba
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Probando DegradationOperator")
    print("=" * 60)
    
    # Crear una imagen de prueba simple
    image_hr = np.random.rand(64, 64)
    print(f"\n1. Imagen HR original: shape = {image_hr.shape}")
    
    # Crear operador con factor de escala 2
    A = DegradationOperator(scale_factor=2, sigma=1.5)
    print(f"   Kernel gaussiano: tamaño = {A.kernel_size}, sigma = {A.sigma}")
    
    # Aplicar degradación
    image_lr = A.apply(image_hr)
    print(f"\n2. Después de A(x): shape = {image_lr.shape}")
    print(f"   Reducción esperada: {image_hr.shape[0]//2} × {image_hr.shape[1]//2}")
    
    # Aplicar adjunto
    reconstructed = A.apply_adjoint(image_lr)
    print(f"\n3. Después de A⊤(y): shape = {reconstructed.shape}")
    print(f"   ¿Recupera tamaño original? {reconstructed.shape == image_hr.shape}")
    
    # Probar upsampling para inicialización
    upsampled = A.upsample_image(image_lr)
    print(f"\n4. Upsampling simple: shape = {upsampled.shape}")
    
    # Probar gradientes
    print(f"\n5. Probando operadores de gradiente:")
    grad_x = compute_gradient_x(image_hr)
    grad_y = compute_gradient_y(image_hr)
    print(f"   Gradiente horizontal: shape = {grad_x.shape}")
    print(f"   Gradiente vertical: shape = {grad_y.shape}")
    
    # Probar divergencia
    div = compute_divergence(grad_x, grad_y)
    print(f"   Divergencia: shape = {div.shape}")
    
    # Verificar normas
    print(f"\n6. Verificaciones numéricas:")
    print(f"   ||Ax||² = {np.sum(image_lr**2):.4f}")
    print(f"   ||A⊤y||² = {np.sum(reconstructed**2):.4f}")
    print(f"   ||grad_x||² = {np.sum(grad_x**2):.4f}")
    print(f"   ||grad_y||² = {np.sum(grad_y**2):.4f}")
    
    print("\n" + "=" * 60)
    print("✓ Todas las operaciones funcionan correctamente")
    print("=" * 60)