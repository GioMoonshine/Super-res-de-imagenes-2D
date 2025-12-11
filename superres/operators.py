import numpy as np
from scipy.ndimage import convolve

class DegradationOperator:
    
    #Operador que simula el desenfoque y la reducción de resolución.
    
    def __init__(self, scale_factor=2, sigma=1.0, kernel_size=None):
        self.scale_factor = scale_factor
        self.sigma = sigma
        
        # Calculamos el tamaño del filtro si no nos lo dan
        if kernel_size is None:
            kernel_size = 2 * int(np.ceil(3 * sigma)) + 1
        
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        self.kernel_size = kernel_size
        self.kernel = self._create_gaussian_kernel()
        
    def _create_gaussian_kernel(self):

        k_half = self.kernel_size // 2
        kernel = np.zeros((self.kernel_size, self.kernel_size))
        
        # Recorremos cada posición para aplicar la fórmula de la campana de Gauss
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                # Coordenadas centradas (ej: -1, 0, 1)
                y = i - k_half
                x = j - k_half
                
                # Fórmula exponencial
                kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * self.sigma**2))
        
        # Normalizamos para que la suma sea 1 (conservación de energía)
        suma_total = np.sum(kernel)
        kernel = kernel / suma_total
        
        return kernel
    
    def apply(self, x):
        # 1. Convolución (Blur)
        blurred = convolve(x, self.kernel, mode='reflect')
        
        # 2. Decimación (tomar 1 de cada 's' píxeles)
        # Hacemos el slicing directo, que es lo estándar
        downsampled = blurred[::self.scale_factor, ::self.scale_factor]
        return downsampled
    
    def apply_adjoint(self, y):
        # 1. Upsample (rellenar con ceros)
        h_lr, w_lr = y.shape
        h_hr = h_lr * self.scale_factor
        w_hr = w_lr * self.scale_factor
        
        upsampled = np.zeros((h_hr, w_hr))
        
        # Rellenar la grilla manualmente
        for i in range(h_lr):
            for j in range(w_lr):
                upsampled[i * self.scale_factor, j * self.scale_factor] = y[i, j]
        
        # 2. Convolución transpuesta (mismo kernel por ser simétrico)
        result = convolve(upsampled, self.kernel, mode='reflect')
        return result
    
    def upsample_image(self, y):
        #Vecino más cercano manual.
        h, w = y.shape
        new_h = h * self.scale_factor
        new_w = w * self.scale_factor
        x = np.zeros((new_h, new_w))
        
        for i in range(new_h):
            for j in range(new_w):
                # Mapeo inverso de coordenadas
                orig_i = i // self.scale_factor
                orig_j = j // self.scale_factor
                x[i, j] = y[orig_i, orig_j]
        return x


def compute_gradient_x(image):
    #Derivada horizontal usando diferencias finitas: f(x+1) - f(x).
    filas, cols = image.shape
    grad_x = np.zeros_like(image)
    
    for i in range(filas):
        for j in range(cols - 1): # Hasta el penúltimo para no salirnos del borde
            grad_x[i, j] = image[i, j+1] - image[i, j]
            
    return grad_x

def compute_gradient_y(image):
    #Derivada vertical usando diferencias finitas: f(y+1) - f(y).
    filas, cols = image.shape
    grad_y = np.zeros_like(image)
    
    for i in range(filas - 1):
        for j in range(cols):
            grad_y[i, j] = image[i+1, j] - image[i, j]
            
    return grad_y

def compute_divergence(grad_x, grad_y):
    #Divergencia (Adjunto negativo del gradiente).
    #Usa diferencias hacia atrás (Backward differences).
    filas, cols = grad_x.shape
    div = np.zeros((filas, cols))
    
    # Aporte eje X
    for i in range(filas):
        for j in range(cols):
            if j == 0:
                val_x = grad_x[i, j] # Borde izquierdo
            elif j == cols - 1:
                val_x = -grad_x[i, j-1] # Borde derecho
            else:
                val_x = grad_x[i, j] - grad_x[i, j-1] # Centro
            div[i, j] += val_x

    # Aporte eje Y
    for i in range(filas):
        for j in range(cols):
            if i == 0:
                val_y = grad_y[i, j]
            elif i == filas - 1:
                val_y = -grad_y[i-1, j]
            else:
                val_y = grad_y[i, j] - grad_y[i-1, j]
            div[i, j] += val_y
            
    return div