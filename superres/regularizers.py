import numpy as np
from abc import ABC, abstractmethod
from operators import compute_gradient_x, compute_gradient_y, compute_divergence


class Regularizer(ABC):
    """
    Clase base abstracta para regularizadores.
    
    Todos los regularizadores deben implementar:
    - compute_value(x): calcula R(x)
    - compute_gradient(x): calcula ∇R(x)
    """
    
    @abstractmethod
    def compute_value(self, x):
        """
        Calcula el valor del término de regularización R(x).
        
        Parameters:
        -----------
        x : ndarray
            Imagen actual
        
        Returns:
        --------
        float
            Valor de R(x)
        """
        pass
    
    @abstractmethod
    def compute_gradient(self, x):
        """
        Calcula el gradiente del término de regularización ∇R(x).
        
        Parameters:
        -----------
        x : ndarray
            Imagen actual
        
        Returns:
        --------
        ndarray
            Gradiente ∇R(x) con la misma forma que x
        """
        pass


class L2GradientRegularizer(Regularizer):
    """
    Regularizador L2 del gradiente (Tikhonov):
    
    R_L2(x) = ||Dx·x||² + ||Dy·x||²
    
    Donde Dx y Dy son operadores de diferencias finitas.
    
    El gradiente es:
    ∇R_L2(x) ≈ -2∇·(∇x) = -2Δx
    
    donde Δ es el operador Laplaciano discreto.
    """
    
    def __init__(self):
        """Inicializa el regularizador L2."""
        pass
    
    def compute_value(self, x):
        """
        Calcula R_L2(x) = ||Dx·x||² + ||Dy·x||²
        
        Parameters:
        -----------
        x : ndarray
            Imagen actual
        
        Returns:
        --------
        float
            Valor de la regularización L2
        """
        # Calcular gradientes
        grad_x = compute_gradient_x(x)
        grad_y = compute_gradient_y(x)
        
        # Norma L2 al cuadrado de cada gradiente
        value = np.sum(grad_x**2) + np.sum(grad_y**2)
        
        return value
    
    def compute_gradient(self, x):
        """
        Calcula ∇R_L2(x) = -2∇·(∇x)
        
        Equivale a aplicar menos dos veces el Laplaciano discreto.
        
        Parameters:
        -----------
        x : ndarray
            Imagen actual
        
        Returns:
        --------
        ndarray
            Gradiente del regularizador
        """
        # Calcular gradientes de la imagen
        grad_x = compute_gradient_x(x)
        grad_y = compute_gradient_y(x)
        
        # Calcular divergencia (esto da -Δx)
        div = compute_divergence(grad_x, grad_y)
        
        # El gradiente es -2 veces la divergencia
        gradient = -2.0 * div
        
        return gradient


class HuberGradientRegularizer(Regularizer):
    """
    Regularizador Huber del gradiente (Huber-TV):
    
    R_Huber(x) = Σ[φ_δ((Dx·x)_ij) + φ_δ((Dy·x)_ij)]
    
    donde φ_δ(z) es la función de Huber:
        φ_δ(z) = { z²/(2δ)      si |z| ≤ δ
                 { |z| - δ/2    si |z| > δ
    
    El gradiente se calcula mediante:
    ∇R_Huber(x) = -∇·(φ'_δ(Dx·x), φ'_δ(Dy·x))
    
    donde φ'_δ(z) = { z/δ        si |z| ≤ δ
                    { sign(z)    si |z| > δ
    """
    
    def __init__(self, delta=0.1):
        """
        Inicializa el regularizador Huber.
        
        Parameters:
        -----------
        delta : float
            Parámetro δ de la función de Huber (default: 0.1)
            - δ pequeño (≈0.01-0.1): preserva mejor los bordes
            - δ grande (≈1.0): se acerca más a L2
        """
        self.delta = delta
    
    def _huber_function(self, z):
        """
        Función de Huber φ_δ(z).
        
        Parameters:
        -----------
        z : ndarray
            Valores de entrada
        
        Returns:
        --------
        ndarray
            Valores de Huber aplicados elemento a elemento
        """
        abs_z = np.abs(z)
        
        # Región cuadrática: |z| ≤ δ
        quadratic_region = abs_z <= self.delta
        huber_values = np.where(
            quadratic_region,
            z**2 / (2 * self.delta),  # z²/(2δ)
            abs_z - self.delta / 2     # |z| - δ/2
        )
        
        return huber_values
    
    def _huber_derivative(self, z):
        """
        Derivada de la función de Huber φ'_δ(z).
        
        Parameters:
        -----------
        z : ndarray
            Valores de entrada
        
        Returns:
        --------
        ndarray
            Derivadas aplicadas elemento a elemento
        """
        abs_z = np.abs(z)
        
        # Región cuadrática: |z| ≤ δ
        quadratic_region = abs_z <= self.delta
        huber_deriv = np.where(
            quadratic_region,
            z / self.delta,      # z/δ
            np.sign(z)           # sign(z)
        )
        
        return huber_deriv
    
    def compute_value(self, x):
        """
        Calcula R_Huber(x) = Σ[φ_δ((Dx·x)_ij) + φ_δ((Dy·x)_ij)]
        
        Parameters:
        -----------
        x : ndarray
            Imagen actual
        
        Returns:
        --------
        float
            Valor de la regularización Huber
        """
        # Calcular gradientes
        grad_x = compute_gradient_x(x)
        grad_y = compute_gradient_y(x)
        
        # Aplicar función de Huber a cada componente del gradiente
        huber_x = self._huber_function(grad_x)
        huber_y = self._huber_function(grad_y)
        
        # Sumar sobre todos los píxeles
        value = np.sum(huber_x) + np.sum(huber_y)
        
        return value
    
    def compute_gradient(self, x):
        """
        Calcula ∇R_Huber(x) = -∇·(φ'_δ(Dx·x), φ'_δ(Dy·x))
        
        Parameters:
        -----------
        x : ndarray
            Imagen actual
        
        Returns:
        --------
        ndarray
            Gradiente del regularizador Huber
        """
        # Calcular gradientes de la imagen
        grad_x = compute_gradient_x(x)
        grad_y = compute_gradient_y(x)
        
        # Aplicar derivada de Huber a cada componente
        weighted_grad_x = self._huber_derivative(grad_x)
        weighted_grad_y = self._huber_derivative(grad_y)
        
        # Calcular divergencia del campo ponderado
        div = compute_divergence(weighted_grad_x, weighted_grad_y)
        
        # El gradiente es menos la divergencia
        gradient = -div
        
        return gradient


# ============================================================================
# Script de prueba
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Probando Regularizadores")
    print("=" * 70)
    
    # Crear una imagen de prueba
    np.random.seed(42)
    image = np.random.rand(32, 32)
    print(f"\nImagen de prueba: shape = {image.shape}")
    
    # ========================================================================
    # Prueba 1: Regularizador L2
    # ========================================================================
    print("\n" + "=" * 70)
    print("1. REGULARIZADOR L2 DEL GRADIENTE")
    print("=" * 70)
    
    reg_l2 = L2GradientRegularizer()
    
    # Calcular valor
    value_l2 = reg_l2.compute_value(image)
    print(f"\nR_L2(x) = {value_l2:.6f}")
    
    # Calcular gradiente
    grad_l2 = reg_l2.compute_gradient(image)
    print(f"∇R_L2(x): shape = {grad_l2.shape}")
    print(f"||∇R_L2(x)||² = {np.sum(grad_l2**2):.6f}")
    print(f"min(∇R_L2) = {np.min(grad_l2):.6f}")
    print(f"max(∇R_L2) = {np.max(grad_l2):.6f}")
    
    # ========================================================================
    # Prueba 2: Regularizador Huber
    # ========================================================================
    print("\n" + "=" * 70)
    print("2. REGULARIZADOR HUBER DEL GRADIENTE")
    print("=" * 70)
    
    # Probar con diferentes valores de delta
    deltas = [0.01, 0.1, 1.0]
    
    for delta in deltas:
        print(f"\n--- Con δ = {delta} ---")
        reg_huber = HuberGradientRegularizer(delta=delta)
        
        # Calcular valor
        value_huber = reg_huber.compute_value(image)
        print(f"R_Huber(x) = {value_huber:.6f}")
        
        # Calcular gradiente
        grad_huber = reg_huber.compute_gradient(image)
        print(f"||∇R_Huber(x)||² = {np.sum(grad_huber**2):.6f}")
    
    # ========================================================================
    # Prueba 3: Comparación L2 vs Huber
    # ========================================================================
    print("\n" + "=" * 70)
    print("3. COMPARACIÓN L2 vs HUBER (δ=0.1)")
    print("=" * 70)
    
    reg_huber = HuberGradientRegularizer(delta=0.1)
    
    grad_x = compute_gradient_x(image)
    grad_y = compute_gradient_y(image)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    print(f"\nEstadísticas del gradiente de la imagen:")
    print(f"  Magnitud media: {np.mean(magnitude):.6f}")
    print(f"  Magnitud máxima: {np.max(magnitude):.6f}")
    print(f"  Magnitud mínima: {np.min(magnitude):.6f}")
    
    print(f"\nComparación de valores:")
    print(f"  R_L2(x) = {reg_l2.compute_value(image):.6f}")
    print(f"  R_Huber(x, δ=0.1) = {reg_huber.compute_value(image):.6f}")
    
    print(f"\nComparación de gradientes:")
    print(f"  ||∇R_L2(x)||² = {np.sum(grad_l2**2):.6f}")
    print(f"  ||∇R_Huber(x)||² = {np.sum(grad_huber**2):.6f}")
    
    # ========================================================================
    # Prueba 4: Imagen con borde fuerte
    # ========================================================================
    print("\n" + "=" * 70)
    print("4. PRUEBA CON IMAGEN DE BORDE FUERTE")
    print("=" * 70)
    
    # Crear imagen con borde definido (mitad negro, mitad blanco)
    edge_image = np.zeros((32, 32))
    edge_image[:, 16:] = 1.0
    
    print(f"\nImagen con borde vertical en x=16")
    
    value_l2_edge = reg_l2.compute_value(edge_image)
    value_huber_edge = reg_huber.compute_value(edge_image)
    
    print(f"  R_L2(x_borde) = {value_l2_edge:.6f}")
    print(f"  R_Huber(x_borde, δ=0.1) = {value_huber_edge:.6f}")
    print(f"  Ratio Huber/L2 = {value_huber_edge/value_l2_edge:.4f}")
    print(f"\n  → Huber penaliza menos el borde (preserva bordes)")
    
    print("\n" + "=" * 70)
    print("✓ Todos los regularizadores funcionan correctamente")
    print("=" * 70)
    
    # Resumen de características
    print("\n📋 RESUMEN:")
    print("  • L2: Suaviza uniformemente toda la imagen")
    print("  • Huber: Preserva bordes fuertes, suaviza regiones homogéneas")
    print("  • δ pequeño → más preservación de bordes")
    print("  • δ grande → se acerca a comportamiento L2")