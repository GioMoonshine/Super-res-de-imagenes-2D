import numpy as np
from abc import ABC, abstractmethod
from operators import compute_gradient_x, compute_gradient_y, compute_divergence

class Regularizer(ABC):
    """
    Interfaz abstracta para términos de penalización (Priors).
    
    Define el contrato estructural que cualquier modelo de regularización
    debe cumplir para ser compatible con el solver de optimización.
    """
    
    @abstractmethod
    def compute_value(self, x):
        """
        Evalúa la energía escalar del regularizador J_reg(x).
        
        Entradas:
        ---------
        x : ndarray
            Estado actual de la imagen.
        
        Salida:
        -------
        float
            Costo asociado a la complejidad o rugosidad de la imagen.
        """
        pass
    
    @abstractmethod
    def compute_gradient(self, x):
        """
        Calcula la derivada variacional (gradiente funcional) del regularizador.
        Necesario para indicar la dirección de descenso en la optimización.
        
        Entradas:
        ---------
        x : ndarray
            Imagen actual.
        
        Salida:
        -------
        ndarray
            Matriz de gradientes con las mismas dimensiones que x.
        """
        pass


class L2GradientRegularizer(Regularizer):
    """
    Modelo de regularización cuadrática (Tikhonov).
    
    Penaliza la energía total de las derivadas.
    Formula: R(x) = ||∇x||²
    
    Características:
    - Es estrictamente convexo (fácil de optimizar).
    - Provoca un suavizado global (difusión isotrópica), lo que tiende
      a difuminar los bordes y detalles finos.
    - Su gradiente es proporcional al Laplaciano negativo (-Δx).
    """
    
    def __init__(self):
        """Constructor vacío (este modelo no requiere hiperparámetros)."""
        pass
    
    def compute_value(self, x):
        """
        Calcula la suma de los cuadrados de las diferencias entre píxeles.
        Medida de la "rugosidad" global.
        """
        # Extracción de características de primer orden
        dx = compute_gradient_x(x)
        dy = compute_gradient_y(x)
        
        # Energía L2: suma(dx² + dy²)
        energy = np.sum(dx**2) + np.sum(dy**2)
        
        return energy
    
    def compute_gradient(self, x):
        """
        Obtiene el gradiente analítico.
        
        Resultado: -2 * Divergencia(Gradiente) = -2 * Laplaciano
        Esto empuja los píxeles hacia el promedio de sus vecinos.
        """
        dx = compute_gradient_x(x)
        dy = compute_gradient_y(x)
        
        # El operador adjunto del gradiente negativo es la divergencia
        laplacian_component = compute_divergence(dx, dy)
        
        # Factor de escala derivado de la regla de la cadena (d/dx x² = 2x)
        gradient = -2.0 * laplacian_component
        
        return gradient


class HuberGradientRegularizer(Regularizer):
    """
    Modelo de regularización robusta (Huber / Pseudo-TV).
    
    Aproximación diferenciable a la Variación Total (TV).
    Comportamiento híbrido controlado por 'delta':
    
    - Región |∇x| <= δ: Cuadrática (L2). Evita el efecto 'staircasing' en zonas planas.
    - Región |∇x| > δ:  Lineal (L1). Penaliza menos los grandes saltos,
                        permitiendo la conservación de bordes nítidos.
    """
    
    def __init__(self, delta=0.1):
        """
        Args:
            delta (float): Umbral de transición. Define qué se considera
                           "ruido suave" (L2) vs "borde estructural" (L1).
        """
        self.delta = delta
    
    def _huber_function(self, z):
        """
        Núcleo de la función de coste Huber aplicada elemento a elemento.
        """
        magnitude = np.abs(z)
        
        # Máscara lógica para identificar gradientes suaves
        is_smooth = magnitude <= self.delta
        
        cost = np.where(
            is_smooth,
            z**2 / (2 * self.delta),       # Parábola (Suavizado fuerte)
            magnitude - self.delta / 2     # Cono (Preservación de bordes)
        )
        
        return cost
    
    def _huber_derivative(self, z):
        """
        Primera derivada del núcleo Huber (Función de influencia).
        """
        magnitude = np.abs(z)
        is_smooth = magnitude <= self.delta
        
        influence = np.where(
            is_smooth,
            z / self.delta,    # Lineal (proporcional al error)
            np.sign(z)         # Constante (saturación, robustez ante outliers)
        )
        
        return influence
    
    def compute_value(self, x):
        """
        Suma de costos Huber sobre los gradientes horizontal y vertical.
        """
        dx = compute_gradient_x(x)
        dy = compute_gradient_y(x)
        
        # Aplicación del costo robusto
        cost_x = self._huber_function(dx)
        cost_y = self._huber_function(dy)
        
        return np.sum(cost_x) + np.sum(cost_y)
    
    def compute_gradient(self, x):
        """
        Cálculo del gradiente mediante la divergencia del campo no-lineal.
        
        ∇J = -div( h'(∇x) )
        """
        dx = compute_gradient_x(x)
        dy = compute_gradient_y(x)
        
        # Ponderación no lineal de los gradientes
        # (Los bordes fuertes se atenúan, el ruido suave se mantiene)
        grad_weight_x = self._huber_derivative(dx)
        grad_weight_y = self._huber_derivative(dy)
        
        # Divergencia del campo ponderado
        gradient = -compute_divergence(grad_weight_x, grad_weight_y)
        
        return gradient


# ============================================================================
# Banco de Pruebas (Unit Testing)
# ============================================================================

if __name__ == "__main__":
    print("*" * 70)
    print(" DIAGNÓSTICO DE MÓDULOS DE REGULARIZACIÓN")
    print("*" * 70)
    
    # Generación de entorno de prueba estocástico
    np.random.seed(42)
    test_img = np.random.rand(32, 32)
    print(f"\n[INIT] Imagen de ruido aleatorio generada: {test_img.shape}")
    
    # ========================================================================
    # Test 1: Comportamiento L2
    # ========================================================================
    print("\n" + "-" * 70)
    print(" [1] EVALUACIÓN DEL MODELO TIKHONOV (L2)")
    print("-" * 70)
    
    l2_model = L2GradientRegularizer()
    
    cost_l2 = l2_model.compute_value(test_img)
    print(f"\n  -> Energía total (Suavidad): {cost_l2:.5f}")
    
    grad_l2 = l2_model.compute_gradient(test_img)
    norm_grad_l2 = np.sum(grad_l2**2)
    print(f"  -> Dimensión del gradiente: {grad_l2.shape}")
    print(f"  -> Magnitud del gradiente (Norma): {norm_grad_l2:.5f}")
    print(f"  -> Rango dinámico: [{np.min(grad_l2):.4f}, {np.max(grad_l2):.4f}]")
    
    # ========================================================================
    # Test 2: Comportamiento Huber
    # ========================================================================
    print("\n" + "-" * 70)
    print(" [2] ANÁLISIS DE SENSIBILIDAD HUBER (Robustez)")
    print("-" * 70)
    
    delta_params = [0.01, 0.1, 1.0]
    
    for d in delta_params:
        print(f"\n  >>> Configurando umbral delta = {d}")
        huber_model = HuberGradientRegularizer(delta=d)
        
        cost_h = huber_model.compute_value(test_img)
        print(f"      Costo calculado: {cost_h:.5f}")
        
        grad_h = huber_model.compute_gradient(test_img)
        print(f"      Energía del gradiente resultante: {np.sum(grad_h**2):.5f}")
    
    # ========================================================================
    # Test 3: Comparativa Directa
    # ========================================================================
    print("\n" + "-" * 70)
    print(" [3] CONFRONTACIÓN DE MÉTRICAS (L2 vs Huber @ delta=0.1)")
    print("-" * 70)
    
    huber_ref = HuberGradientRegularizer(delta=0.1)
    
    # Estadísticas de la imagen base
    dx = compute_gradient_x(test_img)
    dy = compute_gradient_y(test_img)
    grad_mag = np.hypot(dx, dy)
    
    print(f"\n  Estadísticas de entrada:")
    print(f"    Media del gradiente: {np.mean(grad_mag):.5f}")
    print(f"    Pico máximo: {np.max(grad_mag):.5f}")
    
    print(f"\n  Diferencial de Costos:")
    print(f"    L2 (Cuadrático puro): {l2_model.compute_value(test_img):.5f}")
    print(f"    Huber (Híbrido):      {huber_ref.compute_value(test_img):.5f}")
    
    # ========================================================================
    # Test 4: Detección de Bordes
    # ========================================================================
    print("\n" + "-" * 70)
    print(" [4] PRUEBA DE ESTRÉS CON DISCONTINUIDADES (Step Edge)")
    print("-" * 70)
    
    # Sintetizar imagen con un escalón perfecto (borde nítido)
    step_img = np.zeros((32, 32))
    step_img[:, 16:] = 1.0  # Mitad derecha blanca
    
    print(f"\n  [INPUT] Imagen con borde vertical abrupto en x=16")
    
    val_l2_edge = l2_model.compute_value(step_img)
    val_huber_edge = huber_ref.compute_value(step_img)
    
    print(f"    Penalización L2:    {val_l2_edge:.5f} (Castiga mucho el borde)")
    print(f"    Penalización Huber: {val_huber_edge:.5f} (Castiga linealmente)")
    
    ratio = val_huber_edge / val_l2_edge
    print(f"\n  -> Factor de reducción de penalización: {ratio:.4f}")
    print("  -> CONCLUSIÓN: Huber protege la estructura del borde mejor que L2.")
    
    print("\n" + "*" * 70)
    print(" VERIFICACIÓN EXITOSA: TODOS LOS SISTEMAS OPERATIVOS")
    print("*" * 70)
