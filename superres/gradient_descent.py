import numpy as np
from operators import DegradationOperator
from regularizers import Regularizer


class SuperResolutionSolver:
    """
    Solver de super-resolución mediante descenso de gradiente.
    
    Resuelve el problema:
        min_x J(x) = (1/2)||Ax - b||² + λR(x)
    
    mediante iteraciones:
        x^(k+1) = x^(k) - τ[A⊤(Ax^(k) - b) + λ∇R(x^(k))]
    
    donde:
    - A: operador de degradación
    - b: imagen LR observada
    - R(x): término de regularización
    - λ: peso de regularización
    - τ: learning rate (paso de descenso)
    """
    
    def __init__(self, degradation_op, regularizer, lambda_reg=0.01, 
                 tau=0.001, max_iter=100, tolerance=1e-6, verbose=True):
        """
        Inicializa el solver de super-resolución.
        
        Parameters:
        -----------
        degradation_op : DegradationOperator
            Operador A con métodos apply() y apply_adjoint()
        regularizer : Regularizer
            Regularizador con métodos compute_value() y compute_gradient()
        lambda_reg : float
            Peso del término de regularización λ (default: 0.01)
        tau : float
            Paso de descenso (learning rate) τ (default: 0.001)
        max_iter : int
            Número máximo de iteraciones (default: 100)
        tolerance : float
            Tolerancia para convergencia (default: 1e-6)
        verbose : bool
            Si True, imprime información durante la optimización
        """
        self.A = degradation_op
        self.regularizer = regularizer
        self.lambda_reg = lambda_reg
        self.tau = tau
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.verbose = verbose
        
        # Historial de la optimización
        self.cost_history = []
        self.data_term_history = []
        self.reg_term_history = []
        self.gradient_norm_history = []
        
    def compute_cost(self, x, b):
        """
        Calcula el costo total J(x) = (1/2)||Ax - b||² + λR(x)
        
        Parameters:
        -----------
        x : ndarray
            Imagen HR actual
        b : ndarray
            Imagen LR observada
        
        Returns:
        --------
        total_cost : float
            Valor total del costo
        data_term : float
            Término de fidelidad a los datos
        reg_term : float
            Término de regularización
        """
        # Término de datos: (1/2)||Ax - b||²
        residual = self.A.apply(x) - b
        data_term = 0.5 * np.sum(residual**2)
        
        # Término de regularización: λR(x)
        reg_value = self.regularizer.compute_value(x)
        reg_term = self.lambda_reg * reg_value
        
        # Costo total
        total_cost = data_term + reg_term
        
        return total_cost, data_term, reg_term
    
    def compute_gradient(self, x, b):
        """
        Calcula el gradiente de J(x):
            ∇J(x) = A⊤(Ax - b) + λ∇R(x)
        
        Parameters:
        -----------
        x : ndarray
            Imagen HR actual
        b : ndarray
            Imagen LR observada
        
        Returns:
        --------
        gradient : ndarray
            Gradiente completo ∇J(x)
        """
        # Gradiente del término de datos: A⊤(Ax - b)
        residual = self.A.apply(x) - b
        data_gradient = self.A.apply_adjoint(residual)
        
        # Gradiente del término de regularización: λ∇R(x)
        reg_gradient = self.lambda_reg * self.regularizer.compute_gradient(x)
        
        # Gradiente total
        gradient = data_gradient + reg_gradient
        
        return gradient
    
    def solve(self, b, x_init=None):
        """
        Resuelve el problema de super-resolución mediante descenso de gradiente.
        
        Parameters:
        -----------
        b : ndarray
            Imagen LR observada (H_LR, W_LR)
        x_init : ndarray, optional
            Inicialización para x. Si es None, usa upsampling de b
        
        Returns:
        --------
        x : ndarray
            Imagen HR reconstruida (H_HR, W_HR)
        info : dict
            Diccionario con información de la optimización:
            - 'cost_history': historial de costos
            - 'data_term_history': historial del término de datos
            - 'reg_term_history': historial de regularización
            - 'gradient_norm_history': historial de normas del gradiente
            - 'iterations': número de iteraciones realizadas
            - 'converged': True si convergió
        """
        # Reiniciar historiales
        self.cost_history = []
        self.data_term_history = []
        self.reg_term_history = []
        self.gradient_norm_history = []
        
        # Inicialización
        if x_init is None:
            x = self.A.upsample_image(b)
        else:
            x = x_init.copy()
        
        if self.verbose:
            print("=" * 70)
            print("INICIANDO DESCENSO DE GRADIENTE")
            print("=" * 70)
            print(f"Imagen LR: {b.shape}")
            print(f"Imagen HR inicial: {x.shape}")
            print(f"λ = {self.lambda_reg}, τ = {self.tau}, max_iter = {self.max_iter}")
            print(f"Regularizador: {type(self.regularizer).__name__}")
            print("=" * 70)
        
        # Iteraciones de descenso de gradiente
        converged = False
        
        for k in range(self.max_iter):
            # Calcular costo actual
            cost, data_term, reg_term = self.compute_cost(x, b)
            
            # Calcular gradiente
            gradient = self.compute_gradient(x, b)
            grad_norm = np.linalg.norm(gradient)
            
            # Guardar en historial
            self.cost_history.append(cost)
            self.data_term_history.append(data_term)
            self.reg_term_history.append(reg_term)
            self.gradient_norm_history.append(grad_norm)
            
            # Imprimir progreso
            if self.verbose and (k % 10 == 0 or k == self.max_iter - 1):
                print(f"Iter {k:4d}: J(x) = {cost:.6e} | "
                      f"Data = {data_term:.6e} | Reg = {reg_term:.6e} | "
                      f"||∇J|| = {grad_norm:.6e}")
            
            # Criterio de convergencia: ||∇J|| < tolerance
            if grad_norm < self.tolerance:
                converged = True
                if self.verbose:
                    print(f"\n✓ Convergencia alcanzada en iteración {k}")
                    print(f"  ||∇J|| = {grad_norm:.6e} < {self.tolerance}")
                break
            
            # Actualización de descenso de gradiente
            # x^(k+1) = x^(k) - τ·∇J(x^(k))
            x = x - self.tau * gradient
            
            # Opcional: proyectar x al rango [0, 1] si trabajamos con imágenes normalizadas
            # x = np.clip(x, 0, 1)
        
        if self.verbose:
            print("=" * 70)
            if converged:
                print(f"✓ OPTIMIZACIÓN COMPLETADA (convergencia en {k+1} iteraciones)")
            else:
                print(f"⚠ OPTIMIZACIÓN COMPLETADA (máximo de iteraciones alcanzado)")
            print(f"  Costo final: J(x) = {self.cost_history[-1]:.6e}")
            print(f"  ||∇J|| final = {self.gradient_norm_history[-1]:.6e}")
            print("=" * 70)
        
        # Información de retorno
        info = {
            'cost_history': np.array(self.cost_history),
            'data_term_history': np.array(self.data_term_history),
            'reg_term_history': np.array(self.reg_term_history),
            'gradient_norm_history': np.array(self.gradient_norm_history),
            'iterations': len(self.cost_history),
            'converged': converged
        }
        
        return x, info


# ============================================================================
# Script de prueba completo
# ============================================================================

if __name__ == "__main__":
    from regularizers import L2GradientRegularizer, HuberGradientRegularizer
    import matplotlib.pyplot as plt
    
    print("\n" + "=" * 70)
    print("PRUEBA COMPLETA DEL SOLVER DE SUPER-RESOLUCIÓN")
    print("=" * 70)
    
    # ========================================================================
    # 1. Crear imagen HR sintética
    # ========================================================================
    print("\n1. Creando imagen HR sintética...")
    
    # Crear imagen con estructura (cuadrados)
    hr_size = 64
    x_true = np.zeros((hr_size, hr_size))
    
    # Agregar algunos cuadrados
    x_true[10:30, 10:30] = 0.8
    x_true[35:55, 35:55] = 0.6
    x_true[15:25, 40:50] = 1.0
    
    # Agregar ruido suave
    np.random.seed(42)
    x_true += 0.05 * np.random.randn(hr_size, hr_size)
    x_true = np.clip(x_true, 0, 1)
    
    print(f"   Imagen HR verdadera: {x_true.shape}")
    print(f"   Rango de valores: [{x_true.min():.3f}, {x_true.max():.3f}]")
    
    # ========================================================================
    # 2. Crear imagen LR degradada
    # ========================================================================
    print("\n2. Degradando imagen para obtener LR...")
    
    scale_factor = 2
    A = DegradationOperator(scale_factor=scale_factor, sigma=1.5)
    
    b = A.apply(x_true)
    print(f"   Imagen LR observada: {b.shape}")
    print(f"   Factor de reducción: {scale_factor}x")
    
    # ========================================================================
    # 3. Super-resolución con regularización L2
    # ========================================================================
    print("\n3. Resolviendo con regularización L2...")
    print("-" * 70)
    
    reg_l2 = L2GradientRegularizer()
    solver_l2 = SuperResolutionSolver(
        degradation_op=A,
        regularizer=reg_l2,
        lambda_reg=0.01,
        tau=0.0005,
        max_iter=100,
        verbose=True
    )
    
    x_l2, info_l2 = solver_l2.solve(b)
    
    # ========================================================================
    # 4. Super-resolución con regularización Huber
    # ========================================================================
    print("\n4. Resolviendo con regularización Huber...")
    print("-" * 70)
    
    reg_huber = HuberGradientRegularizer(delta=0.1)
    solver_huber = SuperResolutionSolver(
        degradation_op=A,
        regularizer=reg_huber,
        lambda_reg=0.01,
        tau=0.0005,
        max_iter=100,
        verbose=True
    )
    
    x_huber, info_huber = solver_huber.solve(b)
    
    # ========================================================================
    # 5. Comparación de resultados
    # ========================================================================
    print("\n5. Comparación de resultados:")
    print("-" * 70)
    
    # Errores respecto a la verdad
    error_l2 = np.linalg.norm(x_l2 - x_true) / np.linalg.norm(x_true)
    error_huber = np.linalg.norm(x_huber - x_true) / np.linalg.norm(x_true)
    
    print(f"Error relativo L2:    {error_l2:.6f}")
    print(f"Error relativo Huber: {error_huber:.6f}")
    
    # Costo final
    print(f"\nCosto final L2:    {info_l2['cost_history'][-1]:.6e}")
    print(f"Costo final Huber: {info_huber['cost_history'][-1]:.6e}")
    
    # Iteraciones
    print(f"\nIteraciones L2:    {info_l2['iterations']}")
    print(f"Iteraciones Huber: {info_huber['iterations']}")
    
    # ========================================================================
    # 6. Visualización (opcional, requiere matplotlib)
    # ========================================================================
    print("\n6. Generando gráficas de convergencia...")
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Costo total
        axes[0, 0].semilogy(info_l2['cost_history'], 'b-', label='L2', linewidth=2)
        axes[0, 0].semilogy(info_huber['cost_history'], 'r-', label='Huber', linewidth=2)
        axes[0, 0].set_xlabel('Iteración')
        axes[0, 0].set_ylabel('J(x)')
        axes[0, 0].set_title('Costo Total')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Término de datos
        axes[0, 1].semilogy(info_l2['data_term_history'], 'b-', label='L2', linewidth=2)
        axes[0, 1].semilogy(info_huber['data_term_history'], 'r-', label='Huber', linewidth=2)
        axes[0, 1].set_xlabel('Iteración')
        axes[0, 1].set_ylabel('(1/2)||Ax - b||²')
        axes[0, 1].set_title('Término de Datos')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Término de regularización
        axes[1, 0].semilogy(info_l2['reg_term_history'], 'b-', label='L2', linewidth=2)
        axes[1, 0].semilogy(info_huber['reg_term_history'], 'r-', label='Huber', linewidth=2)
        axes[1, 0].set_xlabel('Iteración')
        axes[1, 0].set_ylabel('λR(x)')
        axes[1, 0].set_title('Término de Regularización')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Norma del gradiente
        axes[1, 1].semilogy(info_l2['gradient_norm_history'], 'b-', label='L2', linewidth=2)
        axes[1, 1].semilogy(info_huber['gradient_norm_history'], 'r-', label='Huber', linewidth=2)
        axes[1, 1].set_xlabel('Iteración')
        axes[1, 1].set_ylabel('||∇J(x)||')
        axes[1, 1].set_title('Norma del Gradiente')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('convergence_comparison.png', dpi=150, bbox_inches='tight')
        print("   ✓ Gráfica guardada como 'convergence_comparison.png'")
        
        # Visualizar imágenes
        fig2, axes2 = plt.subplots(2, 3, figsize=(12, 8))
        
        axes2[0, 0].imshow(x_true, cmap='gray', vmin=0, vmax=1)
        axes2[0, 0].set_title('Ground Truth (HR)')
        axes2[0, 0].axis('off')
        
        axes2[0, 1].imshow(b, cmap='gray', vmin=0, vmax=1)
        axes2[0, 1].set_title('Observada (LR)')
        axes2[0, 1].axis('off')
        
        axes2[0, 2].imshow(A.upsample_image(b), cmap='gray', vmin=0, vmax=1)
        axes2[0, 2].set_title('Upsampling Simple')
        axes2[0, 2].axis('off')
        
        axes2[1, 0].imshow(x_l2, cmap='gray', vmin=0, vmax=1)
        axes2[1, 0].set_title(f'Reconstrucción L2\nError: {error_l2:.4f}')
        axes2[1, 0].axis('off')
        
        axes2[1, 1].imshow(x_huber, cmap='gray', vmin=0, vmax=1)
        axes2[1, 1].set_title(f'Reconstrucción Huber\nError: {error_huber:.4f}')
        axes2[1, 1].axis('off')
        
        # Diferencia
        diff = np.abs(x_l2 - x_huber)
        axes2[1, 2].imshow(diff, cmap='hot', vmin=0, vmax=0.1)
        axes2[1, 2].set_title('|L2 - Huber|')
        axes2[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig('reconstruction_comparison.png', dpi=150, bbox_inches='tight')
        print("   ✓ Gráfica guardada como 'reconstruction_comparison.png'")
        
    except ImportError:
        print("   ⚠ matplotlib no disponible, saltando visualización")
    
    print("\n" + "=" * 70)
    print("✓ PRUEBA COMPLETA EXITOSA")
    print("=" * 70)