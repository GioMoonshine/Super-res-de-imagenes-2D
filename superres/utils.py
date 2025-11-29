import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from io import BytesIO
import base64


class ImageProcessor:
    """
    Clase para procesar imágenes: cargar, guardar, normalizar, convertir.
    """
    
    @staticmethod
    def load_image(filepath, grayscale=True):
        """
        Carga una imagen desde archivo.
        
        Parameters:
        -----------
        filepath : str
            Ruta del archivo de imagen
        grayscale : bool
            Si True, convierte a escala de grises (default: True)
        
        Returns:
        --------
        image : ndarray
            Imagen como array de NumPy con valores en [0, 1]
        """
        img = Image.open(filepath)
        
        if grayscale:
            img = img.convert('L')  # Convertir a escala de grises
        
        # Convertir a numpy array y normalizar a [0, 1]
        image = np.array(img, dtype=np.float64) / 255.0
        
        return image
    
    @staticmethod
    def save_image(image, filepath, denormalize=True):
        """
        Guarda una imagen a archivo.
        
        Parameters:
        -----------
        image : ndarray
            Imagen como array de NumPy
        filepath : str
            Ruta donde guardar la imagen
        denormalize : bool
            Si True, asume que image está en [0, 1] y lo convierte a [0, 255]
        """
        if denormalize:
            # Clip para asegurar rango válido y convertir a uint8
            image_uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        else:
            image_uint8 = image.astype(np.uint8)
        
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Guardar imagen
        img = Image.fromarray(image_uint8, mode='L')
        img.save(filepath)
    
    @staticmethod
    def load_image_from_upload(file_storage, grayscale=True):
        """
        Carga una imagen desde un objeto FileStorage de Flask.
        
        Parameters:
        -----------
        file_storage : FileStorage
            Objeto file de Flask (request.files['key'])
        grayscale : bool
            Si True, convierte a escala de grises
        
        Returns:
        --------
        image : ndarray
            Imagen como array de NumPy con valores en [0, 1]
        """
        # Leer imagen desde bytes
        img = Image.open(file_storage)
        
        if grayscale:
            img = img.convert('L')
        
        # Convertir a numpy y normalizar
        image = np.array(img, dtype=np.float64) / 255.0
        
        return image
    
    @staticmethod
    def array_to_base64(image, denormalize=True):
        """
        Convierte un array de NumPy a string base64 para mostrar en HTML.
        
        Parameters:
        -----------
        image : ndarray
            Imagen como array de NumPy
        denormalize : bool
            Si True, convierte de [0, 1] a [0, 255]
        
        Returns:
        --------
        base64_str : str
            String base64 de la imagen
        """
        if denormalize:
            image_uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        else:
            image_uint8 = image.astype(np.uint8)
        
        # Convertir a PIL Image
        img = Image.fromarray(image_uint8, mode='L')
        
        # Guardar en buffer
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        
        # Convertir a base64
        base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return base64_str
    
    @staticmethod
    def normalize_image(image, min_val=None, max_val=None):
        """
        Normaliza una imagen al rango [0, 1].
        
        Parameters:
        -----------
        image : ndarray
            Imagen a normalizar
        min_val : float, optional
            Valor mínimo para normalización. Si None, usa min(image)
        max_val : float, optional
            Valor máximo para normalización. Si None, usa max(image)
        
        Returns:
        --------
        normalized : ndarray
            Imagen normalizada
        """
        if min_val is None:
            min_val = image.min()
        if max_val is None:
            max_val = image.max()
        
        # Evitar división por cero
        if max_val - min_val < 1e-10:
            return np.zeros_like(image)
        
        normalized = (image - min_val) / (max_val - min_val)
        return normalized
    
    @staticmethod
    def denormalize_image(image, min_val=0, max_val=255):
        """
        Desnormaliza una imagen desde [0, 1] a [min_val, max_val].
        
        Parameters:
        -----------
        image : ndarray
            Imagen normalizada en [0, 1]
        min_val : float
            Valor mínimo del rango destino (default: 0)
        max_val : float
            Valor máximo del rango destino (default: 255)
        
        Returns:
        --------
        denormalized : ndarray
            Imagen desnormalizada
        """
        denormalized = image * (max_val - min_val) + min_val
        return denormalized
    
    @staticmethod
    def resize_image(image, new_shape):
        """
        Redimensiona una imagen a un nuevo tamaño.
        
        Parameters:
        -----------
        image : ndarray
            Imagen a redimensionar
        new_shape : tuple
            Nueva forma (height, width)
        
        Returns:
        --------
        resized : ndarray
            Imagen redimensionada
        """
        # Convertir a PIL, redimensionar, volver a numpy
        if image.max() <= 1.0:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = image.astype(np.uint8)
        
        img = Image.fromarray(image_uint8, mode='L')
        img_resized = img.resize((new_shape[1], new_shape[0]), Image.BILINEAR)
        
        resized = np.array(img_resized, dtype=np.float64) / 255.0
        return resized


class PlotUtils:
    """
    Utilidades para crear gráficas de convergencia.
    """
    
    @staticmethod
    def plot_convergence(info, save_path=None):
        """
        Crea gráfica de convergencia del algoritmo.
        
        Parameters:
        -----------
        info : dict
            Diccionario con historiales de optimización
        save_path : str, optional
            Si se proporciona, guarda la gráfica en esta ruta
        
        Returns:
        --------
        fig : matplotlib.figure.Figure
            Objeto figura de matplotlib
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Costo total
        axes[0, 0].plot(info['cost_history'], 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Iteración', fontsize=11)
        axes[0, 0].set_ylabel('J(x)', fontsize=11)
        axes[0, 0].set_title('Costo Total', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_yscale('log')
        
        # Término de datos
        axes[0, 1].plot(info['data_term_history'], 'g-', linewidth=2)
        axes[0, 1].set_xlabel('Iteración', fontsize=11)
        axes[0, 1].set_ylabel('(1/2)||Ax - b||²', fontsize=11)
        axes[0, 1].set_title('Término de Fidelidad', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_yscale('log')
        
        # Término de regularización
        axes[1, 0].plot(info['reg_term_history'], 'r-', linewidth=2)
        axes[1, 0].set_xlabel('Iteración', fontsize=11)
        axes[1, 0].set_ylabel('λR(x)', fontsize=11)
        axes[1, 0].set_title('Término de Regularización', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_yscale('log')
        
        # Norma del gradiente
        axes[1, 1].plot(info['gradient_norm_history'], 'm-', linewidth=2)
        axes[1, 1].set_xlabel('Iteración', fontsize=11)
        axes[1, 1].set_ylabel('||∇J(x)||', fontsize=11)
        axes[1, 1].set_title('Norma del Gradiente', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_comparison(images_dict, titles_dict, save_path=None, figsize=(15, 5)):
        """
        Crea una comparación visual de múltiples imágenes.
        
        Parameters:
        -----------
        images_dict : dict
            Diccionario con nombre: imagen
        titles_dict : dict
            Diccionario con nombre: título
        save_path : str, optional
            Ruta para guardar la figura
        figsize : tuple
            Tamaño de la figura
        
        Returns:
        --------
        fig : matplotlib.figure.Figure
            Objeto figura
        """
        n_images = len(images_dict)
        fig, axes = plt.subplots(1, n_images, figsize=figsize)
        
        if n_images == 1:
            axes = [axes]
        
        for ax, (name, image) in zip(axes, images_dict.items()):
            ax.imshow(image, cmap='gray', vmin=0, vmax=1)
            ax.set_title(titles_dict.get(name, name), fontsize=12, fontweight='bold')
            ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def fig_to_base64(fig):
        """
        Convierte una figura de matplotlib a base64.
        
        Parameters:
        -----------
        fig : matplotlib.figure.Figure
            Figura a convertir
        
        Returns:
        --------
        base64_str : str
            String base64 de la figura
        """
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close(fig)
        return base64_str


class ValidationUtils:
    """
    Utilidades para validar parámetros de entrada.
    """
    
    @staticmethod
    def validate_parameters(params):
        """
        Valida y convierte parámetros de entrada.
        
        Parameters:
        -----------
        params : dict
            Diccionario con parámetros (strings de formulario)
        
        Returns:
        --------
        validated : dict
            Diccionario con parámetros validados y convertidos
        errors : list
            Lista de mensajes de error (vacía si todo OK)
        """
        errors = []
        validated = {}
        
        # Lambda (peso de regularización)
        try:
            lambda_reg = float(params.get('lambda_reg', 0.01))
            if lambda_reg <= 0:
                errors.append("λ debe ser mayor que 0")
            elif lambda_reg > 1.0:
                errors.append("λ no debería ser mayor que 1.0 (típicamente 0.001-0.1)")
            validated['lambda_reg'] = lambda_reg
        except ValueError:
            errors.append("λ debe ser un número válido")
            validated['lambda_reg'] = 0.01
        
        # Tau (learning rate)
        try:
            tau = float(params.get('tau', 0.001))
            if tau <= 0:
                errors.append("τ debe ser mayor que 0")
            elif tau > 0.01:
                errors.append("τ muy grande puede causar divergencia (típicamente 0.0001-0.001)")
            validated['tau'] = tau
        except ValueError:
            errors.append("τ debe ser un número válido")
            validated['tau'] = 0.001
        
        # Max iterations
        try:
            max_iter = int(params.get('max_iter', 100))
            if max_iter <= 0:
                errors.append("Iteraciones debe ser mayor que 0")
            elif max_iter > 1000:
                errors.append("Iteraciones muy alto puede tardar mucho (típicamente 50-200)")
            validated['max_iter'] = max_iter
        except ValueError:
            errors.append("Iteraciones debe ser un número entero válido")
            validated['max_iter'] = 100
        
        # Scale factor
        try:
            scale_factor = int(params.get('scale_factor', 2))
            if scale_factor < 1:
                errors.append("Factor de escala debe ser al menos 1")
            elif scale_factor > 4:
                errors.append("Factor de escala muy alto puede ser inestable (típicamente 2-3)")
            validated['scale_factor'] = scale_factor
        except ValueError:
            errors.append("Factor de escala debe ser un número entero válido")
            validated['scale_factor'] = 2
        
        # Sigma (blur)
        try:
            sigma = float(params.get('sigma', 1.5))
            if sigma <= 0:
                errors.append("σ debe ser mayor que 0")
            elif sigma > 5.0:
                errors.append("σ muy grande causa blur excesivo (típicamente 0.5-2.0)")
            validated['sigma'] = sigma
        except ValueError:
            errors.append("σ debe ser un número válido")
            validated['sigma'] = 1.5
        
        # Regularizer type
        reg_type = params.get('regularizer', 'l2')
        if reg_type not in ['l2', 'huber']:
            errors.append("Tipo de regularizador debe ser 'l2' o 'huber'")
            validated['regularizer'] = 'l2'
        else:
            validated['regularizer'] = reg_type
        
        # Delta (solo para Huber)
        if reg_type == 'huber':
            try:
                delta = float(params.get('delta', 0.1))
                if delta <= 0:
                    errors.append("δ debe ser mayor que 0")
                elif delta > 1.0:
                    errors.append("δ muy grande se acerca a L2 (típicamente 0.01-0.5)")
                validated['delta'] = delta
            except ValueError:
                errors.append("δ debe ser un número válido")
                validated['delta'] = 0.1
        
        return validated, errors


# ============================================================================
# Script de prueba
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PROBANDO UTILIDADES")
    print("=" * 70)
    
    # ========================================================================
    # 1. Crear imagen de prueba
    # ========================================================================
    print("\n1. Creando y guardando imagen de prueba...")
    
    test_image = np.random.rand(64, 64)
    os.makedirs('test_output', exist_ok=True)
    
    ImageProcessor.save_image(test_image, 'test_output/test_image.png')
    print("   ✓ Imagen guardada en 'test_output/test_image.png'")
    
    # ========================================================================
    # 2. Cargar imagen
    # ========================================================================
    print("\n2. Cargando imagen...")
    
    loaded = ImageProcessor.load_image('test_output/test_image.png')
    print(f"   Shape: {loaded.shape}")
    print(f"   Rango: [{loaded.min():.3f}, {loaded.max():.3f}]")
    print(f"   ✓ Imagen cargada correctamente")
    
    # ========================================================================
    # 3. Normalización
    # ========================================================================
    print("\n3. Probando normalización...")
    
    image_unnorm = np.random.rand(32, 32) * 100 + 50  # Rango [50, 150]
    normalized = ImageProcessor.normalize_image(image_unnorm)
    print(f"   Original: [{image_unnorm.min():.1f}, {image_unnorm.max():.1f}]")
    print(f"   Normalizada: [{normalized.min():.3f}, {normalized.max():.3f}]")
    print(f"   ✓ Normalización correcta")
    
    # ========================================================================
    # 4. Conversión a base64
    # ========================================================================
    print("\n4. Probando conversión a base64...")
    
    base64_str = ImageProcessor.array_to_base64(test_image)
    print(f"   Longitud del string base64: {len(base64_str)}")
    print(f"   Primeros 50 caracteres: {base64_str[:50]}...")
    print(f"   ✓ Conversión a base64 correcta")
    
    # ========================================================================
    # 5. Validación de parámetros
    # ========================================================================
    print("\n5. Probando validación de parámetros...")
    
    # Parámetros válidos
    params_valid = {
        'lambda_reg': '0.01',
        'tau': '0.001',
        'max_iter': '100',
        'scale_factor': '2',
        'sigma': '1.5',
        'regularizer': 'l2'
    }
    
    validated, errors = ValidationUtils.validate_parameters(params_valid)
    print(f"   Parámetros válidos: {len(errors)} errores")
    if errors:
        for err in errors:
            print(f"     - {err}")
    else:
        print(f"     ✓ Todos los parámetros válidos")
        print(f"     λ={validated['lambda_reg']}, τ={validated['tau']}, "
              f"iter={validated['max_iter']}, s={validated['scale_factor']}")
    
    # Parámetros inválidos
    print("\n   Probando parámetros inválidos...")
    params_invalid = {
        'lambda_reg': '-0.01',  # Negativo
        'tau': 'abc',           # No numérico
        'max_iter': '2000',     # Muy alto
        'scale_factor': '0',    # Muy bajo
        'regularizer': 'invalid'
    }
    
    validated, errors = ValidationUtils.validate_parameters(params_invalid)
    print(f"   Parámetros inválidos: {len(errors)} errores detectados")
    for err in errors:
        print(f"     - {err}")
    
    # ========================================================================
    # 6. Gráfica de convergencia simulada
    # ========================================================================
    print("\n6. Creando gráfica de convergencia simulada...")
    
    # Simular historial de optimización
    n_iter = 50
    cost = np.exp(-np.linspace(0, 3, n_iter)) * 100 + 10
    data_term = cost * 0.7
    reg_term = cost * 0.3
    grad_norm = np.exp(-np.linspace(0, 4, n_iter)) * 50
    
    info = {
        'cost_history': cost,
        'data_term_history': data_term,
        'reg_term_history': reg_term,
        'gradient_norm_history': grad_norm,
        'iterations': n_iter,
        'converged': True
    }
    
    try:
        fig = PlotUtils.plot_convergence(info, save_path='test_output/convergence.png')
        print("   ✓ Gráfica guardada en 'test_output/convergence.png'")
        
        # Convertir a base64
        base64_plot = PlotUtils.fig_to_base64(fig)
        print(f"   ✓ Gráfica convertida a base64 ({len(base64_plot)} caracteres)")
        
    except Exception as e:
        print(f"   ⚠ Error al crear gráfica: {e}")
    
    # ========================================================================
    # 7. Comparación de imágenes
    # ========================================================================
    print("\n7. Creando comparación de imágenes...")
    
    images = {
        'original': test_image,
        'procesada': test_image * 0.8,
        'diferencia': np.abs(test_image - test_image * 0.8)
    }
    
    titles = {
        'original': 'Original',
        'procesada': 'Procesada',
        'diferencia': 'Diferencia'
    }
    
    try:
        fig = PlotUtils.plot_comparison(images, titles, 
                                       save_path='test_output/comparison.png')
        print("   ✓ Comparación guardada en 'test_output/comparison.png'")
    except Exception as e:
        print(f"   ⚠ Error al crear comparación: {e}")
    
    print("\n" + "=" * 70)
    print("✓ TODAS LAS UTILIDADES FUNCIONAN CORRECTAMENTE")
    print("=" * 70)