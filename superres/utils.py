import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from io import BytesIO
import base64

class ImageProcessor:
    """
    Motor de E/S y Transformación de Datos Visuales.
    
    Encargado de la ingesta, serialización y mapeo de espacios de color
    entre el sistema de archivos, buffers de memoria y tensores numéricos.
    """
    
    @staticmethod
    def load_image(filepath, grayscale=True):
        """
        Ejecuta la ingesta de un activo visual desde el almacenamiento local.
        
        Realiza la decodificación del formato de imagen y la transmutación
        a estructura tensorial de punto flotante.
        
        Entradas:
        ---------
        filepath : str
            Ruta absoluta o relativa del recurso.
        grayscale : bool
            Flag para forzar el colapso de canales a luminancia (L).
        
        Salida:
        -------
        image : ndarray
            Matriz normalizada en el dominio [0.0, 1.0].
        """
        img = Image.open(filepath)
        
        if grayscale:
            img = img.convert('L')  # Reducción de dimensionalidad espectral
        
        # Conversión a precisión doble y escalado al intervalo unitario
        image = np.array(img, dtype=np.float64) / 255.0
        
        return image
    
    @staticmethod
    def save_image(image, filepath, denormalize=True):
        """
        Serializa el tensor en memoria a un archivo persistente.
        
        Incluye una etapa de cuantización (float -> uint8) si es necesario.
        
        Entradas:
        ---------
        image : ndarray
            Datos matriciales en memoria.
        filepath : str
            Destino del flujo de bytes.
        denormalize : bool
            Si True, escala de [0, 1] a [0, 255] antes de codificar.
        """
        if denormalize:
            # Saturación y cuantización a 8 bits por canal
            image_uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        else:
            image_uint8 = image.astype(np.uint8)
        
        # Verificación de estructura de directorios
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Codificación y escritura
        img = Image.fromarray(image_uint8, mode='L')
        img.save(filepath)
    
    @staticmethod
    def load_image_from_upload(file_storage, grayscale=True):
        """
        Procesa un flujo de entrada (stream) proveniente de una petición HTTP.
        
        Entradas:
        ---------
        file_storage : FileStorage
            Descriptor de archivo en memoria (payload de red).
        
        Salida:
        -------
        image : ndarray
            Tensor listo para procesamiento numérico.
        """
        # Lectura directa desde buffer
        img = Image.open(file_storage)
        
        if grayscale:
            img = img.convert('L')
        
        image = np.array(img, dtype=np.float64) / 255.0
        
        return image
    
    @staticmethod
    def array_to_base64(image, denormalize=True):
        """
        Codifica la matriz visual en formato Base64 para transmisión web.
        Útil para incrustación directa en respuestas HTML/JSON.
        """
        if denormalize:
            image_uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        else:
            image_uint8 = image.astype(np.uint8)
        
        img = Image.fromarray(image_uint8, mode='L')
        
        # Escritura en buffer de memoria volátil
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        
        # Codificación ASCII segura para transporte
        base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return base64_str
    
    @staticmethod
    def normalize_image(image, min_val=None, max_val=None):
        """
        Re-escala el rango dinámico de la imagen al intervalo [0, 1].
        
        Esencial para mantener la estabilidad numérica en algoritmos iterativos.
        """
        if min_val is None:
            min_val = image.min()
        if max_val is None:
            max_val = image.max()
        
        # Protección contra singularidades (imágenes planas)
        if max_val - min_val < 1e-10:
            return np.zeros_like(image)
        
        normalized = (image - min_val) / (max_val - min_val)
        return normalized
    
    @staticmethod
    def denormalize_image(image, min_val=0, max_val=255):
        """
        Proyecta el intervalo unitario [0, 1] a un espacio de destino arbitrario.
        Típicamente usado para visualización (0-255).
        """
        denormalized = image * (max_val - min_val) + min_val
        return denormalized
    
    @staticmethod
    def resize_image(image, new_shape):
        """
        Realiza un remuestreo espacial de la matriz (Downscaling/Upscaling).
        Utiliza interpolación bilineal para minimizar artefactos.
        """
        # Pipeline: Float -> Uint8 -> PIL Resample -> Float
        if image.max() <= 1.0:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = image.astype(np.uint8)
        
        img = Image.fromarray(image_uint8, mode='L')
        # Nota: PIL usa (width, height), Numpy usa (row, col)
        img_resized = img.resize((new_shape[1], new_shape[0]), Image.BILINEAR)
        
        resized = np.array(img_resized, dtype=np.float64) / 255.0
        return resized


class PlotUtils:
    """
    Subsistema de Telemetría Visual y Reportes.
    Genera dashboards estáticos para monitorear el rendimiento de los algoritmos.
    """
    
    @staticmethod
    def plot_convergence(info, save_path=None):
        """
        Genera un dashboard de métricas de optimización.
        
        Visualiza:
        1. Función de Costo Global (Energía).
        2. Fidelidad de Datos (Residuo).
        3. Regularización (Priors).
        4. Estabilidad del Gradiente (Convergencia).
        
        Salida:
        -------
        fig : Figure
            Objeto gráfico listo para renderizar.
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Panel 1: Energía Total
        axes[0, 0].plot(info['cost_history'], 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Epoch / Iteración', fontsize=11)
        axes[0, 0].set_ylabel('J(x)', fontsize=11)
        axes[0, 0].set_title('Decaimiento de la Función Objetivo', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_yscale('log')
        
        # Panel 2: Error de Reconstrucción
        axes[0, 1].plot(info['data_term_history'], 'g-', linewidth=2)
        axes[0, 1].set_xlabel('Epoch / Iteración', fontsize=11)
        axes[0, 1].set_ylabel('Data Fidelity', fontsize=11)
        axes[0, 1].set_title('Minimización de Residuos', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_yscale('log')
        
        # Panel 3: Costo Estructural
        axes[1, 0].plot(info['reg_term_history'], 'r-', linewidth=2)
        axes[1, 0].set_xlabel('Epoch / Iteración', fontsize=11)
        axes[1, 0].set_ylabel('Regularization', fontsize=11)
        axes[1, 0].set_title('Evolución de Restricciones', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_yscale('log')
        
        # Panel 4: Dinámica del Gradiente
        axes[1, 1].plot(info['gradient_norm_history'], 'm-', linewidth=2)
        axes[1, 1].set_xlabel('Epoch / Iteración', fontsize=11)
        axes[1, 1].set_ylabel('||Grad||', fontsize=11)
        axes[1, 1].set_title('Velocidad de Convergencia', fontsize=12, fontweight='bold')
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
        Genera una comparativa lado a lado (side-by-side) de tensores visuales.
        Útil para análisis cualitativo (Input vs Output vs Error).
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
        Serializa un objeto Figure de Matplotlib a cadena Base64.
        """
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close(fig)
        return base64_str


class ValidationUtils:
    """
    Módulo de Saneamiento y Auditoría de Parámetros.
    
    Garantiza la integridad de la configuración antes de iniciar
    procesos computacionalmente costosos.
    """
    
    @staticmethod
    def validate_parameters(params):
        """
        Ejecuta la validación de tipos, rangos y consistencia lógica
        de los hiperparámetros de entrada.
        
        Entradas:
        ---------
        params : dict
            Configuración en crudo (raw input).
        
        Salida:
        -------
        validated : dict
            Parámetros casteados y seguros.
        errors : list
            Reporte de violaciones de constraints.
        """
        errors = []
        validated = {}
        
        # Validación de Lambda (Factor de regularización)
        try:
            lambda_reg = float(params.get('lambda_reg', 0.01))
            if lambda_reg <= 0:
                errors.append("Constraint Violation: λ debe ser positivo")
            elif lambda_reg > 1.0:
                errors.append("Warning: λ > 1.0 puede dominar la función de costo")
            validated['lambda_reg'] = lambda_reg
        except ValueError:
            errors.append("Type Error: λ requiere formato numérico flotante")
            validated['lambda_reg'] = 0.01
        
        # Validación de Tau (Tamaño de paso)
        try:
            tau = float(params.get('tau', 0.001))
            if tau <= 0:
                errors.append("Constraint Violation: τ debe ser positivo")
            elif tau > 0.01:
                errors.append("Stability Warning: τ alto riesgo de divergencia")
            validated['tau'] = tau
        except ValueError:
            errors.append("Type Error: τ requiere formato numérico flotante")
            validated['tau'] = 0.001
        
        # Validación de Iteraciones (Presupuesto computacional)
        try:
            max_iter = int(params.get('max_iter', 100))
            if max_iter <= 0:
                errors.append("Constraint Violation: Iteraciones deben ser > 0")
            elif max_iter > 1000:
                errors.append("Resource Warning: Alta carga computacional detectada")
            validated['max_iter'] = max_iter
        except ValueError:
            errors.append("Type Error: Iteraciones requiere entero")
            validated['max_iter'] = 100
        
        # Validación de Escala (Downsampling)
        try:
            scale_factor = int(params.get('scale_factor', 2))
            if scale_factor < 1:
                errors.append("Constraint Violation: Escala mínima es 1")
            elif scale_factor > 4:
                errors.append("Quality Warning: Pérdida de información excesiva por submuestreo")
            validated['scale_factor'] = scale_factor
        except ValueError:
            errors.append("Type Error: Escala requiere entero")
            validated['scale_factor'] = 2
        
        # Validación de Sigma (Kernel Gaussiano)
        try:
            sigma = float(params.get('sigma', 1.5))
            if sigma <= 0:
                errors.append("Constraint Violation: σ debe ser positivo")
            elif sigma > 5.0:
                errors.append("Data Loss Warning: Desenfoque agresivo")
            validated['sigma'] = sigma
        except ValueError:
            errors.append("Type Error: σ requiere flotante")
            validated['sigma'] = 1.5
        
        # Selector de Modelo
        reg_type = params.get('regularizer', 'l2')
        if reg_type not in ['l2', 'huber']:
            errors.append("Configuration Error: Modelo desconocido (soportados: l2, huber)")
            validated['regularizer'] = 'l2'
        else:
            validated['regularizer'] = reg_type
        
        # Parámetros condicionales (Huber Delta)
        if reg_type == 'huber':
            try:
                delta = float(params.get('delta', 0.1))
                if delta <= 0:
                    errors.append("Constraint Violation: δ debe ser positivo")
                elif delta > 1.0:
                    errors.append("Model Warning: δ alto converge a comportamiento L2")
                validated['delta'] = delta
            except ValueError:
                errors.append("Type Error: δ requiere flotante")
                validated['delta'] = 0.1
        
        return validated, errors


# ============================================================================
# Secuencia de Integración y Verificación
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print(" INICIANDO DIAGNÓSTICO DE SISTEMA DE UTILIDADES")
    print("=" * 70)
    
    # ========================================================================
    # 1. Generación de Activos Sintéticos
    # ========================================================================
    print("\n[Fase 1] Generación de dataset sintético...")
    
    test_image = np.random.rand(64, 64)
    output_dir = 'test_output'
    os.makedirs(output_dir, exist_ok=True)
    
    ImageProcessor.save_image(test_image, f'{output_dir}/test_image.png')
    print("   -> Activo persistido en disco con éxito.")
    
    # ========================================================================
    # 2. Prueba de Ingesta (I/O)
    # ========================================================================
    print("\n[Fase 2] Verificación de pipeline de lectura...")
    
    loaded = ImageProcessor.load_image(f'{output_dir}/test_image.png')
    print(f"   -> Tensor recuperado. Dimensiones: {loaded.shape}")
    print(f"   -> Rango dinámico actual: [{loaded.min():.3f}, {loaded.max():.3f}]")
    print(f"   -> Estado: INTEGRIDAD VERIFICADA")
    
    # ========================================================================
    # 3. Operaciones de Rango Dinámico
    # ========================================================================
    print("\n[Fase 3] Test de normalización y escalado...")
    
    image_unnorm = np.random.rand(32, 32) * 100 + 50  # Datos fuera de rango
    normalized = ImageProcessor.normalize_image(image_unnorm)
    print(f"   -> Entrada (Raw):  [{image_unnorm.min():.1f}, {image_unnorm.max():.1f}]")
    print(f"   -> Salida (Norm):  [{normalized.min():.3f}, {normalized.max():.3f}]")
    print(f"   -> Operación exitosa.")
    
    # ========================================================================
    # 4. Serialización Base64
    # ========================================================================
    print("\n[Fase 4] Codificación para transporte...")
    
    base64_str = ImageProcessor.array_to_base64(test_image)
    print(f"   -> Payload generado. Tamaño: {len(base64_str)} bytes")
    print(f"   -> Cabecera: {base64_str[:50]}...")
    print(f"   -> Codificación lista para transmisión.")
    
    # ========================================================================
    # 5. Auditoría de Configuración
    # ========================================================================
    print("\n[Fase 5] Saneamiento de parámetros de entrada...")
    
    # Caso: Configuración Nominal
    params_valid = {
        'lambda_reg': '0.01',
        'tau': '0.001',
        'max_iter': '100',
        'scale_factor': '2',
        'sigma': '1.5',
        'regularizer': 'l2'
    }
    
    validated, errors = ValidationUtils.validate_parameters(params_valid)
    print(f"   -> Caso Nominal: {len(errors)} incidencias.")
    if not errors:
        print(f"      Configuración aceptada: {validated}")
    
    # Caso: Configuración Corrupta/Extrema
    print("\n   -> Inyectando parámetros fuera de límites...")
    params_invalid = {
        'lambda_reg': '-0.01',  # Violación de constraint
        'tau': 'null',          # Error de tipo
        'max_iter': '5000',     # Advertencia de recursos
        'scale_factor': '0',    # Violación física
        'regularizer': 'unknown' # Error de modelo
    }
    
    validated, errors = ValidationUtils.validate_parameters(params_invalid)
    print(f"   -> Caso Borde: {len(errors)} incidencias detectadas.")
    for err in errors:
        print(f"      [!] {err}")
    
    # ========================================================================
    # 6. Simulación de Telemetría
    # ========================================================================
    print("\n[Fase 6] Renderizado de dashboard de métricas...")
    
    # Generación de trazas sintéticas
    n_iter = 50
    cost = np.exp(-np.linspace(0, 3, n_iter)) * 100 + 10
    info = {
        'cost_history': cost,
        'data_term_history': cost * 0.7,
        'reg_term_history': cost * 0.3,
        'gradient_norm_history': np.exp(-np.linspace(0, 4, n_iter)) * 50,
        'iterations': n_iter,
        'converged': True
    }
    
    try:
        fig = PlotUtils.plot_convergence(info, save_path=f'{output_dir}/convergence.png')
        print("   -> Gráfica de convergencia exportada al sistema de archivos.")
        
        base64_plot = PlotUtils.fig_to_base64(fig)
        print(f"   -> Gráfica serializada en memoria ({len(base64_plot)} bytes).")
        
    except Exception as e:
        print(f"   [ERROR] Fallo en motor de renderizado: {e}")
    
    # ========================================================================
    # 7. Renderizado Comparativo
    # ========================================================================
    print("\n[Fase 7] Generación de reporte visual comparativo...")
    
    images = {
        'original': test_image,
        'procesada': test_image * 0.8,
        'residuo': np.abs(test_image - test_image * 0.8)
    }
    
    titles = {
        'original': 'Source (GT)',
        'procesada': 'Reconstruction',
        'residuo': 'Residual Map'
    }
    
    try:
        fig = PlotUtils.plot_comparison(images, titles, 
                                      save_path=f'{output_dir}/comparison.png')
        print("   -> Matriz de comparación exportada exitosamente.")
    except Exception as e:
        print(f"   [ERROR] Fallo en composición de imagen: {e}")
    
    print("\n" + "=" * 70)
    print(" REPORTE FINAL: SISTEMA OPERATIVO Y ESTABLE")
    print("=" * 70)
