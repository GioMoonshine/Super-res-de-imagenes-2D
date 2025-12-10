import os
from flask import Flask, render_template, request, redirect, url_for, flash
import numpy as np

# Importación de tus módulos personalizados
# Nota: Asegúrate de que gradient_descent.py contenga la clase SuperResolutionSolver
# y que no haya conflictos de nombres con operators.py y regularizers.py
from superres import (
    DegradationOperator,
    L2GradientRegularizer,
    HuberGradientRegularizer,
    SuperResolutionSolver,
    ImageProcessor,
    PlotUtils,
    ValidationUtils
)

app = Flask(__name__)
app.secret_key = 'super_resolucion_secret_key'  # Necesario para mensajes flash

# Extensiones permitidas para seguridad
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def index():
    """
    Ruta principal: Muestra el formulario de configuración.
    [cite: 121]
    """
    return render_template('index.html')

@app.route('/superresolution', methods=['POST'])
def superresolution():
    """
    Endpoint POST obligatorio:
    1. Recibe imagen y parámetros.
    2. Valida datos.
    3. Ejecuta descenso de gradiente.
    4. Renderiza resultados.
    [cite: 108, 110, 111]
    """
    # 1. Validación de Archivo
    if 'image' not in request.files:
        flash('No se seleccionó ninguna imagen')
        return redirect(request.url)
    
    file = request.files['image']
    
    if file.filename == '':
        flash('El nombre del archivo está vacío')
        return redirect(request.url)

    if not (file and allowed_file(file.filename)):
        flash('Formato de archivo no permitido (use PNG, JPG)')
        return redirect(request.url)

    # 2. Obtención y Validación de Parámetros usando tu utils.py
    # Convertimos request.form (ImmutableMultiDict) a dict normal
    raw_params = request.form.to_dict()
    
    # Usamos tu clase de validación
    params, errors = ValidationUtils.validate_parameters(raw_params)
    
    if errors:
        for error in errors:
            flash(error)
        return redirect(url_for('index'))

    try:
        # 3. Procesamiento de Imagen (Ingesta)
        # Cargamos directamente desde memoria usando tu ImageProcessor
        # La imagen se carga normalizada [0, 1] y en escala de grises
        img_lr = ImageProcessor.load_image_from_upload(file, grayscale=True)
        
        # 4. Construcción de Operadores (A y Regularizador) [cite: 113]
        degradation_op = DegradationOperator(
            scale_factor=params['scale_factor'],
            sigma=params['sigma']
        )
        
        if params['regularizer'] == 'l2':
            regularizer = L2GradientRegularizer()
            reg_name = "L2 (Tikhonov)"
        else:
            regularizer = HuberGradientRegularizer(delta=params['delta'])
            reg_name = f"Huber (delta={params['delta']})"

        # 5. Configuración y Ejecución del Solver [cite: 114]
        solver = SuperResolutionSolver(
            degradation_op=degradation_op,
            regularizer=regularizer,
            lambda_reg=params['lambda_reg'],
            tau=params['tau'],
            max_iter=params['max_iter'],
            verbose=True # Ver logs en la consola del servidor
        )
        
        # Ejecutamos la optimización
        # Nota: Asumimos que la imagen subida ES la imagen de baja resolución (b)
        img_hr, stats = solver.solve(img_lr)

        # 6. Preparación de Resultados para Visualización [cite: 116]
        
        # Convertir tensores a Base64 para enviarlos al HTML sin guardar en disco
        # Usamos tu utilidad array_to_base64
        img_lr_b64 = ImageProcessor.array_to_base64(img_lr, denormalize=True)
        img_hr_b64 = ImageProcessor.array_to_base64(img_hr, denormalize=True)
        
        # Generar gráfico de convergencia
        fig_conv = PlotUtils.plot_convergence(stats)
        plot_conv_b64 = PlotUtils.fig_to_base64(fig_conv)

        # Renderizar plantilla de resultados
        return render_template(
            'result.html',
            img_lr_data=img_lr_b64,
            img_hr_data=img_hr_b64,
            plot_data=plot_conv_b64,
            params=params,
            reg_name=reg_name,
            final_cost=stats['cost_history'][-1] if len(stats['cost_history']) > 0 else 0
        )

    except Exception as e:
        # Captura de errores inesperados durante el procesamiento numérico
        print(f"Error interno: {e}")
        flash(f"Ocurrió un error durante el procesamiento: {str(e)}")
        return redirect(url_for('index'))

if __name__ == '__main__':
    # Ejecución de la app
    app.run(debug=True, port=5000)
