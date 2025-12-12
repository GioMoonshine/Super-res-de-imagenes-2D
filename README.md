# Proyecto de Super-Resolución de Imágenes (Problema Inverso)

Este proyecto implementa una solución de **Super-Resolución de Imágenes** (SR) formulada como un problema inverso de optimización. Utiliza el método de **Descenso de Gradiente** para reconstruir una imagen de alta resolución (HR) a partir de una imagen de baja resolución (LR) ruidosa y degradada, incorporando técnicas de regularización para asegurar la calidad de la reconstrucción.

El objetivo es minimizar la función de costo:

$$J(\mathbf{x}) = \frac{1}{2} \| \mathbf{A}\mathbf{x} - \mathbf{b} \|_2^2 + \lambda \cdot R(\mathbf{x})$$

Donde:
* $\mathbf{x}$ es la imagen de alta resolución a reconstruir.
* $\mathbf{b}$ es la imagen de baja resolución de entrada.
* $\mathbf{A}$ es el Operador de Degradación (desenfoque + submuestreo).
* $R(\mathbf{x})$ es el término de Regularización (suavizado/preservación de bordes).
* $\lambda$ es el parámetro de regularización.

---

## Estructura del Proyecto

El código está organizado en módulos Python que encapsulan las principales componentes del problema inverso.

| Archivo | Descripción Principal | Componentes Clave |
| :--- | :--- | :--- |
| `operators.py` | Implementa el **Operador de Degradación** $\mathbf{A}$ y su adjunto $\mathbf{A}^\top$ (para el cálculo del gradiente). | `DegradationOperator`, `compute_gradient_x/y`, `compute_divergence`. |
| `regularizers.py` | Define las funciones de costo y sus gradientes para las técnicas de regularización. | **`L2GradientRegularizer`** (Tikhonov), **`HuberGradientRegularizer`** (Preservación de bordes). |
| `gradient_descent.py` | Contiene el **Algoritmo de Optimización** para minimizar la función de costo $J(\mathbf{x})$. | `SuperResolutionSolver`, Métodos `compute_cost` y `solve`. |
| `utils.py` | Clases utilitarias para manejo de imágenes y visualización de resultados/convergencia. | `ImageProcessor`, `PlotUtils`, `ValidationUtils`. |
| `index.html` | Interfaz web (Frontend) para configurar y ejecutar la super-resolución (selección de imagen, parámetros). |
| `result.html` | Plantilla para mostrar los resultados de la reconstrucción, parámetros y gráficas de convergencia. |

---

## Parámetros de Configuración

La reconstrucción es sensible a los parámetros del modelo, que pueden configurarse desde la interfaz web (`index.html`):

| Parámetro | Módulo Relacionado | Descripción | Valores Típicos |
| :--- | :--- | :--- | :--- |
| **Factor Escala** | `DegradationOperator` | Factor por el cual se aumentará la resolución (ej. 2x, 4x). | 2, 4 |
| **Sigma ($\sigma$)** | `DegradationOperator` | Desviación estándar del filtro Gaussiano (magnitud del desenfoque). | 0.5 - 2.0 |
| **Regularizador** | `SuperResolutionSolver` | Tipo de función de regularización utilizada. | `L2` (Suavizado), `Huber` (Preserva bordes) |
| **Lambda ($\lambda$)** | `SuperResolutionSolver` | Peso del término de regularización. Controla el balance entre fidelidad al dato y suavizado. | 0.001 - 0.1 |
| **Delta ($\delta$)** | `HuberRegularizer` | Umbral para el Regularizador de Huber. Borde más allá del cual la penalización se vuelve lineal. | 0.01 - 0.5 |
| **Paso ($\tau$)** | `SuperResolutionSolver` | Tasa de aprendizaje (`learning rate`) para el Descenso de Gradiente. | 0.0001 - 0.005 |
| **Máx. Iteraciones** | `SuperResolutionSolver` | Límite de pasos del algoritmo de Descenso de Gradiente. | 100 - 500 |

---

## Uso (Ejemplo Web)

Este proyecto parece estar diseñado para ejecutarse dentro de un entorno web (probablemente usando un framework como Flask o Django, a juzgar por el uso de `index.html` y `result.html` con *templating*).

1.  **Iniciar la Aplicación Web** (Asumiendo un *backend* existente no incluido):
    ```bash
    # Por ejemplo, si usas Flask:
    python app.py
    ```
2.  **Acceder a la Interfaz:** Navegue a la URL local proporcionada (típicamente `http://127.0.0.1:5000/`).
3.  **Configurar y Ejecutar:**
    * Suba una imagen de baja resolución (LR).
    * Ajuste los parámetros del modelo (Regularizador, $\lambda$, $\tau$, etc.).
    * Haga clic en **"Ejecutar Super-Resolución"**.
4.  **Revisar Resultados:** La página `result.html` mostrará:
    * La imagen LR de entrada.
    * La imagen HR reconstruida.
    * La gráfica del historial de convergencia (Costo Total, Costo de Datos, Costo de Reg. vs. Iteración).

---

## Observaciones Técnicas

* **Implementación de Gradiente:** Las funciones de gradiente en `regularizers.py` y el adjunto del operador de degradación en `operators.py` son cruciales para la convergencia del Descenso de Gradiente.
* **Regularizador de Huber:** La implementación del regularizador de Huber en `regularizers.py` es fundamental para lograr una buena **preservación de bordes** en la imagen reconstruida, ya que penaliza menos los grandes saltos de intensidad (bordes) que el regularizador L2.
* **Manejo de Imágenes:** `utils.py` se encarga de normalizar las imágenes a un rango flotante `[0.0, 1.0]`, un paso esencial para el procesamiento numérico estable.
