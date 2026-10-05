# Escáner de Documentos Inteligente (Computer Vision)

En este proyecto, la idea principal es utilizar un escáner de documentos para poder escanear un documento con el objetivo de convertir una imagen a formato PDF. Se utiliza Python y principalmente las librerías de Numpy y OpenCV para el procesamiento de las imágenes y videos.

## 🚀 Características
*   Captura de vídeo en tiempo real.
*   Utilización de filtros de suavizado, algoritmo de Canny para detección de bordes y de Harrys y LCC para detectar keypoints.
*   Detección Geométrica adaptativa, corrección de perspectiva(Homografía) y alineación automática de vértices

## 🧠 Pipeline Técnico (Arquitectura)
Explica aquí brevemente las 4 fases matemáticas por las que pasa cada frame:
1.  **Captura y E/S**: Lectura del buffer de vídeo mediante `cv2.VideoCapture`.
2.  **Preprocesamiento Espacial**: Para la primera fase, vamos a usar una escala de grises para poder encontrar facilmente el contorno del documento que estamos analizando. Posteriormente, aplicaremos un filtro gaussiano para eliminar el posible ruido que tiene la imagen. Detectaremos los bordes utilizando el algoritmo de Canny. Algunas funciones que utilizaremos: cv2.cvtColor, cv2.GaussianBlur y cv2.Canny.
3.  **Extracción de Geometría**: Búsqueda de contornos, aproximación poligonal (Douglas-Peucker) para simplificar vértices y filtrado heurístico para aislar el polígono cuadrangular más grande.
4.  **Transformación de Perspectiva**: Ordenación espacial de las 4 esquinas y aplicación de matriz de homografía (`cv2.warpPerspective`) para obtener el plano cenital.

## ⚙️ Requisitos e Instalación
Estas son las intrucciones y comandos a ejecutar para poder utilizar el sistema en tu máquina:
-Tener instalado Python y tener una cámara web operativa
-Clonar el código de este mismo repositorio
-Crear un entorno virtual `python -m venv venv`
-Activar el entorno (en CMD ya que en PowerShell no funciona por tema de permisos): `.\venv\Scripts\activate` (Todo esto dentro de tu carpeta)
-Instalar numpy y OpenCV dentro del entorno virtual que acabas de crear

\`\`\`bash
git clone https://github.com/javiiacedo/doc-scanner.git
cd Documents/doc-scanner
python -m venv venv
.\venv\Scripts\activate
pip install opencv-python numpy
\`\`\`

## 📂 Estructura del Proyecto

*   `main.py`: Punto de entrada y bucle principal de vídeo.
*   `io/camera_stream.py`: Encapsulación del hardware de captura.
*   `core/`: 

## 🎮 Uso
Cómo se arranca el programa y qué teclas debe usar el usuario (ej: 'q' para salir, 's' para escanear).

\`\`\`bash
python main.py
\`\`\`