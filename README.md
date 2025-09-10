🧠❤️ Herramienta EmoKids
Herramienta EmoKids es una aplicación web diseñada para ayudar a padres y educadores a comprender y apoyar la comunicación emocional en niños. A través del análisis de video, la herramienta detecta emociones faciales, genera un reporte detallado y ofrece recomendaciones personalizadas para abordar las necesidades específicas de cada niño, especialmente aquellos con desafíos en la comunicación.

🌟 Características Principales
Análisis de Video: Procesa archivos de video para detectar rostros y analizar las emociones fotograma a fotograma.

Detección Emocional: Identifica emociones clave como felicidad, tristeza, enojo, sorpresa, disgusto, miedo y neutralidad.

Reporte Detallado: Genera un informe completo en formato TXT que incluye un resumen del análisis, un histograma emocional, recomendaciones prácticas y una galería de fotogramas clave.

Recomendaciones Personalizadas: Utiliza la API de Gemini de Google para ofrecer consejos específicos basados en la emoción dominante y la condición de comunicación del niño (ej. TEA, TDAH).

Consejos Fundamentales: Incluye recomendaciones predefinidas y específicas para cada condición de comunicación, proporcionando una base sólida de apoyo.

Interfaz Intuitiva: Interfaz de usuario limpia y fácil de usar para cargar videos, ingresar datos contextuales y visualizar los resultados del análisis.

⚙️ Tecnologías Utilizadas
Backend: Python con el framework Flask

Procesamiento de Imágenes: OpenCV

Inteligencia Artificial: Google Gemini API

Frontend: HTML5, CSS3 y JavaScript

🛠️ Requisitos del Sistema
Python 3.7 o superior

Las siguientes librerías de Python:

Flask

opencv-python

numpy

google-generativeai

🚀 Instalación y Uso
Sigue estos pasos para poner en marcha el proyecto:

Clona o descarga el proyecto:

Bash

git clone https://github.com/tu_usuario/emokids.git
cd emokids
Crea y activa un entorno virtual (recomendado):

Bash

python -m venv venv
# En Windows
venv\Scripts\activate
# En macOS/Linux
source venv/bin/activate
Instala las dependencias:

Bash

pip install Flask opencv-python numpy google-generativeai
(Nota: la dependencia de TensorFlow y los modelos .h5 son para una versión futura que implemente detección real. La versión actual utiliza una simulación para garantizar el funcionamiento sin fallos.)

Configura tu API Key:
Necesitas una clave de API de Google para usar la API de Gemini. Guárdala como una variable de entorno en tu sistema.

En Windows:

DOS

set GOOGLE_API_KEY="TU_CLAVE_API"
En macOS/Linux:

Bash

export GOOGLE_API_KEY="TU_CLAVE_API"
Ejecuta la aplicación:

Bash

python app.py
La aplicación se iniciará en http://127.0.0.1:5000.

📂 Estructura del Proyecto
/EmoKids
|-- app.py                  # Servidor backend de Flask
|-- index.html              # Interfaz de usuario
|-- haarcascade_frontalface_default.xml # Detector de rostros de OpenCV
|-- /static                 # Archivos estáticos (CSS, JS, etc.)
|   |-- style.css
|-- /uploads                # Videos temporales subidos por el usuario
|-- /output                 # Resultados de análisis (fotogramas, reportes)
|-- README.md               # Este archivo
|-- .gitignore
🤝 Contribución
¡Las contribuciones son bienvenidas! Si encuentras un error o tienes una idea para mejorar, no dudes en abrir un issue o enviar un pull request.