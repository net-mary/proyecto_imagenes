import os
import uuid
import cv2
import json
import numpy as np
import google.generativeai as genai
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory, render_template
from werkzeug.utils import secure_filename

# --- CONFIGURACIÓN DE LA APLICACIÓN ---
app = Flask(__name__)

# Configura la API de Gemini con la clave de la variable de entorno
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

# Rutas y carpetas
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Detección facial (usando el clasificador de OpenCV)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Mapa de emociones y emojis
EMOTION_MAP = {
    'happy': 'Feliz', 'sad': 'Triste', 'angry': 'Enojado',
    'neutral': 'Neutro', 'surprise': 'Sorprendido',
    'disgust': 'Disgustado', 'fear': 'Temeroso'
}
EMOTION_ICONS = {
    'happy': '😊', 'sad': '😔', 'angry': '😠',
    'neutral': '😐', 'surprise': '😮',
    'disgust': '🤢', 'fear': '😱'
}

# --- FUNCIONES DEL MODELO DE IA ---
def get_recommendations_from_gemini(emotion, context, condition):
    """
    Genera recomendaciones personalizadas usando la API de Gemini.
    """
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    prompt = f"""
    Eres un asistente profesional para padres y educadores que apoya a niños con desafíos en la comunicación. Tu objetivo es proporcionar consejos empáticos y prácticos, no clínicos.

    Un padre o educador ha solicitado recomendaciones basadas en el siguiente análisis de video de un niño:
    - Emoción dominante detectada: {EMOTION_MAP.get(emotion, 'desconocida')}
    - Contexto de la situación: {context}
    - Condición de comunicación del niño: {condition}

    Basado en estos datos, genera 3 o 4 consejos concisos y prácticos. Las recomendaciones deben ser:
    1. Enfocadas en la acción (qué hacer).
    2. Evitar la jerga técnica o clínica.
    3. Ser empáticas y positivas.
    4. Considerar la condición específica del niño si aplica.

    Escribe la respuesta en formato de lista JSON, donde cada elemento tiene un 'title' y un 'text'. Por ejemplo:
    [
        {{
            "title": "Un título corto",
            "text": "Una recomendación detallada y amigable."
        }},
        {{
            "title": "Otro título",
            "text": "Otra recomendación..."
        }}
    ]
    """
    try:
        response = model.generate_content(prompt)
        recommendations = json.loads(response.text)
        return recommendations
    except Exception as e:
        print(f"Error al generar recomendaciones con Gemini: {e}")
        return [{
            "title": "Error al generar recomendaciones",
            "text": "No pudimos generar recomendaciones personalizadas. Por favor, inténtalo de nuevo más tarde."
        }]

def get_basic_recommendations(condition):
    """
    Proporciona recomendaciones básicas predefinidas según la condición.
    """
    recs = {
        'TEA': [
            {"title": "Comunicación Clara", "text": "Utiliza frases cortas y claras. Apoya tus palabras con gestos o imágenes para reducir la ambigüedad."},
            {"title": "Establece una Rutina", "text": "La previsibilidad ayuda a los niños con TEA. Mantén rutinas diarias para que sepan qué esperar, lo que reduce la ansiedad."}
        ],
        'TDHA': [
            {"title": "Manejo del Entorno", "text": "Minimiza las distracciones en el entorno. Un espacio tranquilo y ordenado puede ayudar a mantener la concentración."},
            {"title": "Instrucciones Simples", "text": "Divide las tareas grandes en pasos más pequeños. Da una instrucción a la vez y espera a que la complete antes de dar la siguiente."}
        ],
        'ParalisisCerebral': [
            {"title": "Crea un Ambiente Seguro", "text": "Asegura el entorno para evitar caídas. Usa mobiliario adaptado para mayor seguridad."},
            {"title": "Fomenta la Movilidad", "text": "Utiliza juguetes que animen el movimiento y la coordinación. Trabaja en la fuerza muscular y la flexibilidad con ejercicios simples."}
        ],
        'RetrasoMadurativo': [
            {"title": "Estimulación Cognitiva", "text": "Juega con el niño para estimular sus sentidos. Anímalo a resolver problemas sencillos y a participar en actividades motoras finas."},
            {"title": "Refuerzo Positivo", "text": "Reconoce y celebra los logros del niño, sin importar lo pequeños que sean. El refuerzo positivo aumenta la confianza y el deseo de aprender."}
        ],
        'Otro': [
            {"title": "Observación y Registro", "text": "Observa con atención cómo reacciona el niño a diferentes estímulos y situaciones. Lleva un registro de los comportamientos para encontrar patrones."},
            {"title": "Colaboración con Especialistas", "text": "Consulta a un profesional de la salud o educador. Ellos pueden proporcionar herramientas y estrategias adaptadas al niño."}
        ]
    }
    return recs.get(condition, [])

def predict_emotion_simulated(face_image):
    """Simula la predicción de emociones."""
    emotions = ['happy', 'sad', 'angry', 'neutral', 'surprise', 'disgust', 'fear']
    # Distribución de probabilidad simulada
    weights = [0.2, 0.1, 0.1, 0.3, 0.1, 0.1, 0.1]
    import random
    dominant_emotion = random.choices(emotions, weights=weights, k=1)[0]
    confidence = random.uniform(0.7, 0.95)
    return dominant_emotion, confidence

# --- RUTAS DE LA APLICACIÓN ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No se encontró el archivo de video.'}), 400

    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'No se seleccionó ningún archivo.'}), 400

    child_name = request.form.get('name', 'Niño/a')
    situation = request.form.get('situation', 'sin contexto')
    condition = request.form.get('condition', 'sin condición')
    
    # Crea una carpeta única para este análisis
    analysis_dir_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    session_output_path = os.path.join(OUTPUT_FOLDER, analysis_dir_name)
    os.makedirs(session_output_path)

    # Guarda el video temporalmente
    filename = secure_filename(video_file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    video_file.save(filepath)

    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        return jsonify({'error': 'Error al abrir el video.'}), 500

    frame_count = 0
    emotion_histogram = {}
    total_confidence = 0
    captured_frames = []
    
    # Procesamiento del video fotograma a fotograma
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Procesar solo cada X fotogramas
        if frame_count % 30 == 0:  # Cada segundo
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                
                # Simular la detección de emoción y confianza
                emotion, confidence = predict_emotion_simulated(face_roi)

                # Dibuja el recuadro y la etiqueta en el fotograma
                color = (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, EMOTION_MAP[emotion], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
                # Actualizar el histograma
                emotion_histogram[emotion] = emotion_histogram.get(emotion, 0) + 1
                total_confidence += confidence

                # Guarda el fotograma etiquetado
                timestamp = str(datetime.now().strftime('%H-%M-%S'))
                output_filename = f"frame_{timestamp}_{EMOTION_MAP[emotion]}.jpg"
                output_path = os.path.join(session_output_path, output_filename)
                cv2.imwrite(output_path, frame)
                
                captured_frames.append({
                    'image_url': f'/output/{analysis_dir_name}/{output_filename}',
                    'timestamp': timestamp,
                    'emotion': emotion,
                    'emotion_es': EMOTION_MAP[emotion]
                })

    cap.release()
    os.remove(filepath)

    # Calcular resultados
    total_emotions = sum(emotion_histogram.values())
    if total_emotions > 0:
        dominant_emotion = max(emotion_histogram, key=emotion_histogram.get)
        dominant_percentage = (emotion_histogram[dominant_emotion] / total_emotions) * 100
        avg_confidence = total_confidence / total_emotions
    else:
        dominant_emotion = 'neutral'
        dominant_percentage = 100
        avg_confidence = 0
    
    # Obtener recomendaciones básicas y las de Gemini
    basic_recommendations = get_basic_recommendations(condition)
    gemini_recommendations = get_recommendations_from_gemini(dominant_emotion, situation, condition)
    
    # Combinar ambas listas de recomendaciones
    all_recommendations = basic_recommendations + gemini_recommendations

    # Crear el reporte en formato TXT
    report_data = f"""
--- REPORTE DE ANÁLISIS EmoKids ---
Fecha y Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
ID del Análisis: {analysis_dir_name}

> Datos del Niño/a:
Nombre: {child_name}
Contexto de la Situación: {situation}
Condición de Comunicación: {condition}

--- RESUMEN DEL ANÁLISIS ---
Emoción Dominante: {EMOTION_MAP[dominant_emotion]} ({dominant_percentage:.1f}% del tiempo)
Fotogramas Analizados: {frame_count}
Confianza Promedio del Análisis: {avg_confidence:.1f}%

--- HISTOGRAMA EMOCIONAL ---
"""
    for emotion, count in emotion_histogram.items():
        percentage = (count / total_emotions) * 100 if total_emotions > 0 else 0
        report_data += f"{EMOTION_MAP[emotion]}: {percentage:.1f}%\n"

    report_data += "\n--- RECOMENDACIONES PERSONALIZADAS ---\n"
    # Añadir recomendaciones básicas al reporte
    report_data += "--- Consejos Fundamentales ---\n"
    if basic_recommendations:
        for rec in basic_recommendations:
            report_data += f"Título: {rec.get('title', 'N/A')}\n"
            report_data += f"Consejo: {rec.get('text', 'N/A')}\n\n"
    else:
        report_data += "No se encontraron consejos fundamentales para esta condición.\n\n"

    # Añadir recomendaciones de Gemini al reporte
    report_data += "--- Consejos de la IA de Gemini ---\n"
    if gemini_recommendations:
        for rec in gemini_recommendations:
            report_data += f"Título: {rec.get('title', 'N/A')}\n"
            report_data += f"Consejo: {rec.get('text', 'N/A')}\n\n"
    else:
        report_data += "No se pudieron generar recomendaciones personalizadas.\n\n"

    report_data += "\n--- FOTOGRAMAS CLAVE ---\n"
    for frame in captured_frames:
        report_data += f"Fotograma: {frame['image_url']} - Emoción: {frame['emotion_es']} - Tiempo: {frame['timestamp']}\n"
    
    report_filename = f'reporte_{analysis_dir_name}.txt'
    report_path = os.path.join(session_output_path, report_filename)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_data)

    return jsonify({
        'dominant_emotion': {
            'emotion_es': EMOTION_MAP[dominant_emotion],
            'icon': EMOTION_ICONS[dominant_emotion],
            'percentage': dominant_percentage
        },
        'total_frames': frame_count,
        'avg_confidence': avg_confidence,
        'emotion_histogram': {EMOTION_MAP[e]: v for e, v in emotion_histogram.items()},
        'emotion_map': EMOTION_MAP,
        'frames': captured_frames,
        'recommendations': all_recommendations,
        'report_id': analysis_dir_name, # Usamos el nombre de la carpeta como ID
    })

@app.route('/output/<path:filepath>')
def serve_output_file(filepath):
    return send_from_directory(OUTPUT_FOLDER, filepath)

@app.route('/report/<report_id>')
def download_report(report_id):
    report_filename = f'reporte_{report_id}.txt'
    report_path = os.path.join(OUTPUT_FOLDER, report_id, report_filename)
    if not os.path.exists(report_path):
        return jsonify({'error': 'Reporte no encontrado.'}), 404

    return send_from_directory(os.path.join(OUTPUT_FOLDER, report_id), report_filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)