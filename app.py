from flask import Flask, request, render_template, jsonify
import nltk
import spacy
import Levenshtein
from collections import Counter

app = Flask(__name__)

# Cargar el modelo de spaCy para español
try:
    nlp = spacy.load("es_core_news_sm")
except OSError:
    print("Descargando modelo de spaCy para español...")
    spacy.cli.download("es_core_news_sm")
    nlp = spacy.load("es_core_news_sm")

# Descargar recursos de NLTK 
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Descargando stopwords de NLTK...")
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Descargando punkt de NLTK...")
    nltk.download('punkt')

# -------------------- Funciones Auxiliares --------------------

# Función para generar n-gramas
def generate_ngrams(text, n=3):
    text = text.lower()
    text = ''.join([c for c in text if c.isalpha() or c.isspace()])
    ngrams = [text[i:i+n] for i in range(len(text)-n+1)]
    return Counter(ngrams)

# -------------------- Funciones de Procesamiento de Texto (Arquitectura Pipeline) --------------------

def detect_language(text, language_profiles):
    def proportion_similarity(profile1, profile2):
        intersection = set(profile1.keys()) & set(profile2.keys())
        matches = sum(profile1[ngram] for ngram in intersection)
        total = sum(profile1.values())
        return matches / total if total > 0 else 0.0

    text_profile = generate_ngrams(text, n=3)
    similarities = {lang: proportion_similarity(text_profile, profile) for lang, profile in language_profiles.items()}
    detected_language = max(similarities, key=similarities.get)
    return detected_language, similarities

def correct_word(word, dictionary):
    min_distance = float('inf')
    corrected_word = word

    for dict_word in dictionary:
        distance = Levenshtein.distance(word, dict_word)
        if distance < min_distance and distance <= 2:  # Solo corrige si la distancia es pequeña
            min_distance = distance
            corrected_word = dict_word

    return corrected_word

def correct_text(text, dictionary):
    words = text.split()
    corrected_words = [correct_word(word, dictionary) for word in words]
    return " ".join(corrected_words)

def tokenize_text(text):
    return text.split()  # Tokeniza manualmente

def remove_stopwords(tokens, language='spanish'):
    try:
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words(language))
        return [word for word in tokens if word.lower() not in stop_words]
    except Exception as e:
        print(f"Error al eliminar stopwords: {e}")
        return tokens  # Si hay un error, devuelve los tokens originales

def lemmatize_text(text):
    doc = nlp(text)
    return [token.lemma_ for token in doc]

# -------------------- Diccionarios y Modelos Preentrenados --------------------

# Modelos de n-gramas preentrenados para la detección de idioma
language_profiles = {
    'es': generate_ngrams("hola cómo estás amigo este es un ejemplo en español", n=3),
    'en': generate_ngrams("hello how are you friend this is an example in english", n=3),
    'fr': generate_ngrams("bonjour comment ça va ami ceci est un exemple en français", n=3),
}

# Diccionario de palabras para la corrección ortográfica

dictionary = [
    # Saludos y expresiones comunes
    "hola", "adiós", "gracias", "por favor", "sí", "no", "buenos", "días", "buenas", "tardes", "noches",  
   
    # Pronombres personales  
    "yo", "tú", "vos", "usted", "él", "ella", "nosotros", "nosotras", "ustedes", "ellos", "ellas",  
   
    # Pronombres interrogativos  
    "qué", "quién", "quienes", "cuál", "cuáles", "dónde", "cómo", "cuándo", "por qué", "para qué", "cuánto", "cuántos", "cuántas",  

    # Verbos esenciales  
    "ser", "estar", "haber", "tener", "poder", "hacer", "decir", "ver", "dar", "saber", "querer",  
    "llegar", "pasar", "deber", "poner", "parecer", "quedar", "creer", "hablar", "llevar", "dejar",  
    "seguir", "encontrar", "llamar", "pensar", "salir", "volver", "tomar", "conocer", "vivir", "sentir",  
   
    # Conectores  
    "y", "o", "pero", "porque", "aunque", "sin embargo", "además", "entonces", "por lo tanto", "pues",  
    "así que", "mientras", "después", "antes", "durante", "luego", "incluso", "ya que", "a pesar de",  
   
    # Adverbios comunes  
    "bien", "mal", "mucho", "poco", "muy", "demasiado", "bastante", "siempre", "nunca", "a veces",  
    "ahora", "antes", "después", "luego", "pronto", "tarde", "temprano", "aquí", "allí", "allá",  
   
    # Sustantivos clave  
    "persona", "hombre", "mujer", "niño", "niña", "amigo", "familia", "gente", "padre", "madre",  
    "hermano", "hermana", "trabajo", "dinero", "casa", "ciudad", "país", "mundo", "vida", "tiempo",  
    "día", "semana", "mes", "año", "hora", "momento", "problema", "solución", "pregunta", "respuesta",  
    "historia", "palabra", "nombre", "forma", "manera", "caso", "hecho", "ejemplo", "razón", "verdad", "mentira"  
]


# -------------------- Rutas de Flask --------------------

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form['text']
        
        try:
            # Pipeline de procesamiento
            language, language_similarities = detect_language(text, language_profiles)
            corrected_text = correct_text(text, dictionary)
            tokens = tokenize_text(corrected_text)
            filtered_tokens = remove_stopwords(tokens, language)
            lemmas = lemmatize_text(" ".join(filtered_tokens))
            
            return render_template('index.html',
                                   original_text=text,
                                   language=language,
                                   language_similarities=language_similarities,
                                   corrected_text=corrected_text,
                                   tokens=tokens,
                                   filtered_tokens=filtered_tokens,
                                   lemmas=lemmas)
        except Exception as e:
            print(f"Error en el procesamiento: {e}")
            return "Error en el procesamiento", 500
    return render_template('index.html')

# -------------------- Configuración y Ejecución --------------------

if __name__ == '__main__':
    app.run(debug=True)
