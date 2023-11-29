import cv2
import dlib

# Carregando o detector de faces pré-treinado
detector_faces = dlib.get_frontal_face_detector()

# Carregando o modelo de pontos faciais (shape predictor) pré-treinado
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Baixe o modelo a partir do site do dlib

# Carregando o classificador de faces
classificador_faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inicializando a captura de vídeo
cap = cv2.VideoCapture(0)

while True:
    # Lendo o frame do vídeo
    ret, frame = cap.read()

    # Convertendo o frame para escala de cinza
    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectando faces usando o classificador Haarcascade
    faces_haarcascade = classificador_faces.detectMultiScale(cinza, scaleFactor=1.3, minNeighbors=5)

    # Detectando faces usando o detector dlib
    faces_dlib = detector_faces(cinza)

    # Desenhando retângulos ao redor das faces detectadas pelo classificador Haarcascade
    for (x, y, w, h) in faces_haarcascade:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Desenhando retângulos ao redor das faces detectadas pelo detector dlib
    for face in faces_dlib:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Exibindo o frame resultante
    cv2.imshow('Reconhecimento Facial', frame)

    # Saindo do loop se a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberando os recursos
cap.release()
cv2.destroyAllWindows()
