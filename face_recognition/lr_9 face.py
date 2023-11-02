import dlib
import cv2


# Загрузка предварительно обученной модели dlib для распознавания лиц
detector = dlib.get_frontal_face_detector()

# Открытие веб-камеры
cap = cv2.VideoCapture(0)  # 0 обозначает встроенную или первую подключенную камеру

while True:
    # Захват кадра с веб-камеры
    error_code, frame = cap.read()

    if not error_code:
        break

    # Преобразование кадра в оттенки серого
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Обнаружение лиц на кадре
    faces = detector(gray)

    # Отмечаем найденные лица на кадре
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Отображаем результат
    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Нажмите Esc, чтобы выйти
        break

# Освобождаем ресурсы и закрываем окна
cap.release()
cv2.destroyAllWindows()
