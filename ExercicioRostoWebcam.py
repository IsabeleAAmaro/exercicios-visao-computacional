import cv2


def resize(img, largura):
    altura = int(img.shape[0] / img.shape[1] * largura)
    img = cv2.resize(img, (largura, altura), interpolation=cv2.INTER_AREA)
    return img


df = cv2.CascadeClassifier('venv\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
while True:
    (sucesso, frame) = camera.read()
    if not sucesso:
        break

    # reduz tamanho do frame para acelerar processamento
    frame = resize(frame, 320)

    # converte para tons de cinza
    frame_pb = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detecta as faces no frame
    faces = df.detectMultiScale(frame_pb, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20),
                                flags=cv2.CASCADE_SCALE_IMAGE)
    frame_temp = frame.copy()
    for (x, y, lar, alt) in faces:
        cv2.rectangle(frame_temp, (x, y), (x + lar, y + alt), (0, 255, 255), 2)

    # Exibe um frame redimensionado (com perca de qualidade)
    cv2.imshow("Encontrando faces... ", resize(frame_temp, 640))

    # Espera que a tecla 'q' seja pressionada para sair
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
