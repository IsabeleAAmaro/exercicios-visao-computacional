import numpy as np
import cv2

cap = cv2.VideoCapture(0)
while True:
    (x, frame) = cap.read()

    # Se for false sai do while
    if not x:
        break

    # Redimensiona o frame pela metade utilizando a técnica slicing
    frame = frame[::2, ::2]

    # Converte para um frame em tons de cinza
    frame_pb = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # cv2.cvtColor(<imagem>, <Tipo a converter>)

    # Suaviza ou borra o frame e tira o ruído dele. Aumenta a precisão das bordas     Sempre ímpares
    frame_sv = cv2.GaussianBlur(frame_pb, (3, 3), 0)

    (T, frame_bin1) = cv2.threshold(frame_sv, 160, 255,
                                    cv2.THRESH_BINARY)
    (T, frame_bin2) = cv2.threshold(frame_sv, 160, 255, cv2.THRESH_BINARY_INV)  # Inverso

    # Detecar bordas no frame usando a função Canny
    bordas = cv2.Canny(frame_bin1, 100,300)

    # Cria uma pilha vertical contendo pilhas horizontais, tudo isso para mostrar todas as alterações de uma vez
    pilha = np.vstack([
        np.hstack([frame_pb, frame_sv]),
        np.hstack([frame_bin1, bordas])
    ])

    # Aplica a função goodFeaturesToTrack
    corners = cv2.goodFeaturesToTrack(gray, 100, 0.1, 10)

    # 100 = número de cantos
    # 0.1 = qualidade (0 a 1)
    # 10 = distancia minima entre cantos (que são detectados)
    if corners is not None:
        corners = np.int0(corners)
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

    # Mostra o frame
    cv2.imshow("Original", frame)
    cv2.imshow("Resultado", pilha)

    # Aguarda o usuário digitar a tecla 'q' para quebrar o processo
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
