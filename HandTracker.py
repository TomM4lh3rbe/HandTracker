import time

import cv2
import mediapipe as mp

# Détection de la main et dessin de l'armature
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

#Caméra
video = cv2.VideoCapture(0)
video.set(3, 1280) # Résolution de la vidéo
video.set(4, 720)


#Lancement du programme
def main():
    while True:
        essaie = 0
        succes, image = video.read()
        while not succes and essaie < 5 :
            time.sleep(0.2)
            succes, image = video.read()
            essaie += 1
        if not succes:
            print("Impossible de récuperer l'image")
            break

        image = cv2.flip(image, 1)
        cv2.imshow("HandTracking", image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()