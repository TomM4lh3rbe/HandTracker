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
    with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:
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
            hauteur, largeur, couleurs = image.shape
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            resultat = hands.process(rgb)

            if resultat.multi_hand_landmarks:
                for hand_landmarks in resultat.multi_hand_landmarks:
                    mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            cv2.imshow("HandTracking", image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if cv2.getWindowProperty("HandTracking", cv2.WND_PROP_VISIBLE) < 1 :
                break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
