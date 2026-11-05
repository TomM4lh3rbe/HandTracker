import math
import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python.components.containers import landmark

# Détection de la main et dessin de l'armature
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

#Caméra
video = cv2.VideoCapture(0)
video.set(3, 1280) # Résolution de la vidéo
video.set(4, 720)

#Couleurs
COULEUR_CONTOUR = (252, 211, 125)
COULEUR_ANCRAGE = (246, 130, 59)

# Calcule du pincement du pouce et de l'index
def calculer_pincement(hand_landmarks, largeur, hauteur):
    pouce = hand_landmarks.landmark[4]
    index = hand_landmarks.landmark[8]

    distance = math.hypot((pouce.x - index.x) * largeur,
                          (pouce.y - index.y) * hauteur)

    est_pince = distance < 40  # seuil en pixels, à ajuster
    milieu = (int((pouce.x + index.x) / 2 * largeur),
              int((pouce.y + index.y) / 2 * hauteur))
    return est_pince, milieu
#Dessins
def dessiner_ancrage(image, centre):
    cv2.circle(image, centre, 10, COULEUR_ANCRAGE, 3)

def dessiner_rectangle(image, point1, point2, couleur):
    cv2.rectangle(image, point1, point2, couleur, 2)
    largeur = abs(point1[0] - point2[0])
    hauteur = abs(point1[1] - point2[1])
    x = max(point1[0], point2[0])
    y = max(point1[1], point2[1])

    cv2.putText(image, f"{largeur} x {hauteur}", (x - 125, y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

#Lancement du programme
def main():
    afficher_doigts = False
    afficher_armature = True
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

            # Recalculée à chaque image : si plus de pincement, plus d'ancrage
            ancrages = []

            if resultat.multi_hand_landmarks:
                for hand_landmarks in resultat.multi_hand_landmarks:
                    if afficher_armature == True:
                        mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    doigts_desc = {
                        "Pouce": hand_landmarks.landmark[4],
                        "Index": hand_landmarks.landmark[8],
                        "Majeur": hand_landmarks.landmark[12],
                        "Annulaire": hand_landmarks.landmark[16],
                        "Auriculaire": hand_landmarks.landmark[20],
                    }
                    if afficher_doigts:
                        for nom, point in doigts_desc.items():
                            x, y = int(point.x * largeur), int(point.y * hauteur)
                            cv2.putText(image, nom, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    if afficher_armature == True:
                        for nom, point in doigts_desc.items():
                            x, y = int(point.x * largeur), int(point.y * hauteur)
                            cv2.circle(image, (x,y),5,(0,255,0),-1)

                    est_pince, milieu = calculer_pincement(hand_landmarks, largeur, hauteur)
                    if est_pince:
                        ancrages.append(milieu)
            if afficher_armature == True:
                for point in ancrages:
                    dessiner_ancrage(image, point)
            if len(ancrages) == 2:
                dessiner_rectangle(image, ancrages[0], ancrages[1], COULEUR_CONTOUR)

            cv2.imshow("HandTracking", image)

            touche = cv2.waitKey(1) & 0xFF
            if touche == ord('q'):
                break

            if touche == ord('f'):
                if afficher_doigts:
                    afficher_doigts = False
                else:
                    afficher_doigts = True

            if touche == ord("a"):
                if afficher_armature == True:
                    afficher_armature = False
                else:
                    afficher_armature = True


            if cv2.getWindowProperty("HandTracking", cv2.WND_PROP_VISIBLE) < 1 :
                break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()