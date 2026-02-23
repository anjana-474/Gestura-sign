# handpoints_utils.py

import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def create_hands(max_num_hands=2):
    """
    Create and return a MediaPipe Hands object.
    max_num_hands=2 so we can capture 2-hand signs.
    """
    return mp_hands.Hands(max_num_hands=max_num_hands)


def extract_normalized_landmarks(hand_landmarks):
    """
    Single-hand:
    21 landmarks -> 63-dim vector with wrist (landmark 0) used as origin.
    """
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
    wrist = coords[0]
    coords = coords - wrist
    return coords.flatten()  # (21, 3) -> (63,)


def extract_two_hand_vector(results):
    """
    Extracts a 126-dim vector for up to 2 hands.

    - Each hand: 63 dims (normalized 21*(x,y,z))
    - Order: left hand first, right hand second (based on wrist x).
    - If only 1 hand is present, second hand is a zero vector.
    - If 0 hands -> returns None.
    """
    if not results.multi_hand_landmarks:
        return None  # no hands at all

    hands_lms = results.multi_hand_landmarks

    # collect (x_position, landmarks) for each detected hand
    hands_with_x = []
    for hand_lm in hands_lms:
        wrist_x = hand_lm.landmark[0].x
        hands_with_x.append((wrist_x, hand_lm))

    # sort by x so left hand (smaller x) comes first
    hands_with_x.sort(key=lambda t: t[0])

    vectors = []

    # take up to 2 hands in left-to-right order
    for i in range(min(2, len(hands_with_x))):
        _, hand_lm = hands_with_x[i]
        vec63 = extract_normalized_landmarks(hand_lm)
        vectors.append(vec63)

    # if only one hand -> pad second hand with zeros
    if len(vectors) == 1:
        vectors.append(np.zeros_like(vectors[0]))

    # final shape: (126,)
    return np.concatenate(vectors, axis=0)


def draw_hand_landmarks(frame, hand_landmarks):
    """
    Draw landmarks for one hand on the frame.
    Call this in a loop for each detected hand.
    """
    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    return frame
