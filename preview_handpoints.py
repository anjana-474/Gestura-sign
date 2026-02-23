# preview_handpoints.py

import cv2
from handpoints_utils import create_hands, extract_two_hand_vector, draw_hand_landmarks


def main():
    cap = cv2.VideoCapture(0)
    hands = create_hands(max_num_hands=2)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        info_text = "No hand detected"

        if results.multi_hand_landmarks:
            # Get combined 2-hand vector (126-dim)
            vec126 = extract_two_hand_vector(results)
            if vec126 is not None:
                info_text = f"Hands detected | feature len = {len(vec126)}"  # should be 126

            # Draw all hands
            for hand_lm in results.multi_hand_landmarks:
                draw_hand_landmarks(frame, hand_lm)

        cv2.putText(frame, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Handpoints Preview (2 Hands)", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
