import cv2
import os

SAVE_FOLDER = 'my_signatures'
os.makedirs(SAVE_FOLDER, exist_ok=True)

def get_next_filename(folder):
    existing_files = [f for f in os.listdir(folder) if f.startswith("signature_") and f.endswith(".jpg")]
    numbers = [int(f.split('_')[1].split('.')[0]) for f in existing_files if f.split('_')[1].split('.')[0].isdigit()]
    next_num = max(numbers, default=0) + 1
    return os.path.join(folder, f"signature_{next_num}.jpg")

def main():
    cap = cv2.VideoCapture(0)

    print("Press SPACE to capture and save signature. Press ESC to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        cv2.putText(display, "SPACE: Capture | ESC: Quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Capture Signature', display)

        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break
        elif key == 32:  # SPACE
            filename = get_next_filename(SAVE_FOLDER)
            cv2.imwrite(filename, frame)
            print(f"Saved: {filename}")
            cv2.putText(frame, f"Saved: {os.path.basename(filename)}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow('Capture Signature', frame)
            cv2.waitKey(1000)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
