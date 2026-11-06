
from pathlib import Path

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1


# ---------- SETTINGS ----------
SAMPLES_REQUIRED = 50
MIN_FACE_CONFIDENCE = 0.90

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
DATABASE_PATH = DATA_DIR / "face_embeddings.npz"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------- MODELS ----------
detector = MTCNN(
    image_size=160,
    margin=0,
    keep_all=True,
    device=device
)

facenet = InceptionResnetV1(
    pretrained="vggface2"
).eval().to(device)


def save_embeddings(new_embeddings, person_name):
    """Append this person's samples to the existing database."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    new_embeddings = np.stack(new_embeddings).astype(np.float32)
    new_labels = np.array(
        [person_name] * len(new_embeddings),
        dtype=str
    )

    if DATABASE_PATH.exists():
        with np.load(DATABASE_PATH, allow_pickle=False) as data:
            old_embeddings = data["embeddings"]
            old_labels = data["labels"]

        all_embeddings = np.concatenate(
            [old_embeddings, new_embeddings], axis=0
        )
        all_labels = np.concatenate([old_labels, new_labels])
    else:
        all_embeddings = new_embeddings
        all_labels = new_labels

    np.savez_compressed(
        DATABASE_PATH,
        embeddings=all_embeddings,
        labels=all_labels
    )

    print("\nEnrollment saved successfully!")
    print("Person:", person_name)
    print("New samples:", len(new_embeddings))
    print("Total samples in database:", len(all_labels))
    print("File:", DATABASE_PATH)


# ---------- ENROLLMENT ----------
def main():
    person_name = input("Enter the person's name: ").strip()

    if not person_name:
        print("Name cannot be empty.")
        return

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open webcam.")
        return

    embeddings = []

    print("\nRunning on:", device)
    print("Keep only ONE person's face in the frame.")
    print("Press S to save a face sample.")
    print("Press Q to finish and save collected samples.")
    print("Change your head angle or expression between samples.")

    try:
        while len(embeddings) < SAMPLES_REQUIRED:
            success, frame = cap.read()

            if not success:
                print("Could not read webcam frame.")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            boxes, probabilities = detector.detect(rgb_frame)

            valid_face = (
                boxes is not None
                and probabilities is not None
                and len(boxes) == 1
                and probabilities[0] is not None
                and probabilities[0] >= MIN_FACE_CONFIDENCE
            )

            if boxes is not None:
                for box, probability in zip(boxes, probabilities):
                    if probability is None:
                        continue

                    x1, y1, x2, y2 = map(int, box)

                    color = (0, 255, 0) if valid_face else (0, 165, 255)

                    cv2.rectangle(
                        frame, (x1, y1), (x2, y2), color, 2
                    )

                    cv2.putText(
                        frame,
                        f"Face: {probability:.2f}",
                        (x1, max(25, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2
                    )

            status = (
                "Ready - press S to save"
                if valid_face
                else "Keep ONE clear face in view"
            )

            cv2.putText(
                frame,
                f"Samples: {len(embeddings)}/{SAMPLES_REQUIRED}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

            cv2.putText(
                frame,
                status,
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )

            cv2.imshow("Future Vision - Face Enrollment", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                if not valid_face:
                    print("Sample rejected: need exactly one clear face.")
                    continue

                # Extract the aligned 160x160 face crop.
                face = detector.extract(
                    rgb_frame, boxes, save_path=None
                )

                if face is None:
                    print("Could not extract the face. Try again.")
                    continue

                # Convert the face crop into a 512-number embedding.
                face = face[0].unsqueeze(0).to(device)

                with torch.no_grad():
                    embedding = facenet(face)

                embedding = embedding.squeeze(0).cpu().numpy()
                embeddings.append(embedding)

                print(
                    f"Saved sample {len(embeddings)}/{SAMPLES_REQUIRED}"
                )

            elif key == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    if embeddings:
        save_embeddings(embeddings, person_name)
    else:
        print("No samples collected. Nothing was saved.")


if __name__ == "__main__":
    main()
