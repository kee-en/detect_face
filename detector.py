from pathlib import Path
import face_recognition
import pickle
import cv2

DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")

Path("training").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)
Path("validation").mkdir(exist_ok=True)

def load_and_convert_image(file_path):
    # Load the image using OpenCV
    image = cv2.imread(str(file_path))

    # Convert the image from BGR (OpenCV format) to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image_rgb

def encode_known_faces(
        model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH
) -> None:
    names = []
    encodings = []
    for filepath in Path("training").glob("*/*"):
        name = filepath.parent.name
        image = load_and_convert_image(filepath)

        # Debug prints to verify image type and shape
        print(f"Processing {filepath}")
        print(f"Image type: {type(image)}")
        print(f"Image shape: {image.shape}")
        print(f"Image dtype: {image.dtype}")

        face_locations = face_recognition.face_locations(image, model=model)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        for encoding in face_encodings:
            names.append(name)
            encodings.append(encoding)

    name_encodings = {
        "names": names,
        "encodings": encodings}
    with encodings_location.open(mode="wb") as f:
        pickle.dump(name_encodings, f)

encode_known_faces()
