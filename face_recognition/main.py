import cv2
from time import sleep

def face_capture():
    cascade_path = r"E:\Projects\NeuralNetworks\face_recognition\haarcascade_frontalface_default.xml"

    input_path = r"E:\Projects\NeuralNetworks\face_recognition\1.mp4"   

    clf = cv2.CascadeClassifier(cascade_path)
    camera = cv2.VideoCapture(input_path)

    if input_path.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        frame = cv2.imread(input_path)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = clf.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        for (x, y, width, height) in faces:
            cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 0), 2)
        cv2.imshow('Faces', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:

        while True:
            _, frame = camera.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = clf.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            for (x, y, width, height) in faces:
                cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 0), 2)

            cv2.imshow('Faces', frame)

            if cv2.waitKey(1) == ord('q'):
                break

            sleep(0.01)

        
    camera.release()
    cv2.destroyAllWindows()

def main():
    face_capture()

if __name__ == "__main__":
    main()