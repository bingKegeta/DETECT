import cv2

cap = cv2.VideoCapture("test_media/session_video.mp4")  # Change to 0 for webcam
if not cap.isOpened():
    print("Error: Unable to open video source.")
else:
    print("Video source opened successfully.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break
cap.release()
cv2.destroyAllWindows()
