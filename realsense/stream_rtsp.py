import cv2

# Your RTSP URL
rtsp_url = "rtsp://192.168.168.105:4000/fl"

# Open video stream
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Error: Cannot open RTSP stream")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    # Display the video
    cv2.imshow("RTSP Stream", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release and close
cap.release()
cv2.destroyAllWindows()

