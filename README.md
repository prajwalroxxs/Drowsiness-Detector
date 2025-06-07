# Drowsiness-Detector
A Python-based drowsiness detection system that uses real-time webcam feed to monitor eye closure. If your eyes appear drowsy (closed beyond a threshold), it triggers an alert sound and can also send an SMS alert using Twilio.

🔧 Requirements
1. opencv-python
2. imutils
3. scipy
4. dlib
5. numpy
6. pygame
7. twilio

Install them with:
pip install opencv-python imutils scipy dlib numpy pygame twilio

📦 Features
Detects eye closure using Eye Aspect Ratio (EAR)

Plays alert sound if drowsiness is detected

Optional SMS alert with Twilio

📬 Optional: Twilio SMS Alert
Update your Twilio credentials in the script to enable SMS notifications.
