from flask import Flask, render_template, Response 
import cv2
import socket 
import io 

app = Flask(__name__) 
vc = cv2.VideoCapture(0) 

@app.route('/') 
def index(): 
   return render_template('index.html') 
def gen(): 
   while True: 
       rval, frame = vc.read() 
       cv2.imwrite('pic.jpg', frame) 
       yield (b'--frame\r\n' 
              b'Content-Type: image/jpeg\r\n\r\n' + open('pic.jpg', 'rb').read() + b'\r\n') 

@app.route('/video_feed') 
def video_feed(): 
   return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame') 

if __name__ == '__main__': 
	app.run(host='0.0.0.0', debug=True, threaded=True) 