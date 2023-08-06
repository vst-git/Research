# initiating flask for hosting
from flask import Flask, request, render_template, redirect, make_response,Response
from flask_cors import CORS
import time
import os
import cv2
from imutils import paths
import face_recognition
from sklearn.cluster import DBSCAN
import numpy as np
from imutils import build_montages
import collections

app = Flask(__name__)
CORS(app)

# the library for detecting the faces in frames {from opencv}
face_cascade = cv2. CascadeClassifier (cv2.data.haarcascades + "haarcascade_frontalface_default.xml")



data = None
labelIndex = {}
labelLoc = collections.defaultdict(set)

# def testAgainstMap():
#     data = []
#     targetImagePaths = list(paths.list_images("./videos/destinVideoFrames"))
#     for (i, imagePath) in enumerate(targetImagePaths):
#         image = cv2.imread(imagePath)
#         rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         boxes = face_recognition.face_locations(rgb, model="hog")
#         encodings = face_recognition.face_encodings(rgb, boxes)
#         d = [{"imagePath": imagePath, "loc": box, "encoding": enc} for (box, enc) in zip(boxes, encodings)]
#         data.extend(d)
#     data = np.array(data)
#     encodings = [d["encoding"] for d in data]
#     clt.predict(encodings)
#     labelIDs = np.unique(clt.labels_)
#     for labelID in labelIDs:
#         # if labelID<0:
#             # continue
#         idxs = np.where(clt.labels_ == labelID)[0]
#         idxs = np.random.choice(idxs, size=min(4, len(idxs)),
#     replace=False)
#         faces = []
#         for i in idxs:
#             image = cv2.imread(data[i]["imagePath"])
#             (top, right, bottom, left) = data[i]["loc"]
#             face = image[top:bottom, left:right]
#             face = cv2.resize(face, (96, 96))
#             faces.append(face)
#         montage = build_montages(faces, (96, 96), (2, 2))[0]
#         cv2.imwrite(f"./static/targetFaces/{labelID}.jpg",montage)


def getResult(labelToDetect):
    res = ""
    if "source" not in labelLoc[labelToDetect]:
        res = "Person only present in target video"
    elif "destin" not in labelLoc[labelToDetect]:
        res = "Person only present in source video"
    else:
        res = "preson present in both videos"
    
    idxs = labelIndex[labelToDetect]
    dir = './static/foundFaces'
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))
    idxs = np.random.choice(idxs, size=min(6, len(idxs)),
    replace=False)
    for i in idxs:
        image = cv2.imread(data[i]["imagePath"])
        (top, right, bottom, left) = data[i]["loc"]
        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.imwrite(f"./static/foundFaces/{i}.jpg",image)
    return res

def makedensityMap():
    dir = './static/faces'
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))
    global labels
    global data
    data = []
    imagePaths = list(paths.list_images("./videos/videoFrames"))
    for (i, imagePath) in enumerate(imagePaths):
        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model="hog")
        encodings = face_recognition.face_encodings(rgb, boxes)
        d = [{"imagePath": imagePath, "loc": box, "encoding": enc} for (box, enc) in zip(boxes, encodings)]
        data.extend(d)
    data = np.array(data)
    encodings = [d["encoding"] for d in data]
    print("encos")
    clt = DBSCAN(metric="euclidean", n_jobs=-1)
    clt.fit(encodings)
    labels = clt.labels_
    labelIDs = np.unique(labels)
    for labelID in labelIDs:
        if labelID<0:
            continue
        allIdxs = np.where(labels == labelID)[0]
        idxs = np.random.choice(allIdxs, size=min(25, len(allIdxs)),
    replace=False)
        faces = []
        for i in idxs:
            imgpath = data[i]["imagePath"]
            temp = imgpath.split("/")[-1]
            imgtype = temp.split("_")[0]
            if imgtype == "destin":
                labelLoc[labelID].add("destin")
            else:
                labelLoc[labelID].add("source")
            image = cv2.imread(data[i]["imagePath"])
            (top, right, bottom, left) = data[i]["loc"]
            face = image[top:bottom, left:right]
            face = cv2.resize(face, (96, 96))
            faces.append(face)
        labelIndex[labelID] = allIdxs
        montage = build_montages(faces, (96, 96), (5, 5))[0]
        cv2.imwrite(f"./static/faces/{labelID}.jpg",montage)

def createFaceFrames(address,destin):
    vid = cv2.VideoCapture(address)
    count = 0
    frameCount = 0
    while vid.isOpened():
        frameCount+=1
        success,img = vid.read()
        if not success:
            break
        if frameCount%30 != 0:
            continue
        print(f"imageNum{frameCount}")
        faces = face_cascade.detectMultiScale(img,1.3,4)
        if len(faces):
            count+=1
            cv2.imwrite(f"./videos/videoFrames/{destin}_{count}.jpg",img)
    return


def generate_frames(camera):

    while True:
        ## read the camera frame
        success,frame=camera.read()
        if not success:
            break
        else:
            faces = face_cascade.detectMultiScale(frame,1.3,4)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()

        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')




@app.route('/')
def home():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload():
    dir = './videos/videoFrames'
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))
    video = request.files['video']
    video.save("./videos/sourceVideo.mov")
    createFaceFrames("./videos/sourceVideo.mov","source")
    # Process the saved video

    # ...
    return redirect('/destin')


@app.route('/destin', methods=['POST'])
def destin_upload():
    video = request.files['video']
    video.save("./videos/destinVideo.mov")
    createFaceFrames("./videos/destinVideo.mov","destin")
    makedensityMap()
    # Process the saved video
    # ...
    return redirect('/faces')


@app.route('/faces')
def faces():
    # Assuming the faces are stored in the 'faces' directory
    faces_folder = './static/faces'
    faces = os.listdir(faces_folder)
    return render_template('faces.html', faces=faces)


@app.route('/target/<source>')
def target(source):
    global labelToDetect
    labelToDetect = int(source.split(".")[0])
    result = getResult(labelToDetect)
    folder_path = './static/foundFaces'
    file_list = os.listdir(folder_path)
    image_urls = [f'/static/foundfaces/{image}' for image in file_list]
    return render_template(f'found.html', image_urls=image_urls,res = result)

@app.route('/destin')
def destin():
    return render_template('target.html')


@app.route('/video')
def video():
    camera=cv2.VideoCapture(0)
    return Response(generate_frames(camera),mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == '__main__':
    app.run()

