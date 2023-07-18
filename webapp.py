"""
Simple app to upload an image via a web form 
and view the inference results on the image in the browser.
"""
import argparse
import io
import os
from PIL import Image
import datetime

import torch
from flask import Flask, render_template, request, redirect
from flask import jsonify
import json

app = Flask(__name__)

DATETIME_FORMAT = "%Y-%m-%d_%H-%M-%S-%f"


labels = "classes.txt"

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if not file:
            return

        img_bytes = file.read()
        

        img = Image.open(io.BytesIO(img_bytes))

        #yolov5 source code, custom, self trained model, local
        model = torch.hub.load('C:/Users/GYiji/OneDrive - Versuni/Desktop/yolov5-flask/yolov5', 'custom', 'C:/Users/GYiji/OneDrive - Versuni/Desktop/yolov5-flask/screen300_50.pt', source='local')
        model.eval()

        results = model([img])

        Label,Bbox,Confidence=[],[],[]
        classes=[]
        f=open(labels,'r')
        for line in f:
            classes.append(line.strip())

        for res in results.pandas().xyxy:
            for obj in range(len(res)):
                if res['confidence'][obj]>0.2:
                    (x1, y1, x2, y2) = (res['xmin'][obj],res['ymin'][obj],res['xmax'][obj],res['ymax'][obj])
                    bbox=(x1,y1,x2,y2)
                    className = classes[res['class'][obj]]
                    Label.append(className)
                    Bbox.append(bbox)
                    Confidence.append(res['confidence'][obj])
                    

        results.render()  # updates results.imgs with boxes and labels
        now_time = datetime.datetime.now().strftime(DATETIME_FORMAT)
        img_savename = f"static/{now_time}.jpg"
        Image.fromarray(results.ims[0]).save(img_savename)

        jsonfile = {'label':Label, 'bbox':Bbox, 'confidence':Confidence}
        file = json.dumps(jsonfile)
        with open(f"./json/data{now_time}.json", 'w') as f:
            json.dump(file, f)

        return redirect(img_savename)


    return render_template("index.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov5 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()

    #model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # force_reload = recache latest code
    #model.eval()
    app.run(host="0.0.0.0", port=args.port)  # debug=True causes Restarting with stat
