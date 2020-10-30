import os
from flask import Flask, render_template, request
from tensorflow import keras
import cv2

app = Flask(__name__)
UPLOAD_FOLDER = "../Ideathon/Static"
CATEGORIES = ["Covid Positive", "Covid Negative"]


def prepare(filepath):
    IMG_SIZE = 244
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    p = new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    return p


@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            image_location = os.path.join(
                UPLOAD_FOLDER,
                image_file.filename,
            )
            image_file.save(image_location)
            print(image_location)
            p = model.predict([prepare(image_location)])
            return render_template("index.html", prediction=CATEGORIES[int(p[0, 0])])
    return render_template("index.html", prediction=0)


if __name__ == "__main__":
    model = keras.models.load_model("../Ideathon/Model/King.h5")
    app.run(port=12000, debug=True)
