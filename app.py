import os

from flask import Flask, render_template, request, redirect

from inference import get_detection, get_recognition

app = Flask(__name__)

basedir = os.path.abspath(os.path.dirname(__file__))


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return
        path = basedir + "/static/data/"
        file_path = path + file.filename
        file.save(file_path)
        img_bytes = file.read()
        roi = get_detection(file_path, file.filename)
        water_reading = get_recognition(roi)
        return render_template('result.html', filename=file.filename, water_reading=water_reading)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get('PORT', 5000)))
