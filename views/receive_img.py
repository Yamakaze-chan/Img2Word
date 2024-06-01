from flask import Flask, render_template, request
import os
import io
from PIL import Image
import threading
# import webbrowser
import socket
import random
# import flet as ft
import qrcode
import sys

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

app = Flask(__name__, static_folder=resource_path(r"views/static"), template_folder=resource_path(r"views/templates"))
@app.route("/", methods=["GET", "POST"]) 
def upload_image():
    if request.method == "POST": 
        file = request.files.get("file") 
        file_content = file.read() 
        # check if file loaded successfully or not 
        if file_content: 
            if not os.path.isdir(resource_path(r'upload_img')):
                os.makedirs(resource_path(r'upload_img'), exist_ok=False)
            Image.open(io.BytesIO(file_content)).save(os.path.join(resource_path(r"upload_img"),str(file.filename)))
            for (dirpath, dirnames, filenames) in os.walk(resource_path(r'upload_img')):
                filenames.sort(key=lambda item: os.path.getctime(os.path.join(dirpath, item)), reverse=True)
                if len(filenames) == 100:
                    os.remove(os.path.join(dirpath, filenames[-1]))
            return render_template(r'flask.html')
        else: 
            return "Có lỗi xảy ra!!!!"
    return render_template(r'flask.html')

def get_QR_code_from_URL(url):
    qr = qrcode.QRCode(version=3, box_size=20, border=10, error_correction=qrcode.constants.ERROR_CORRECT_H)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    img.save(resource_path(r"views/latest_qr_code/qr.png"))
def run_web_upload_image():
        i = 0
        ip_host = socket.gethostbyname(socket.getfqdn())
        while i < 100:
            try:
                port = random.randint(2000, 6000)
                url = "http://{0}:{1}".format(ip_host,port)
                get_QR_code_from_URL(url)
                # threading.Timer(1.25, lambda: webbrowser.open(url) ).start() #run web on computer
                threading.Thread(target=lambda: app.run(host=ip_host, port=port, use_reloader=False)).start()
                return url
            except Exception as e:
                print(e)
            i = i + 1
        if i == 100:
            print("Something went wrong")