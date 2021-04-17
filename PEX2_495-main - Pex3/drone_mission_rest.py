from flask import Flask, jsonify
import sys
import os
import glob
import json
from flask import send_file
from os import listdir
from os.path import isfile, join

app = Flask(__name__)
pathRoot = "/home/usafa/PycharmProjects/USAFA/Drone001/REST"
#pathRoot = "~REST/"

@app.route("/mission/getFrame/<frame>", methods=["GET"])
@app.route("/mission/getFrame/current", methods=["GET"])

def get_product_backlog():
    print("Hello")