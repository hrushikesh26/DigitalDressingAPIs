
import sys
sys.path.insert(0, 'U-2-Net')
sys.path.insert(0, 'Self-Correction-Human-Parsing-for-ACGPN')
sys.path.insert(0, 'ACGPN')
from predict_pose import generate_pose_keypoints
from flask import Flask, request, jsonify
from flask_ngrok import run_with_ngrok
import os
import io
from werkzeug.utils import secure_filename
from PIL import Image
import json
import pathlib
import u2net_load
import datetime
# import firebase_admin
# from firebase_admin import credentials
# from firebase_admin import storage
from firebase import Firebase
import test
from flask_cors import CORS

def delete_folder(pth) :
    for sub in pth.iterdir() :
        if sub.is_dir() :
            delete_folder(sub)
        else :
            sub.unlink()


from ACGPN.options.test_options import TestOptions
import u2net_run
app = Flask(__name__)

CORS(app)

run_with_ngrok(app)
# app.config["DEBUG"] = True
app.config['CLOTH_DIR'] = 'inputs/cloth'
app.config['IMG_DIR'] = 'inputs/img'
firebaseConfig = {
    "apiKey": "AIzaSyC1fZtfInnwm0yf93g46_50Tv9hX7v0dG4",
    "authDomain": "oose-d997d.firebaseapp.com",
    "databaseURL": "https://oose-d997d-default-rtdb.firebaseio.com",
    "projectId": "oose-d997d",
    "storageBucket": "oose-d997d.appspot.com",
    "messagingSenderId": "882019773220",
    "appId": "1:882019773220:web:f59ae8a9fe3b8c990ce1b1",
    "measurementId": "G-5XKXV8MBH9",
    "serviceAccount": "oose-d997d-firebase-adminsdk-mv4cw-6965bee3db.json"
  }
opt = TestOptions().parse()
firebase = Firebase(firebaseConfig)
storage = firebase.storage()

@app.route('/api', methods=['GET'])
def home():
    return "Welcome to DigitalDressingApIs"


@app.route('/api/dress_the_user', methods=['POST'])
def dress_the_user():
    """ 
    *save files in "input" directory
    """
    files_dict = request.files.to_dict()
    for key in files_dict.keys():
        if(key == "user"):
            file = files_dict[key]
            filename = secure_filename(file.filename)
            pathlib.Path(app.config['IMG_DIR']).mkdir(
                parents=True, exist_ok=True)
            file.save(pathlib.Path(
                app.config['IMG_DIR']).joinpath('userImage.jpg'))
        elif(key == "dress"):
            file = files_dict[key]
            filename = secure_filename(file.filename)
            pathlib.Path(app.config['CLOTH_DIR']).mkdir(
                parents=True, exist_ok=True)
            file.save(pathlib.Path(
                app.config['CLOTH_DIR']).joinpath('cloth.jpg'))
        else:
            print(key)
            return(jsonify({"Error": "Invalid keys"}))
    # create required directories
    pathlib.Path(
        'ACGPN/Data_preprocessing/test_color').mkdir(parents=True, exist_ok=True)
    pathlib.Path(
        'ACGPN/Data_preprocessing/test_img').mkdir(parents=True, exist_ok=True)
    pathlib.Path(
        'ACGPN/Data_preprocessing/test_pose').mkdir(parents=True, exist_ok=True)
    pathlib.Path(
        'ACGPN/Data_preprocessing/test_colormask').mkdir(parents=True, exist_ok=True)
    pathlib.Path(
        'ACGPN/Data_preprocessing/test_label').mkdir(parents=True, exist_ok=True)
    pathlib.Path(
        'ACGPN/Data_preprocessing/test_edge').mkdir(parents=True, exist_ok=True)
    pathlib.Path(
        'ACGPN/Data_preprocessing/test_mask').mkdir(parents=True, exist_ok=True)

    """
    * Save dress image and create test_edge
    """
    cloth_name = 'cloth.jpg'
    cloth_path = pathlib.Path(app.config['CLOTH_DIR']).joinpath(cloth_name)
    cloth = Image.open(cloth_path)
    cloth = cloth.resize((192, 256), Image.BICUBIC).convert('RGB')
    cloth.save(pathlib.Path(
        'ACGPN/Data_preprocessing/test_color').joinpath(cloth_name))

    """
    TODO Below part has been commented for testing without GPU
    TODO Uncomment it before final test
    """
    u2net = u2net_load.model(model_name='u2netp')
    u2net_run.infer(u2net, 'ACGPN/Data_preprocessing/test_color',
                    'ACGPN/Data_preprocessing/test_edge')
    """
    * Save user image and create test_pose and test_label
    """
    # import os
    img_name = 'userImage.jpg'
    img_path = os.path.join('inputs/img', img_name)  # .replace(".png",".jpg")
    img = Image.open(img_path)
    img = img.resize((192, 256), Image.BICUBIC)
    img_path = os.path.join('ACGPN/Data_preprocessing/test_img', img_name)
    img.save(img_path)

    """
    TODO: Next line gives cuda error on cpu  ==> DONE
    TODO: Remove comment from next line while testing on gpu colab ==> DONE
    """
    os.system("python Self-Correction-Human-Parsing-for-ACGPN/simple_extractor.py --dataset 'lip' --model-restore 'Self-Correction-Human-Parsing-for-ACGPN/lip_final.pth' --input-dir 'ACGPN/Data_preprocessing/test_img' --output-dir 'ACGPN/Data_preprocessing/test_label'")

    pose_path = os.path.join(
        'ACGPN/Data_preprocessing/test_pose', img_name.replace('.jpg', '_keypoints.json'))
    # TODO: next line gives some unkown error.  ==> DONE
    # TODO: Check after removeing all previous errors ==> DONE
    generate_pose_keypoints(img_path, pose_path)

    with open('ACGPN/Data_preprocessing/test_pairs.txt', 'w') as f:
        inference_name = img_name+" "+cloth_name
        f.write(inference_name)

    os.system('python ACGPN/test.py')

    return(jsonify({"Success": "Correct keys"}))



@app.route('/api/generate_poseandlabel',methods=['GET'])
def generate_poseandlabel():
    uid = request.args['uid'] 

    # *Download image from firebase at IMG_DIR
    # firebase_storage = pyrebase.initialize_app(firebaseConfig)
    # storage = firebase_storage.storage()
    pathlib.Path(app.config['IMG_DIR']).mkdir(parents=True, exist_ok=True)
    filepath = os.path.join(app.config['IMG_DIR'],uid+".jpg")
    storage.child("user_images/"+uid+".jpg").download(filepath)
    # *create pose and label directories if not present
    pathlib.Path(
        'ACGPN/Data_preprocessing/test_pose').mkdir(parents=True, exist_ok=True)
    pathlib.Path(
        'ACGPN/Data_preprocessing/test_label').mkdir(parents=True, exist_ok=True)
    # * Resize and copy image in test_img
    
    pathlib.Path(
        'ACGPN/Data_preprocessing/test_img').mkdir(parents=True, exist_ok=True)
    img_name = uid+'.jpg'
    img_path = os.path.join(app.config['IMG_DIR'], img_name)  # .replace(".png",".jpg")
    img = Image.open(img_path)
    img = img.resize((192, 256), Image.BICUBIC)
    img_path = os.path.join('ACGPN/Data_preprocessing/test_img', img_name)
    img.save(img_path)

    # * Generate label test_label
    os.system("python Self-Correction-Human-Parsing-for-ACGPN/simple_extractor.py --dataset 'lip' --model-restore 'Self-Correction-Human-Parsing-for-ACGPN/lip_final.pth' --input-dir 'ACGPN/Data_preprocessing/test_img' --output-dir 'ACGPN/Data_preprocessing/test_label'")
    # * Generate pose and save at test_pose
    poseFilename = img_name.replace('.jpg', '_keypoints.json')
    pose_path = os.path.join(
        'ACGPN/Data_preprocessing/test_pose', poseFilename)
    generate_pose_keypoints(img_path, pose_path)

    # *upload pose and label to firebase
    poseFilepath = os.path.join("ACGPN/Data_preprocessing/test_pose/",poseFilename)
    storage.child('user_pose/'+poseFilename).put(poseFilepath)

    labelFilename= uid+".png"
    labelFilepath = os.path.join("ACGPN/Data_preprocessing/test_label/",labelFilename)
    storage.child('user_label/'+labelFilename).put(labelFilepath)


    return jsonify({"Status": "Success"})


@app.route('/api/tryon')
def tryon():
    uid = request.args['uid'] 
    clothname = request.args['clothname'] # ! In name.jpg format
    user_imagename = uid+".jpg"
    # *Download image from firebase at test_cloth

    #*download cloth image
    file_dir = "ACGPN/Data_preprocessing/test_color/"
    pathlib.Path(file_dir).mkdir(parents=True, exist_ok=True)
    delete_folder_contents(pathlib.Path(file_dir))
    filepath = os.path.join(file_dir,clothname)
    storage.child("cloth_images/"+clothname).download(filepath)

    #*download user image
    file_dir = "ACGPN/Data_preprocessing/test_img/"
    pathlib.Path(file_dir).mkdir(parents=True, exist_ok=True)
    delete_folder_contents(pathlib.Path(file_dir))
    filepath = os.path.join(file_dir,user_imagename)
    storage.child("user_images/"+user_imagename).download(filepath)

    # * Download cloth edge
    file_dir = "ACGPN/Data_preprocessing/test_edge/"
    pathlib.Path(file_dir).mkdir(parents=True, exist_ok=True)
    delete_folder_contents(pathlib.Path(file_dir))
    filepath = os.path.join(file_dir,clothname.replace(".jpg",".png"))
    storage.child("cloth_edges/"+clothname.replace(".jpg",".png") ).download(filepath)

    # * Download user pose
    file_dir = "ACGPN/Data_preprocessing/test_pose/"
    pathlib.Path(file_dir).mkdir(parents=True, exist_ok=True)
    delete_folder_contents(pathlib.Path(file_dir))
    filepath = os.path.join(file_dir,user_imagename.replace(".jpg","_keypoints.json"))
    storage.child("user_pose/"+user_imagename.replace(".jpg","_keypoints.json")).download(filepath)

    # * Download user label
    file_dir = "ACGPN/Data_preprocessing/test_label/"
    pathlib.Path(file_dir).mkdir(parents=True, exist_ok=True)
    delete_folder_contents(pathlib.Path(file_dir))
    filepath = os.path.join(file_dir,user_imagename.replace(".jpg",".png"))
    storage.child("user_label/"+user_imagename.replace(".jpg",".png")).download(filepath)
    
    
    file_dir = "ACGPN/Data_preprocessing/test_mask/"
    pathlib.Path(file_dir).mkdir(parents=True, exist_ok=True)
    file_dir = "ACGPN/Data_preprocessing/test_colormask/"
    pathlib.Path(file_dir).mkdir(parents=True, exist_ok=True)

    # * Write test_pairs.txt file
    with open('ACGPN/Data_preprocessing/test_pairs.txt', 'w') as f:
        inference_name = user_imagename+" "+clothname
        f.write(inference_name)

    os.system('python ACGPN/test.py')

    result_img_path = "results/test/try-on/"+user_imagename
    cloud_image_path = "tryon_results/"+user_imagename.replace(".jpg","")+clothname
    storage.child(cloud_image_path).put(result_img_path)

    return jsonify({"Status": "Success"})

@app.route('/api/make_all_cloth_edges')
def make_all_cloth_edges():    
    u2net = u2net_load.model(model_name='u2netp')
    u2net_run.infer(u2net, 'ACGPN/Data_preprocessing/test_color',
                    'ACGPN/Data_preprocessing/test_edge')    
    cloth_edge_arr = os.listdir('ACGPN/Data_preprocessing/test_edge')
    for file in cloth_edge_arr:
        print("Uploading..."+file)
        storage.child("cloth_edges/"+file).put('ACGPN/Data_preprocessing/test_edge/'+file)
    
    return jsonify({"Status":"Success"})


if __name__ == '__main__':
    app.debug = True
    app.run()
