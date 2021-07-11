import u2net_load
import u2net_run
from predict_pose import generate_pose_keypoints
from flask import Flask, request, jsonify
from flask_ngrok import run_with_ngrok
import os
import io
from werkzeug.utils import secure_filename
from PIL import Image
import json
import pathlib
import sys
sys.path.insert(0, 'U-2-Net')
sys.path.insert(0, 'Self-Correction-Human-Parsing-for-ACGPN')
sys.path.insert(0, 'ACGPN')
app = Flask(__name__)
run_with_ngrok(app)
# app.config["DEBUG"] = True
app.config['CLOTH_DIR'] = 'inputs/cloth'
app.config['IMG_DIR'] = 'inputs/img'


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
    import os
    img_name = 'userImage.jpg'
    img_path = os.path.join('inputs/img', img_name)  # .replace(".png",".jpg")
    img = Image.open(img_path)
    img = img.resize((192, 256), Image.BICUBIC)
    img_path = os.path.join('ACGPN/Data_preprocessing/test_img', img_name)
    img.save(img_path)

    """
    TODO: Next line gives cuda error on cpu 
    TODO: Remove comment from next line while testing on gpu colab
    """
    os.system("python Self-Correction-Human-Parsing-for-ACGPN/simple_extractor.py --dataset 'lip' --model-restore 'Self-Correction-Human-Parsing-for-ACGPN/lip_final.pth' --input-dir 'ACGPN/Data_preprocessing/test_img' --output-dir 'ACGPN/Data_preprocessing/test_label'")

    pose_path = os.path.join(
        'ACGPN/Data_preprocessing/test_pose', img_name.replace('.jpg', '_keypoints.json'))
    # TODO: next line gives some unkown error.
    # TODO: Check after removeing all previous errors
    generate_pose_keypoints(img_path, pose_path)

    with open('ACGPN/Data_preprocessing/test_pairs.txt', 'w') as f:
        inference_name = img_name+" "+cloth_name
        f.write(inference_name)

    os.system('python ACGPN/test.py')

    return(jsonify({"Success": "Correct keys"}))


@app.route('/api/dress_the_user2_0', methods=['POST'])
def dress_the_user_2.0():
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
            return(jsonify({"Status": "Invalid keys"}))
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
    import os
    img_name = 'userImage.jpg'
    img_path = os.path.join('inputs/img', img_name)  # .replace(".png",".jpg")
    img = Image.open(img_path)
    img = img.resize((192, 256), Image.BICUBIC)
    img_path = os.path.join('ACGPN/Data_preprocessing/test_img', img_name)
    img.save(img_path)

    """
    TODO: Next line gives cuda error on cpu 
    TODO: Remove comment from next line while testing on gpu colab
    """
    os.system("python Self-Correction-Human-Parsing-for-ACGPN/simple_extractor.py --dataset 'lip' --model-restore 'Self-Correction-Human-Parsing-for-ACGPN/lip_final.pth' --input-dir 'ACGPN/Data_preprocessing/test_img' --output-dir 'ACGPN/Data_preprocessing/test_label'")

    pose_path = os.path.join(
        'ACGPN/Data_preprocessing/test_pose', img_name.replace('.jpg', '_keypoints.json'))
    # TODO: next line gives some unkown error.
    # TODO: Check after removeing all previous errors
    generate_pose_keypoints(img_path, pose_path)

    with open('ACGPN/Data_preprocessing/test_pairs.txt', 'w') as f:
        inference_name = img_name+" "+cloth_name
        f.write(inference_name)

    output_dir = os.path.join(opt.results_dir, opt.phase)
    try:
        os.rmdir(output_dir)
    except OSError as error:
        print("os Error")
        pass
    os.system('python ACGPN/test.py')

    output_dir = os.path.join(opt.results_dir, opt.phase)
    fake_image_dir = os.path.join(output_dir, 'try-on')
    filepath = os.path.join(fake_image_dir,  'userImage.jpg')

    img = Image.open(filepath, mode='r')
    rawBytes = io.BytesIO()
    img.save(rawBytes, "JPEG")
    img_base64 = base64.b64encode(rawBytes.getvalue())
    # return jsonify({'image': img_base64.decode('ascii')})

    return(jsonify({"Status": "Success",
                    "Image": img_base64.decode('ascii')}))


if __name__ == '__main__':
    app.debug = True
    app.run()
