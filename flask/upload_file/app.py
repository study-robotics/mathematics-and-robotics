from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import pykakasi
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import matplotlib.animation as animation
import cv2

import io
import base64
from IPython.display import HTML

app = Flask(__name__)

@app.route('/signup')
def sign_up():
    return render_template('signup.html')

# 日本語のファイル名を英語のアスキー文字に変換
class Kakashi:
    kakashi = pykakasi.kakasi()
    kakashi.setMode('H', 'a')
    kakashi.setMode('K', 'a')
    kakashi.setMode('J', 'a')
    conv = kakashi.getConverter()

    @classmethod
    def japanese_to_ascii(cls, japanese):
        return cls.conv.do(japanese)

# 「signup.htmlから取得した各変数」をクラスのインスタンス変数に格納
class UserInfo:
    def __init__(self, last_name, first_name, job, gender, message):
        self.last_name = last_name
        self.first_name = first_name
        self.job = job
        self.gender = gender
        self.message = message

@app.route('/home', methods=['GET', 'POST'])
def home():

    # request内に格納されている「signup.htmlから取得した各変数」を表示
    print(request.full_path)
    print(request.method)
    print(request.args)

    # GETの場合 ----------------
    # user_info = UserInfo(
    #     request.args.get('last_name'),
    #     request.args.get('first_name'),
    #     request.args.get('job'),
    #     request.args.get('gender'),
    #     request.args.get('message'),
    # )

    # POSTの場合 ----------------
    user_info = UserInfo(
        request.form.get('last_name'),
        request.form.get('first_name'),
        request.form.get('job'),
        request.form.get('gender'),
        request.form.get('message'),
    )
    return render_template('home.html', user_info=user_info)


@app.route('/upload', methods=["GET", "POST"])
def upload():
    if request.method == "GET":
        return render_template("upload.html")
    elif request.method == "POST":
        generate_cspace_files = request.files.getlist('generate_cspace_file') # upload.htmlにアップロードされたファイルを取得
        #ascii_filename = Kakashi.japanese_to_ascii(file.filename)
        #save_file_name = secure_filename(ascii_filename) # セキュリティ的に問題ないファイル名に変更
        #file.save(os.path.join("./static/image", save_file_name)) # ファイルの保存
        true_cspace_files = request.files.getlist('true_cspace_file')
        voxel_cspace_files = request.files.getlist("voxel_file")
        
        #files = [generate_cspace_files, true_cspace_files, voxel_files]

        #print('jijijijji', img_array)
        print("file length", len(generate_cspace_files))
        time_num = len(generate_cspace_files)

        #voxel = voxel_files[0].stream.read()
        #img_array = np.asarray(bytearray(voxel), dtype=np.uint8)
        #img = cv2.imdecode(img_array, 1)
        #cv2.imwrite('./hoge.jpg', img)


        ims = []
        fig = plt.figure(figsize=(10,4))
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        ax1.axis('off')
        ax2.axis('off')
        ax3.axis('off')
        plt.subplots_adjust(wspace=-0.18)
        for i in range(time_num):
            print(i)
            # generate C-space
            generate_cspace = generate_cspace_files[i].stream.read()
            generate_cspace_array = np.asarray(bytearray(generate_cspace), dtype=np.uint8)
            generate_cspace_img = cv2.imdecode(generate_cspace_array, 1)

            # true C-space
            true_cspace = true_cspace_files[i].stream.read()
            true_cspace_array = np.asarray(bytearray(true_cspace), dtype=np.uint8)
            true_cspace_img = cv2.imdecode(true_cspace_array, 1)

            # voxel 
            voxel_cspace_files[i].save('./hoge.npz')
            voxel_cspace_file = np.load('./hoge.npz', allow_pickle=True)
            voxel_cspace_file = voxel_cspace_file['arr_0']
            voxel = voxel_cspace_file[0]
            voxel_len = len(voxel)
            last_time_voxel = voxel[voxel_len-1]

            # "tmp voxel image" save ---
            tmp_fig = plt.figure(figsize=(8, 6))
            ax = tmp_fig.add_subplot(111, projection="3d")
            ax.voxels(last_time_voxel, edgecolor='k')
            plt.savefig('./tmp_vox.jpg')
            plt.clf()
            vox_img = cv2.imread('./tmp_vox.jpg')
            vox_img = cv2.cvtColor(vox_img, cv2.COLOR_BGR2RGB)
            # ----------------------------


            ims.append([ax1.imshow(generate_cspace_img), ax2.imshow(true_cspace_img), ax3.imshow(vox_img)])
            #ims.append([ax1.imshow(generate_cspace_img), ax2.imshow(true_cspace_img), ax3.voxels(last_time_voxel, edgecolor='k')])     

        ani = animation.ArtistAnimation(fig, ims, interval=300)
        #ani = animation.FuncAnimation(fig, update, fargs = (files,), frames = time_num, interval=100)
        ani.save('3501to4000.mp4', writer="ffmpeg",dpi=100)
        plt.clf()
        plt.close()
        #HTML(ani.to_jshtml())
        return redirect(url_for('uploaded_file', filename='hogehoge'))

@app.route('/uploaded_file/<string:filename>')
def uploaded_file(filename):
    return render_template('uploaded_file.html', filename=filename)




if __name__ == '__main__':
    app.run(debug=True)