from __future__ import division, print_function
import os
import numpy as np
from keras.preprocessing import image 
from keras.models import load_model
import tensorflow as tf
from keras.applications.vgg16 import preprocess_input
#global graph
#graph=tf.get_default_graph()

from flask import Flask, request, render_template
#from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

app = Flask(__name__)
model = load_model("Skin_Diseases_1.h5",compile=False)

@app.route('/')
def index():
    return render_template('base.html')

@app.route('/predict',methods = ['GET','POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        #print("current path")
        filename = file.filename
        #basepath = os.path.dirname(__file__)
        #print("current path", basepath)
        filepath = os.path.join('uploads',filename)
        #print("upload folder is ", filepath)
        file.save(filepath)
        
        img = image.load_img(filepath,target_size = (64,64))
        x = image.img_to_array(img)
        x = np.expand_dims(x,axis =0)
        img_data = preprocess_input(x)
        #with graph.as_default():
        pred = model.predict(img_data)
        l=np.amax(pred)
        l   
        f=pred.astype(int)
        f
        a1=np.array(f)
        print(a1)
        from functools import reduce
        single_list = reduce(lambda x,y: x+y, a1)
        print(single_list)
        val=np.where(single_list==1)
        dou_list = reduce(lambda x,y: x+y, val)
        print(dou_list)
        stringed = ''.join(map(str,val))
        stringed
        v=stringed[1]
        v1= int(v)
        v1
        #print("prediction",preds)
            
        index = ['Acne', 'Melanoma', 'Psoriasis', 'Rosacea', 'Vitiligo']
        
        text = "the predicted skin disease is : " + index[v1]
        return text
if __name__ == '__main__':
    app.run(debug = False, threaded = False)
 
        
    
    
    