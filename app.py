from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import pandas as pd
import numpy as np
import tensorflow
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Flatten
from keras import metrics
from sklearn.metrics import accuracy_score
from tensorflow.keras.regularizers import l1
from tensorflow.keras.regularizers import l2
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.models import save_model
import joblib




app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER ='results'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULTS_FOLDER  

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'xlsx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    return render_template('index.html', final_mask=None)


@app.route('/predict', methods=['POST'])
def upload():
    try:
     if request.method == 'POST':
        dataset = request.files['file']
        dataset_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(dataset.filename))
        dataset.save(dataset_path)

        df=pd.read_csv(dataset_path)
        colstr=df.shape[1]-1
        X_train=df.iloc[:,0:colstr]
        Y_train=df.iloc[:,colstr]

        dynamic_inputs = request.form.getlist('dynamicInput[]')
        dynamic_dropdowns0 = request.form.getlist('dynamicDropdown0[]')        
        dynamic_dropdowns1 = request.form.getlist('dynamicDropdown1[]')
        dynamic_dropdowns2 = request.form.getlist('dynamicDropdown2[]')
        dynamic_dropdowns3 = request.form.getlist('dynamicDropdown3[]')

        dynamic_dropdownc1=request.form.getlist('dynamicDropdownc1[]')
        dynamic_dropdownc2=request.form.getlist('dynamicDropdownc2[]')
        dynamic_dropdownc3=request.form.getlist('dynamicDropdownc3[]')

        dynamic_inputsf1 = request.form.getlist('dynamicInputf1[]')
        dynamic_inputsf2 = request.form.getlist('dynamicInputf2[]')
        dynamic_inputsf3 = request.form.getlist('dynamicInputf3[]')
        dynamic_inputsf4 = request.form.getlist('dynamicInputf4[]')
        dynamic_dropdownf1 = request.form.getlist('dynamicDropdownf1[]')  



        templen=len(dynamic_dropdowns1)
        model=Sequential()
        for i in range(0,templen):
            input1=int(dynamic_inputs[i])
            act=dynamic_dropdowns0[i]
            kernalinit=dynamic_dropdowns1[i]
            biasinit=dynamic_dropdowns2[i]
            kernelregul=dynamic_dropdowns3[i]
            if 'L1' in kernelregul:
                kernelregul=l1(0.01)
            elif 'L2' in kernelregul:
                kernelregul=l2(0.01)
            elif 'L1_L2' in kernelregul:
                kernelregul=l1_l2(l1=0.01, l2=0.01)
            else:
                kernelregul=None
            model.add(Dense(input1,activation=act,kernel_initializer=kernalinit,bias_initializer=biasinit,kernel_regularizer=kernelregul))
            
        opt=dynamic_dropdownc1[0]
        los=dynamic_dropdownc2[0]
        met=dynamic_dropdownc3[0]
        print("Before model compilation")
        print(kernelregul,type(kernelregul))

        model.compile(loss=los,optimizer=opt,metrics=[met])
        print("After model compilation")

        epoch=int(dynamic_inputsf1[0])
        batch_siz=int(dynamic_inputsf2[0])
        initial_epoch=int(dynamic_inputsf3[0])
        stps_per_epoch=int(dynamic_inputsf4[0])
        shuffle=dynamic_dropdownf1[0]
        print("Before model fit")

        his=model.fit(X_train,Y_train,epochs=epoch,steps_per_epoch=stps_per_epoch,initial_epoch=initial_epoch,batch_size=batch_siz,shuffle=shuffle)
        print("After model fit")

        model.summary()
        wt_model=model.get_weights()
        model.save('results/Final_model.h5')
        joblib.dump(model, 'results/Final_model.pkl')
       
        result_text1 = "results/Final_model.h5"
        result_text2 = "results/Final_model.pkl"
    
        return jsonify(result_text1=result_text1,result_text2=result_text2)
    except Exception as e:
        print("An error occurred:", str(e))
        return jsonify(error="Internal Server Error"), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)