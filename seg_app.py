import base64
import numpy as np
import io
import datetime
import pymongo
from PIL import Image
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras import models
from tensorflow.python.keras.preprocessing.image import img_to_array
from flask import Flask, jsonify, request
import cv2

tf.compat.v1.disable_eager_execution()

app =Flask('__name__')
save_model_path = 'C:/Users/yasin/OneDrive/Desktop/Bur/weights_5.hdf5'
def connect_db():
	client =pymongo.MongoClient('localhost', 27017)
	db =client["seg_logs_db"]
	col =db['logs']
	print(client.list_database_names())
	return col

def dice_coeff(y_true, y_pred):
  smooth = 1.
  y_true_f =tf.cast(tf.reshape(y_true, [-1]), tf.float32)
  y_pred_f =tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
  intersection = tf.reduce_sum(y_true_f * y_pred_f)
  score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
  
  return score

def mse_loss(y_true, y_pred):
  y_true =tf.cast(y_true, tf.float32)
  y_pred =tf.cast(y_pred, tf.float32)
  loss =tf.reduce_mean(tf.pow(tf.reshape(y_true, shape=[-1, 1]) - tf.reshape(y_pred, shape=[-1, 1]), 2)) * 100
  
  return loss


def get_model():
	global model
	global graph
	global sess
	sess = tf.compat.v1.Session()
	set_session(sess)
	model = models.load_model(save_model_path, custom_objects={'mse_loss': mse_loss,'dice_coeff': dice_coeff})
	model._make_predict_function()
	graph = tf.compat.v1.get_default_graph()
	print('model loaded')


def preproc_func(img, size):
	img =img.resize(size)
	img =img_to_array(img)*1/255.
	img =np.expand_dims(img, axis=0)
	return img

@app.route('/segment', methods=['POST'])

def segment():
	#print(request.remote_addr, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), request.method, request.url)
	message =request.get_json(force=True)
	encoded =message['img']
	decoded =base64.b64decode(encoded)
	img =Image.open(io.BytesIO(decoded))
	preproc_img =preproc_func(img, (256, 256))
	print('shape: ' +str(preproc_img.shape))
	doc ={'ip': request.remote_addr, 'datetime':datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'request':'%s %s'%(request.method, request.url), 'img': encoded}
	x =col.insert_one(doc)	
	global graph
	with graph.as_default():
		set_session(sess)
		pred_mask =model.predict(preproc_img, batch_size=1)[0]*255
	print(pred_mask.shape)
	mask_list =pred_mask.tolist()
	#print(mask_list[0][1][2])
	responce ={'mask': mask_list}

	return jsonify(responce)

if __name__ == "__main__":
	print ('loading model....')
	get_model()
	col =connect_db()
	print('connected to db')
	print('go to http://localhost:5000/static/segment.html')
	app.run(host='0.0.0.0', port=5000)

