from bgFeatureExtraction import bgExtractor
import joblib
import tensorflow as tf
from tensorflow import keras
import cv2
import os
import numpy as np
from sklearn.preprocessing import StandardScaler

class Predictor():
    
    def __init__(self) -> None:
        print(os.path.isfile('webapp/models/model_color.pkl'))
        self.color_model = joblib.load('webapp/models/model_color.pkl')
        self.transfer_model = keras.models.load_model('webapp/models/model_transfer160.h5', compile=False)
        self.transfer_model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001))
        self.scaler = joblib.load('webapp/models/std_scaler.bin')
        self.color_model_ratio = 0.1
        self.transfer_model_ratio = 0.6

    def color_model_predict(self, img):
        preprocessed, _ = bgExtractor.run(img)
        X = np.array([col_v for col in preprocessed for col_v in col]).reshape(1,-1)
        X = self.scaler.transform(X)
        prediction = self.color_model.predict(X)[0]
        return prediction

    def transfer_model_predict(self, img):
        print(np.shape(img))
        img_reshaped = np.array([cv2.resize(img, (160,160), interpolation = cv2.INTER_AREA)])
        with tf.device('/cpu:0'):
            prediction = self.transfer_model.predict(img_reshaped)[0]
        return prediction

    def predict(self, img):
        return (self.color_model_predict(img)*self.color_model_ratio + self.transfer_model_predict(img)*self.transfer_model_ratio)/(self.color_model_ratio + self.transfer_model_ratio)

predictor = Predictor()