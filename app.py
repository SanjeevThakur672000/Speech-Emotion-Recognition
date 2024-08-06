from flask import Flask, request, render_template, redirect, url_for
import pickle
from tensorflow.keras.models import model_from_json
import librosa
import numpy as np

app = Flask(__name__)

# Load the scaler
with open('scaler2.pickle', 'rb') as f:
    scaler2 = pickle.load(f)

# Load the encoder
with open('encoder2.pickle', 'rb') as f:
    encoder2 = pickle.load(f)

# Load the model architecture and weights
json_file = open('CNN_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("best_model1_weights.h5")

def zcr(data, frame_length, hop_length):
    zcr = librosa.feature.zero_crossing_rate(data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(zcr)

def rmse(data, frame_length=2048, hop_length=512):
    rmse = librosa.feature.rms(data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(rmse)

def mfcc(data, sr, frame_length=2048, hop_length=512, flatten=True):
    mfcc = librosa.feature.mfcc(data, sr=sr)
    return np.squeeze(mfcc.T) if not flatten else np.ravel(mfcc.T)

def extract_features(data, sr=22050, frame_length=2048, hop_length=512):
    result = np.array([])
    result = np.hstack((result,
                        zcr(data, frame_length, hop_length),
                        rmse(data, frame_length, hop_length),
                        mfcc(data, sr, frame_length, hop_length)
                        ))
    return result

def get_predict_feat(path):
    d, s_rate = librosa.load(path, duration=2.5, offset=0.6)
    res = extract_features(d)
    print("Shape of extracted features:", res.shape)
    result = np.array(res)
    print("Size of result before reshape:", result.size)
    
    # Modify the reshape operation to be more flexible
    if result.size < 2376:
        result = np.pad(result, (0, 2376 - result.size), mode='constant')
    elif result.size > 2376:
        result = result[:2376]
    
    result = result.reshape(1, -1)
    print("Shape of result after reshape:", result.shape)
    
    i_result = scaler2.transform(result)
    final_result = np.expand_dims(i_result, axis=2)
    return final_result

emotions1 = {1: 'Neutral', 2: 'Calm', 3: 'Happy', 4: 'Sad', 5: 'Angry', 6: 'Fear', 7: 'Disgust', 8: 'Surprise'}

def prediction(path1):
    res = get_predict_feat(path1)
    predictions = loaded_model.predict(res)
    y_pred = encoder2.inverse_transform(predictions)
    return y_pred[0][0]

@app.route('/')
def upload_file():
    return render_template('index.html')

@app.route('/predict_audio', methods=['POST'])
def predict_audio():
    if 'audio_data' not in request.files:
        return redirect(request.url)
    
    file = request.files['audio_data']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        file_path = 'uploaded_audio.wav'
        file.save(file_path)
        
        predicted_emotion = prediction(file_path)
        
        return render_template('index.html', prediction_text=f"Predicted Emotion: {predicted_emotion}")

if __name__ == "__main__":
    app.run(debug=True)