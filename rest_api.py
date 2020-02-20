from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.contrib import predictor
from skimage import io
import numpy as np
from preprocess import scale_to_fit

app = Flask(__name__)


@app.route('/api/recognize_image', methods=['POST'])
def recognize_image():
    img_url = request.get_json()['img_url']

    # prepare image for prediction
    image = io.imread(img_url, as_gray=True)
    image = scale_to_fit(image, **{'image_size': [256, 256]})
    image = (image / 255.) - 0.5
    image = image.astype(np.float32)
    image = np.expand_dims(image, axis=0)

    predictions = predict_fn({'image': image})
    cls = predictions['class']

    # prepare api response
    class_names = {0: 'Bad', 1: 'Good'}
    result = {
        "prediction": class_names[cls],
        "confidence": '{:2.0f}%'.format(100 * np.max(predictions['probabilities']))
    }

    return jsonify(isError=False, message="Success", statusCode=200, data=result), 200


if __name__ == '__main__':
    # based on https://github.com/hzitoun/tensorflow-2-image-classification-rest-api
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    predict_fn = predictor.from_saved_model('./models/saved_model/',
                                            config=config)
    app.run(debug=True, host='0.0.0.0')