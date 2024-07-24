from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
from detect_animal import c_to_python

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    filename = secure_filename(file.filename)
    filepath = os.path.join('', filename)
    file.save(filepath)

    # 여기서 이미지 처리를 수행하고 JSON 응답을 반환합니다.
    result = c_to_python(filepath)  
    response = f'{result}'.replace('\"', '')
    return response, 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)