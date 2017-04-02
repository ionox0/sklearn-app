import os
import json
from flask import Flask
from flask import request
from meta_classify import do_classifications
from utils import strtobool


app = Flask(__name__, static_url_path='', static_folder='.')



@app.route('/classify', methods=['POST'])
def classify_route():
    """
    Flask route for scikit-learn classification
    :return: Json summary response
    """
    if 'training_file' not in request.files:
        return 'No training file provided'

    train_csv = request.files['training_file']

    if train_csv.filename == '':
        return 'Train filename missing'

    src = os.getcwd() + '/uploaded_data/'
    train_csv.save(os.path.join(src, train_csv.filename))

    train_label_column_name = request.form['train_label_column_name']
    classifiers_selected = json.loads(request.form['classifiers_selected'])
    scalers_selected = json.loads(request.form['scalers_selected'])

    print(classifiers_selected, scalers_selected, train_label_column_name)

    results = do_classifications(train_data_file=train_csv.filename,
                                 label_column_name=train_label_column_name,
                                 classifiers=classifiers_selected,
                                 scalers=scalers_selected)

    return json.dumps(results)


if __name__ == '__main__':
    env_port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=env_port, debug=True, threaded=True)
