from skimage import color, exposure, transform
import numpy as np
import glob
from skimage import io
import pickle
import os, sys
import pandas as pd
from skimage.filters import gaussian

ROOT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
IMG_SIZE = 32

def preprocess_img(img):
    # Histogram normalization in v channel
    hsv = color.rgb2hsv(img)
    hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
    img = color.hsv2rgb(hsv)

    # central square crop
    min_side = min(img.shape[:-1])
    centre = img.shape[0] // 2, img.shape[1] // 2
    img = img[centre[0] - min_side // 2:centre[0] + min_side // 2,
              centre[1] - min_side // 2:centre[1] + min_side // 2,
              :]
    
    # rescale to standard size
    img = transform.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # roll color axis to axis 0
    img = np.rollaxis(img, -1)
    return img

def get_class(img_path):
    return int(img_path.split('/')[-2])
    
def get_data(root=ROOT_DIR , name='GTSRB'):
    if not os.path.exists(os.path.join(root, name)):
        os.mkdir(os.path.join(root, name))
    # check for existing preprocessed data
    if 'x_train.pkl' in os.listdir(os.path.join(root, name)):
        load_pickle = True
        save_pickle = False
    else:
        load_pickle = False
        save_pickle = True

    data_dir = os.path.join(root, name)

    if not load_pickle:
        print('Pre-processed data is not found. Pre-processing ..')
        train_data_dir = os.path.join(root,'GTSRB_Final_Training_Images/GTSRB/Final_Training/Images')
        x_train = []
        y_train = []

        all_img_paths = glob.glob(os.path.join(train_data_dir, '*/*.ppm'))
        np.random.shuffle(all_img_paths)
        for img_path in all_img_paths:
            img = preprocess_img(io.imread(img_path))
            label = get_class(img_path)
            x_train.append(img)
            y_train.append(int(label))
            
        test_data_dir = os.path.join(root, 'GTSRB_Final_Test_Images/GTSRB/Final_Test/Images')
        #test = pd.read_csv(root + name + '/GT-final_test.csv', sep=';')
        test = pd.read_csv(os.path.join(root, 'GT-final_test.csv'), sep=';')
        # Load test dataset
        x_test = []
        y_test = []
        i = 0
        for file_name, class_id in zip(list(test['Filename']), list(test['ClassId'])):
            #img_path = os.path.join(root + name + '/Final_Test/Images/', file_name)
            img_path = os.path.join(test_data_dir, file_name)
            x_test.append(preprocess_img(io.imread(img_path)))
            y_test.append(int(class_id))

        x_test = np.array(x_test)
        y_test = np.array(y_test)

        x_train = np.array(x_train, dtype='float32')

#         data_dir = ROOT_DIR + name
#         file_name = os.path.join(data_dir, types + '.p')
#         with open(file_name, mode='rb') as f:
#             data = pickle.load(f)
#         if types == 'Training':
#             x_train, y_train = data['features'], data['labels']
#         elif types == 'Testing':
#             x_test, y_test = data['features'][:-30], data['labels'][:-30]
    else:
        print('Loading pre-processed data.')
        x_train = pickle.load(open(os.path.join(data_dir, 'x_train.pkl'), 'rb'))
        y_train = pickle.load(open(os.path.join(data_dir, 'y_train.pkl'), 'rb'))
        x_test = pickle.load(open(os.path.join(data_dir, 'x_test.pkl'), 'rb'))
        y_test = pickle.load(open(os.path.join(data_dir, 'y_test.pkl'), 'rb'))
        x_train = np.array(x_train, dtype='float32')

    y_train = np.reshape(y_train, [-1])
    y_test = np.reshape(y_test, [-1])
    
    if save_pickle:
        pickle.dump(x_train, open(os.path.join(data_dir, 'x_train.pkl'), 'wb'))
        pickle.dump(y_train, open(os.path.join(data_dir, 'y_train.pkl'), 'wb'))
        pickle.dump(x_test, open(os.path.join(data_dir, 'x_test.pkl'), 'wb'))
        pickle.dump(y_test, open(os.path.join(data_dir, 'y_test.pkl'), 'wb'))

    return x_train, y_train, x_test, y_test

if __name__ == '__main__':
    _, _, _, _ = get_data('data')
