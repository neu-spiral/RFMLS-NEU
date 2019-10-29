'''
This source code was developed under the DARPA Radio Frequency Machine 
Learning Systems (RFMLS) program contract N00164-18-R-WQ80. All the code 
released here is unclassified and the Government has unlimited rights 
to the code.
'''

import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import DataGenerators.NewDataGenerator as DG
import pickle,math
import numpy as np
from keras.models import load_model
from keras import backend as K
import numpy as np
import imageio
from keras.applications import ResNet50

def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    #print x[:,0,0]
    x -= x.mean()
    x /= x.std()
    #print x[:,0,0]
    x *= 0.1
    x += 0.5
    x = np.clip(x, 0, 1)
    # convert to RGB array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x    

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())

def get_insight(img_width, img_height, layer_name='conv1'):
    model = load_model('/home/zhou.fan1/research/trained/lv4_resnet.hdf5')
    input_img = model.input
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
    kept_filters = []
    for filter_index in range(4):
        print ("Extract layer: %s, filter:%d" % (layer_name, filter_index))
        layer_output = layer_dict[layer_name].output
        loss = K.mean(layer_output[:, :, :, filter_index])
        grads = K.gradients(loss, input_img)[0]
        grads = normalize(grads)
        iterate = K.function([input_img], [loss, grads])
        step = 1
        input_img_data = np.random.random((1, img_width, img_height, 3))
        input_img_data = (input_img_data - 0.5) * 20 + 128
        for i in range(100):
            loss_value, grads_value = iterate([input_img_data])
            input_img_data += grads_value * step
        img = deprocess_image(input_img_data[0])
        kept_filters.append((img, loss_value))
    return kept_filters 
            
def plot_insight(layer_name='conv1'):
    n = 2
    img_width = img_height = 64
    kept_filters = get_insight(img_width, img_height, layer_name)[:n * n]
    kept_filters.sort(key=lambda x: x[1], reverse=True)
    f, axarr = plt.subplots(n, n)
    for i in range(n):
        for j in range(n):
            img, loss = kept_filters[i * n + j]
            axarr[i, j].imshow(img)
            axarr[i, j].set_title('layer_name:%s loss:%.2f' % (layer_name, loss))
    plt.axis('off')
    plt.savefig("./tmp/%s_%d_%d" % (layer_name, n, n))
    
    
    
    
if __name__ == "__main__":
    base_path = "/scratch/RFMLS/dec18_darpa/v3_list/raw_samples/1Cv2/wifi/"
    stats_path = base_path

    # load dataset pickles
    file = open(base_path + "label.pkl",'rb')
    labels = pickle.load(file)
    file.close()

    file = open(stats_path + "device_ids.pkl", 'rb')
    device_ids = pickle.load(file)
    file.close()

    file = open(stats_path + "stats.pkl", 'rb')
    stats = pickle.load(file)
    file.close()

    file = open(base_path + "partition.pkl",'rb')
    partition = pickle.load(file)
    file.close()

    #extract training set
    ex_list = partition['train']

    file = open(stats_path + "ex_per_device.pkl", 'r')
    ex_per_device = pickle.load(file)
    file.close()
    max_num_ex_per_dev = max(ex_per_device.values())
    rep_time_per_device = {dev: math.floor(max_num_ex_per_dev / num) for dev,num in ex_per_device.items()}

    #generator = DG.IQPreprocessDataGenerator(ex_list, labels, device_ids, stats['avg_samples'] * len(ex_list), DG.IQTensorPreprocessor(), num_classes=len(device_ids), files_per_IO=4096, slice_size=64, batch_size=32, K=16, normalize=True, crop=1000)
    
    plot_insight()
    #plt.imshow(deprocess_image(generator.__getitem__(0)[0][0,:,:,:]))
    #plt.savefig('./tmp/test.png')
    
    
