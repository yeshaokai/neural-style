import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from time import time
import glob as glob

data = {}
def read_response(name):
    return np.load(name)
def process_data():
    for k,v in data.items():
        v = v[0,:,:,:]
        v = v.reshape(-1,v.shape[2])
        v = np.transpose(v,(1,0))

        data[k] = v
def read_npy_files():

    files = glob.glob('*.npy')
    for file in files:
        name = file[:-4]
        data[name] = read_response(file)    
    process_data()
def get_cluster_data():
    # assign colors
    colors = []
    datas = []
    count = 0
    for k,v in data.items():
        classes = np.zeros(v.shape[0])
        classes+=count
        #print classes.shape
        colors.append(classes)

        datas.append(v)
        count+=1
        

    colors = np.vstack(colors)
    datas = np.vstack(datas)
    return colors,datas
n_iter = 5000
read_npy_files()
colors,cluster_X = get_cluster_data()

#print cluster_X.shape
colors = colors.reshape(-1)
print colors.shape
print cluster_X.shape
for i in [2, 5, 30, 50, 100]:
    t0 = time()
    model = TSNE(n_components=2, n_iter = n_iter,random_state=0, perplexity =i)
    np.set_printoptions(suppress=True)
    Y = model.fit_transform(cluster_X)
    t1 =time()    
    print( "t-SNE: %.2g sec" % (t1 -t0))
    plt.scatter(Y[:, 0], Y[:, 1], c= colors)
    plt.title('t-SNE with perplexity = {}'.format(i))
    plt.show()    
