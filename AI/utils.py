from operator import imod
import pandas as pd 
import numpy as np 
import os 
import random
import matplotlib.image as mimg
import cv2
import imgaug.augmenters as aug
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import seaborn as sns

MAX_SAMPLE_SIZE = 600
IMG_CSV_FILE_NAME = './data/data.csv'
NUM_BINS = 5

def split_data(inputs,outputs,percent=(0.8,0.1,0.1)):
    xTrain,x,yTrain,y = train_test_split(inputs,outputs,test_size=(1-percent[0]),random_state=5)
    xTest,xVal,yTest,yVal = train_test_split(x,y,test_size=(percent[2]/(percent[1]+percent[2])))
    return xTrain,yTrain,xVal,yVal,xTest,yTest
  
def print_results(model):
    fig = plt.figure()
    print('Plotting training results...')
    size = int(len(model.history['loss'])/model.num_epocs)
    for epoch in range(model.num_epocs):
        plt.plot(model.history['loss'][epoch*size:(epoch+1)*size],label=f'Training Loss epoch({epoch})')
    plt.plot(model.history['val_loss'],label='Validation Loss')
    plt.legend()
    plt.title('Plot of loss vs no. of epocs')
    plt.xlabel('No. of steps')
    fig.text(.8,.4,f"""
                        No. steps     : {model.n_steps}\n
                        No. epocs     : {model.num_epocs}\n
                        Batch Size    : {model.batch_size}\n
                        Learning Rate : {model.learning_rate}\n
                        Loss Function : Mean Squared Error Loss\n
                        Optimizer     : Adam Optimizer\n """)
    plt.ylabel('Loss')
    plt.show()


def importData(filename=IMG_CSV_FILE_NAME):
    print('Importing data...')
    #Importing CSV data and formatting image filename
    data = pd.read_csv(filename,index_col=0)
    data['env'] = data['env'].apply(lambda x : os.path.join(os.getcwd(),x))
    return data 

def plotData(data : pd.DataFrame,data_label=None):
    ax = sns.countplot(x="labels", hue='labels',data=data)
    ax.set_xlabel("Range of speed (in km/hr)")
    ax.set_ylabel("No. of samples")
    ax.set_title('Data Summary')
    ax.legend(loc='upper right')
    ax.plot([0,data['labels'].max()],[MAX_SAMPLE_SIZE,MAX_SAMPLE_SIZE],label='Max samples')
    ax.legend()

def preFormatData(data : pd.DataFrame,plot : bool = False,nBins=NUM_BINS):
    print('Formatting data...')
    
    # Object to encode the labels. 
    label_encoder = preprocessing.LabelEncoder()

    # Encode labels in column 'labels'. 
    data['labels']= label_encoder.fit_transform(data['labels'])
    
    #Getting bins
    _,bins = np.histogram(data['labels'],4)
    
    #Getting the interval
    dn = data['labels'].groupby(pd.cut(data['labels'], bins)).count()

    #Max no. of samples
    max_num_vals = dn.loc[dn.idxmax()]

    #Range of max samples
    interval = dn.idxmax()
    
    removable_items = []

    #Removing the items if the size is greater than the max size
    if max_num_vals > MAX_SAMPLE_SIZE:
            #Getting thee index of the items
            items = data.index[(interval.left < data['labels']) & (data['labels'] <= interval.right)].tolist()

            #Shuffling the removable data indices 
            random.shuffle(items)

            #Selecting the removable element indices
            removable_items = items[MAX_SAMPLE_SIZE:]

            #Dropping the rows
            data.drop(removable_items,inplace=True)

            #Resettingthe indices
            data.reset_index(inplace = True, drop = True)
    
    # Plotting the data
    if plot:
        plotData(data)
        
    print("Removed data size: ",len(removable_items))
    print("Remaining data size: ",len(data))
    
   
    # Separating the data into inputs and outputs
    inputs = data['env'].to_numpy().reshape(-1,1)
    
    # Performing one hot encoding. 
    outputs = pd.get_dummies(data, columns = ['labels']).iloc[:,2:].to_numpy()

    return inputs,outputs


    
#Function to make random image oprations
def augImage(img):

    #Panning the image
    if np.random.rand() <= 0.5:
        pan = aug.Affine(translate_percent={"x":(-0.1,0.1),"y":(-0.1,0.1)})
        img = pan.augment_image(img)

    #Zooming on the the region of interest
    if np.random.rand() <= 0.5:
        zoom = aug.Affine(scale=(1,1.2))
        img = zoom.augment_image(img)

    #Changing the brightness of the image
    if np.random.rand() <= 0.5:
        brightness = aug.Multiply((0.4,1.2))
        img = brightness.augment_image(img)

    return img

def preProcessImg(img):     
    #Cropping the image having only concentrated regions
    #img = img[200:,100:-100,:]

    #Changing color space from RGB to YUV
    img = cv2.cvtColor(img,cv2.COLOR_RGB2YUV)
    
    #Applying Blur
    img = cv2.GaussianBlur(img,(5,5),0)

    #Resizing the image
    img = cv2.resize(img,(200,66))

    #Normalizing the image
    img = img*(1/255)

    return img

class BatchGenerator:
    def __init__(self,inputs,outputs,batch_size=-1,train=True):
        self.total_size = len(inputs)
        self.inputs = inputs
        self.batch_size = batch_size if batch_size != -1 else self.total_size
        self.outputs=outputs
        self.train = train

        
        #Computing no. of batches to be returned
        if batch_size == -1:
            self.num_batches = 1
        else:
            self.num_batches = math.floor(self.total_size / self.batch_size)

    def __len__(self):
        return self.num_batches

    def generate(self):

        #Combining inputs and outputs
        data = np.column_stack((self.inputs,self.outputs))

        #Shuffling if required
        if self.train:
            random.shuffle(data)

        #Generating batches
        for batch_num in range(self.num_batches):
            curr_batch = data[batch_num*self.batch_size:(batch_num+1)*self.batch_size]
            images = []
            outputs = []
            
            for item in curr_batch:
                imgPath = item[0]
                output = item[1]
                img = mimg.imread(imgPath)

                #Augmenting if required
                #if self.train:
                #    img = augImage(img)
                    
                #Preprocessing the images
                img = preProcessImg(img)
                
                images.append(img)
                outputs.append(output)

            yield [np.asarray(images,dtype=np.float32),np.asarray(outputs,dtype=np.float32)]
    
    def __iter__(self):
        return self.generate()