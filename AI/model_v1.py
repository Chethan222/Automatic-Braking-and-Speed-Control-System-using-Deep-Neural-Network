import tensorflow as tf
from AI.utils import *
import torch 
import torch.nn.functional as F
import torch.nn as nn
import os
import numpy as np
import torch
from torch.autograd import Variable
import time
from time import sleep
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix
import seaborn as sns
from torch.utils.data import DataLoader,Dataset,WeightedRandomSampler
import torchvision
sns.set()
torch.set_grad_enabled(True)



if torch.cuda.is_available():
    print('CUDA is available.')
    print('Using GPU for computation.')
else:
    print('CUDA is not available!')
    print('Using CPU for computation.')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        # [(W−K+2P)/S] + 1
        # size =2,stride = 2 (shift kernal to 2px)
        # W is the input volume(),K is the Kernel size(5),P is the padding (0),S is the stride (2)
        # input_shape = (66,200,3)
        self.conv1 = nn.Conv2d(3,24,kernel_size=(5,5),stride=(2,2)) #O/p 31*98*24
        self.conv2 = nn.Conv2d(24,36,kernel_size=(5,5),stride=(2,2)) #O/p 14*47*36
        self.conv3 = nn.Conv2d(36,48,kernel_size=(5,5),stride=(2,2)) #O/p 5*22*48
        self.conv4= nn.Conv2d(48,64,kernel_size=(3,3),stride=(1,1)) #O/p 3*20*64
        self.conv5 = nn.Conv2d(64,64,kernel_size=(3,3),stride=(1,1)) #O/p 1*18*64
        
        self.fc1 = nn.Linear(18*64,100) # Flattening -> 1*18*64
        self.fc2 = nn.Linear(100,50)
        self.fc3 = nn.Linear(50,5)

    def config_model(self,n_steps,num_epocs=4,batch_size=10,learning_rate=0.0001):
        #Model Hyperparameters
        self.num_epocs = num_epocs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.loss_func = nn.CrossEntropyLoss()
        self.n_steps = n_steps
        self.optimizer = torch.optim.SGD(self.parameters(),lr=self.learning_rate)

        print("#"*100)
        print('***************** Model hyperparameters ********************\n')
        print(f"\t\t\tNo. steps     : {self.n_steps}\n\
                        No. epocs     : {self.num_epocs}\n\
                        Batch Size    : {self.batch_size}\n\
                        Learning Rate : {self.learning_rate}\n\
                        Loss Function : Cross Entropy Loss\n\
                        Optimizer     : Adam Optimizer\n ")
        print("#"*100)

    def forward(self,x):
        x = F.elu(self.conv1(x)) # First convolution layer
        x = F.elu(self.conv2(x)) # Second convolution layer
        x = F.elu(self.conv3(x)) # Third convolution layer
        x = F.elu(self.conv4(x)) # Fourth convolution layer
        x = F.elu(self.conv5(x)) # Fifth convolution layer
        x = x.reshape(-1,18*64)  # Flattening
        x = F.elu(self.fc1(x))  # Firts FCL
        x = F.elu(self.fc2(x))  # Second FCL
        x = self.fc3(x)  # Third FCL
        
        return x
    
    def train(self,train_loader,validation_loader,device=device,validate_for_steps=5):
        self.device = device
        print('Training model...')
        self.to(device)
        self.history =  {'loss':[],'val_loss':[]}
        val_loss = 0
        self.st_time = time.time()
        self.steps = len(train_loader)

        #Training loop
        for epoch in range(self.num_epocs):
            for step,(inputs,labels) in enumerate(train_loader):
                inputs = Variable(inputs.float().to(device),requires_grad=True)
                labels = Variable(labels.float().to(device))
                
                
                # Forward pass
                outputs = self.forward(inputs)
                
                predictions_softmax = torch.log_softmax(outputs, dim = 1)
                predictions = torch.max(predictions_softmax, dim = 1).values.view(-1,1)
                preds = torch.max(predictions_softmax, dim = 1).indices.view(-1,1).float()
                
                # Empty the values in grad
                self.optimizer.zero_grad()
                train_loss = self.loss_func(predictions,labels)
                
                correct = (preds == labels).float()
    
                accuracy = correct.sum() / len(labels)
                accuracy = torch.round(accuracy * 100)
                
                self.history['loss'].append(train_loss.item())

                # Backward pass
                train_loss.backward()

                # Updating weights
                self.optimizer.step()
                
                #Performing validation
                if (step+1) % validate_for_steps == 0:
                    val_loss =self.validate(validation_loader)
                    
                self.show_opt(epoch=epoch,step=step,loss=train_loss.item(), accuracy=accuracy,val_loss=val_loss,ftime=self.calculate_time(step,epoch))
        print('Training Finished.')
        
    def calculate_time(self,step,epoch):
        current_time = time.time()
        dt = (self.st_time-current_time)
        total_time = dt*self.num_epocs*self.batch_size
        remaining_time = total_time - (epoch*self.batch_size+step)
        self.st_time = current_time
        return time.strftime("%M:%S", time.gmtime(remaining_time))
        
        
    def show_opt(self,epoch,step,loss, accuracy,val_loss,ftime):
        print(f"Epoch={epoch}/{self.num_epocs} step={step}/{self.steps} loss={loss} accuracy={accuracy} val_loss={val_loss} remaining time={ftime}",end="\r")
        
        
    #Saving model parameters
    def save(self,file_path='model.pth'):
        torch.save({
            'state_dict':self.state_dict(),
            'optimizer':self.optimizer.state_dict()
        },file_path)

    #Loading the model parameters
    def load(self,path='model.pth'):
        if os.path.isfile(path):
            print('=> Loading model params...')
            #Loading the existing brain
            brain = torch.load(path)
            self.load_state_dict(brain['state_dict'])
            self.optimizer.load_state_dict(brain['optimizer'])

        else:
            raise Exception('No file found! Please check file path once again')
        self.to(self.device)
            
    def validate(self,validation_loader):
        size = len(validation_loader)
        min_valid_loss = np.inf
        idx = np.random.randint(0,size-1)
        with torch.no_grad():
            for step,(inputs, labels) in enumerate(validation_loader):
                if(idx == step):
                    inputs = inputs.to(self.device).type(torch.float)
                    labels = labels.to(self.device).type(torch.float)

                    # Forward pass
                    outputs = self.forward(inputs)
                
                    predictions_softmax = torch.log_softmax(outputs, dim = 1)
                    predictions = torch.max(predictions_softmax, dim = 1).values.view(-1,1)
                
                    val_loss = self.loss_func(predictions,labels).item()
                    self.history['val_loss'].append(val_loss)

                    if min_valid_loss > val_loss:
                        min_valid_loss = val_loss

                        # Saving State Dict
                        torch.save(self.state_dict(), 'saved_model.pth')
                    return val_loss
                        
    
    def test(self,test_loader):
        # Testing 
        with torch.no_grad():
            for step,(inputs,labels) in enumerate(test_loader):
                
                num_samples = inputs.shape[0]
                
                inputs = inputs.to(self.device).type(torch.float)
                labels = labels.to(self.device).type(torch.float)

                # Forward pass
                outputs = self.forward(inputs)

                predictions_softmax = torch.log_softmax(outputs, dim = 1)
                preds = torch.max(predictions_softmax, dim = 1).indices.view(-1,1)
                
                correct = (preds == labels).float()
                accuracy = correct.sum() / len(labels)
                accuracy = torch.round(accuracy * 100)
                print('Accuracy of the network :',preds)
                
                self.show_res(labels,predictions)
                
    def show_res(self,labels,predictions):
        #Evaluation
        cm = confusion_matrix(labels,predictions)
        
        #Analyzing the results through heat map
        sns.heatmap(cm,annot=True,fmt='d')
        plt.show()
        print(classification_report(labels,predictions))

MAX_SAMPLE_SIZE = 1000
FILE_NAME = './data.csv'


###Model hyper-parameters
BATCH_SIZE = 20
MAX_EPOCS = 50
learning_rate = 0.001


def remove_extras(data,plot=False):
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
    if plot:
        plot_data(data,'After removal')
        
    return data

def plot_data(data,data_label=None):
        ax = sns.countplot(x="labels", hue='labels',data=data)
        ax.set_xlabel("Range of speed (in km/hr)")
        ax.set_ylabel("No. of samples")
        ax.set_title('Data Summary')
        ax.legend(loc='upper right')
        if MAX_SAMPLE_SIZE > 0:
            ax.plot([0,data['labels'].max()],[MAX_SAMPLE_SIZE,MAX_SAMPLE_SIZE],label='Max samples')
        ax.legend()

class Transform:
    def __init__(self):
        self.transform = torchvision.transforms.Compose([
                                                 torchvision.transforms.ToTensor(),
                                                 torchvision.transforms.GaussianBlur(5),
                                                 torchvision.transforms.Resize((200,66)),
                                                 torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    def __call__(self,sample):
        sample = cv2.imread(sample[0]).astype(np.float32)
        #Changing color space
        #sample = cv2.cvtColor(sample,cv2.COLOR_RGB2YUV)
        return self.transform(sample)

class CustomDataset(Dataset):
    def __init__(self,data,transform=None,plot=False):
        self.transform = transform
        self.data = data 
        self.data['env'] = self.data['env'].apply(lambda x : os.path.join(os.getcwd(),x))
        
        # Separating the data into inputs and outputs
        self.inputs = self.data['env'].to_numpy().reshape(-1,1)
    
        # Performing one hot encoding. 
        #self.outputs = pd.get_dummies(self.data, columns = ['labels']).iloc[:,2:].to_numpy()
        self.outputs = self.data['labels'].to_numpy().reshape(-1,1)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):
        if self.transform:
            return self.transform(self.inputs[index]),self.outputs[index]
        return self.inputs[index],self.outputs[index]
        
    
    
    
def split_data(data,percent=(0.8,0.1,0.1)):
    train = data.sample(frac = percent[0])
    r_data = data.drop(train.index)
    test = r_data.sample(frac = (percent[2]/(percent[1]+percent[2])))
    val = r_data.drop(test.index)
    return train,test,val  



data = pd.read_csv(FILE_NAME,index_col=0)
data = remove_extras(data,plot=True)
train,test,val = split_data(data)

transforms = Transform()

train = CustomDataset(data=train,transform=transforms)
test = CustomDataset(data=test,transform=transforms)
val = CustomDataset(data=val,transform=transforms)



TRAINING_SIZE = len(train)
VALIDATION_SIZE = len(val)
TEST_SIZE = len(test)

print('Total training images :',TRAINING_SIZE)
print('Total validation images :',VALIDATION_SIZE)
print('Total test images :',TEST_SIZE)

train_data_loader = DataLoader(dataset=train,batch_size=BATCH_SIZE,shuffle=True)
test_data_loader = DataLoader(dataset=test)
val_data_loader = DataLoader(dataset=val,batch_size=BATCH_SIZE,shuffle=True)




model = CNN()
# Configuring model hyperparameters
model.config_model(n_steps=len(test),num_epocs=MAX_EPOCS,batch_size=BATCH_SIZE,learning_rate=learning_rate)


model.train(train_loader=train_data_loader,validation_loader=val_data_loader)
model.save()
