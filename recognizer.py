import torch
import torchvision
import numpy as np
import cv2
from PIL import Image
import string

def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device
device = get_device()

print('device:', device)

# the model architecture
class Network(torch.nn.Module):
    def __init__(self):
        
        super(Network, self).__init__()
        
        number_of_classes = 26
        
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=(2, 2))
        self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.max_pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = torch.nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = torch.nn.Linear(in_features=120, out_features=84)
        self.fc3 = torch.nn.Linear(in_features=84, out_features=number_of_classes)    
        
        self.relu = torch.nn.ReLU()
        self.dpt = torch.nn.Dropout(0.4) # 40% probability
    
    def forward(self, t):
        t = self.conv1(t) # 1 * 28 * 28 -> 6 * 28 * 28 
        t = self.relu(t)

        t = self.max_pool(t) # 6 * 14 * 14 
        
        t = self.conv2(t) # 16 * 10 * 10
        t = self.relu(t)

        t = self.max_pool(t) # 16 * 5 * 5
        
        t = self.dpt(t)
        
        t = t.reshape(-1, 16 * 5 * 5) # flatten in order to feed to the FC layers 

        t = self.fc1(t) # 400 -> 120
        t = self.relu(t)

        t = self.fc2(t) # 120 -> 84
        t = self.relu(t)
        
        t = self.fc3(t) # 84 -> 26 (number of classes)
        
        return t
    


# load saved model
path = "./saved_models/my_cnn_model.pt" 
net = Network() # create a model object
net.to(device)
try:
    net.load_state_dict(torch.load(path, map_location=torch.device(device))) # load saved object
    print('model loaded successfully')
except:
    print('ERROR: cannot load the saved model!')
    
    
# Press 'esc' to quit
# Press 'r' to reset
# Try to draw a big fat alphabet in the center covering almost the whole board

net = net.eval()
drawing = False 

def draw_circle(event, x, y, flags, param):
    global drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.circle(frame, (x, y), 30, (255, 255, 255), -1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
            
            
frame = np.zeros((650, 650), np.uint8)
cv2.namedWindow('ALPHABET RECOGNIZER')
cv2.setMouseCallback('ALPHABET RECOGNIZER', draw_circle)

while True:    
    
    frame[:100][:100] = 0 # removing the previous digit text on top left
    
    # Transforming the image from frame to feed it to the netowrk
    input_img = Image.fromarray(frame) # PIL image
    resized_img = input_img.resize((28, 28), Image.ANTIALIAS) # resizing captured image to (28, 28)
    tensor_img = torchvision.transforms.ToTensor()(resized_img) # (1, 28, 28)
    tensor_img = tensor_img.reshape(1, 1, 28, 28)
    
    # Feed the image to the Network
    predicted_index = net(tensor_img).argmax(dim=1).item()
    #predicted_index = 0
    predicted_alphabet = list(string.ascii_uppercase)[predicted_index] # [A: 0... Z: 25]
    
    # Write the prediction on the frame
    frame[:100][:100] = 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    text1 = f"Predicted Alphabet: {str(predicted_alphabet)}..."
    text2 = f"(quit: 'esc', reset: 'r')" 
    frame = cv2.putText(frame, text1, (10, 30), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
    frame = cv2.putText(frame, text2, (10, 70), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
    
    cv2.imshow('ALPHABET RECOGNIZER', frame)
    
    k = cv2.waitKey(1) & 0xFF
    if k == ord('r'): # reset screen
        frame = np.zeros((650, 650), np.uint8)
    elif k == 27: # quit
        break

cv2.destroyAllWindows()

# Plotting the bar of the output neuron values (of the latest Network output prediction)
# plt.figure(figsize=(20, 10))
# plt.bar(list(string.ascii_uppercase), net(tensor_img).flatten().detach().numpy())
# plt.show()
