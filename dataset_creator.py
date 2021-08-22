from string import ascii_uppercase # for getting list uppercase english letters
import os
import csv # for reading the csv file
import numpy as np # for reading the image data array
from PIL import Image # for retrieving the image from np array


csv_file_path = './downloaded_data/A_Z Handwritten Data.csv' # path to downloded csv file
image_folder_path = './AlphabetDataset' # path to target folder to save images. Note: Path will have 26 empty folder with name as alphabets

alphabet_list = list(ascii_uppercase) # List of uppercase alphabets


# create directories for each alphabet
for alphabet in alphabet_list:
    path = image_folder_path + '/' + alphabet
    if not os.path.exists(path):
        os.makedirs(path)


# read the csv file and retrie the images, labels and store them in the respective folders
last_alphabet_name =  None
with open(csv_file_path, newline='') as csvfile:
    
    reader = csv.reader(csvfile, delimiter=',', quotechar='|') # Iterator over each of the total 372451 rows.
    count = 0
    for row in reader: # Each row has 785 colums(elements). The 1st element is the Label. Remaining 784 elements are the flattened 28*28 image
      
        alphabet_number = row.pop(0) # len(row): 785 -> 784(decreased). digit_name: 0 to 25(a to z)
        
        image_array = np.asarray(row) # image_array.shape -> (784,)
        image_array = image_array.reshape(28, 28)
        new_image = Image.fromarray(image_array.astype('uint8')) # converting to PIL image
        
        if last_alphabet_name != str(alphabet_list[int(alphabet_number)]):
            # The alphabets are sorted in the dataset. 
            # Thus, this if condition satisfies when all images of one alphabet is scanned. 
            if last_alphabet_name != None: 
                print(count, 'occurences\n') # Count of the last processed alphabet.
            last_alphabet_name = str(alphabet_list[int(alphabet_number)]) 
            print ("Processing Alphabet - " + str(last_alphabet_name)) 
            count = 0 
            
        image_path = image_folder_path + '/' + last_alphabet_name + '/' + str(last_alphabet_name) + '-' + str(count) + '.png'
        new_image.save(image_path)
            
        count = count + 1
        
print(count, 'occurences') # as occurences of Z would not be printed in the loop's if's print statement
