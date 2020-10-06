from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img
from tensorflow.keras.models import model_from_json
import numpy as np
from tensorflow.keras.optimizers import RMSprop
import tensorflow as tf
# model.load_weights("weights.best_dropout.hdf5")
# Compile model (required to make predictions)
# model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.001), metrics = ['accuracy'])
# print("Created model and loaded weights from file")
import streamlit as st
import cv2
import os # inbuilt module
import random # inbuilt module
import webbrowser # inbuilt module
# st.title("""Pulmonary disease  Recognizer""")
from PIL import Image

st.set_option('deprecation.showfileUploaderEncoding', False)
#=================================== Title ===============================
st.title("""Pulmonary Diseases  Recognizer""")





loaded_model = tf.keras.models.Sequential([  ## initializing and making an empty model with sequential
  
    # Note the input shape is the desired size of the image 300x300 with 3 bytes color
    # This is the first convolution layer
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300,1)), ## image input shape is 300x300x3 
                           #16 neurons in this layer


    tf.keras.layers.MaxPooling2D(2,2),    # doing max_pooling
    tf.keras.layers.Dropout(0.2),

  
    # The second convolution layer
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'), # another layer with 32 neurons
    tf.keras.layers.MaxPooling2D(2,2),     # doing max_pooling
    tf.keras.layers.Dropout(0.2),


    # The third convolution layer
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'), # another layer with 64 neurons
    tf.keras.layers.MaxPooling2D(2,2),        # doing max_pooling
    tf.keras.layers.Dropout(0.2),



    # The fourth convolution layer
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'), # another layer with 64 neurons
    tf.keras.layers.MaxPooling2D(2,2),          # doing max_pooling
    tf.keras.layers.Dropout(0.2),  


    # The fifth convolution 
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'), # another layer with 64 neurons
    tf.keras.layers.MaxPooling2D(2,2),        # doing max_pooling
    tf.keras.layers.Dropout(0.2),



    tf.keras.layers.Flatten(),  # reducing layers arrays 
    tf.keras.layers.Dense(512, activation='relu'), # 512 neuron hidden layer



    # Only 1 output neuron. It will contain a value from 0-1 where 0 for ('normal') clas and 
    # 1 for ('pneumonia') class
    tf.keras.layers.Dense(1, activation='sigmoid')

])

# to get the summary of the model
loaded_model.summary()  # summarising a model

# configure the model for traning by adding metrics
# model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.001), metrics = ['accuracy'])


















#================================= Title Image ===========================
st.text("""""")
img_path_list = ['1.jpg']
index = random.choice([0])
image = Image.open(img_path_list[index])
st.image(
 	        image,
 	        use_column_width=True,
 	    )

#================================= About =================================
st.write("""
## 1.  About
	""")
st.write("""
Hi all 🧑🏽‍💻 . It is Normal or Infected Recognizer site!!!
	""")
st.write("""
You have to upload your own test images to test it!!!
	""")
st.write("""
If you dont have images then we already selected some test images for you, you have to just go to that section & click the **⬇️ Download** button to download those pictures!  
	""")

#============================ How To Use It ===============================
st.write("""
## 2. How To Use It
	""")
st.write("""
Well, it's pretty simple!!!
- Let me clear first, the model has power to predict image of Lungs only, so you are requested to give image of a Lung, unless useless prediction can be done!!! 😆 
- First of all,get images!
- Next, just Browse that file or Drag & drop that file!
- Please make sure that, you are uploading a picture file!
- Press the ** 👉🏼 Predict button to see the Result!!!

🔘 **NOTE :** *If you upload other than an image file, then it will show an error massage when you will click the* **👉🏼 Predict** *button!!!*
	""")

#========================= What It Will Predict ===========================
st.write("""
## 3. What It Will Predict
	""")
st.write("""
Well, it can predict wheather the image you have uploaded is the image of a Infected Person or Not""")

#============================== Sample Images For Testing ==================
st.write("""
## 4. Get Some Images For Testing!!!
	""")
st.write("""
Hey there! here is some images of Lungs !
- Here you can find a total of 10 images **[**5 for each category**]**
- Just click on **⬇️ Download** button & download those images!!!
- You can also use your own images!!!
	""")

#============================= Download Button =============================
st.text("""""")
st.text("""""")
download = st.button("          ⬇️ Download")

#============================ Download Clicked =============================
if download:
	link = "https://drive.google.com/drive/folders/1i_ukZQxJsCWq2WpISwNa5HFD8smxNdee?usp=sharing"
	try:
		webbrowser.open(link)
	except:
		st.write("""
    		⭕ Something Went Wrong!!! Please Try Again Later!!!
    		""")

# 
#============================ Behind The Scene ==========================


#======================== Time To See The Magic ===========================
st.write("""
## 👁️‍🗨 Time To See The Results. Upload a picture 🌀. """)

#========================== File Uploader ===================================
img_file_buffer = st.file_uploader("Upload an image here 👇🏻")
#  👀 O
try:
	image = Image.open(img_file_buffer)
	img_array = np.array(image)
	st.write("""
		Preview Of Image 👀  """)				
	if image is not None:
	    st.image(
	        image,
	        use_column_width=True
	    )
	st.write("""
		Now, you are just one step ahead of prediction.
		""")
	st.write("""
		**Just Click The '👉🏼 Predict' Button To See The Prediction Corresponding To This Image! 😄**
		""")
except Exception as e:
	print(e)
	

#================================= Predict Button ============================
st.text("""""")
submit = st.button("👉🏼 Predict")

#==================================== Model ==================================
#def processing(testing_image_path):
#    IMG_SIZE = 300
#    img = load_img(testing_image_path, 
#            target_size=(IMG_SIZE, IMG_SIZE), color_mode="grayscale")
#    img_array = img_to_array(img)
#    img_array = img_array/255.0
#    img_array = img_array.reshape((1, 300, 300, 1))
#    img_array = np.expand_dims(img_array,axis=0)
#    prediction =loaded_model.predict(img_array)    
#    return prediction



def processing(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(300,300))
    from keras.preprocessing import image

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis =0)

    images = np.vstack([x])



    classes = loaded_model.predict(images)
    return classes

def generate_result(prediction):
	st.write("""
	## 🎯 RESULT
		""")
	if prediction[0]<0.5:
	    st.write("""
	    	## Model predicts it as an image of a Normal Person !
	    	""")
	else:
	    st.write("""
	    	## Model predicts it as an image of a Infected Person !
	    	""")

#=========================== Predict Button Clicked ==========================
if submit:
	try:
		# Creating Directory
		not_created = True
		while not_created:
			name_of_directory = random.choice(list(range(0, 1885211)))
			try:
				ROOT_DIR = os.path.abspath(os.curdir)
				if str(name_of_directory) not in os.listdir(ROOT_DIR):
					not_created = False
					path = ROOT_DIR + "\\" + str(name_of_directory)
					os.mkdir(path)
					# directory made!
			except:
				st.write("""
					### ❗ Oops!!! Seems like it will not support in you OS!!!
					""")

		# save image on that directory
		save_img(path+"\\test_image.png", img_array)

		image_path = path+"\\test_image.png"
		# Predicting
		st.write("👁️ Predicting...")
		model_path_h5 = "weights.best_dropout.hdf5"
		loaded_model.load_weights(model_path_h5)



                
# model_path_json = "model.json"
# json_file = open(model_path_json, 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
                


		loaded_model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.001), metrics = ['accuracy'])

		prediction = processing(image_path)

		# Delete the folder
		dir_path = path
		try:
		    shutil.rmtree(dir_path)
		except:
			pass

		generate_result(prediction)
		print(generate_result)

	except Exception as e:
		st.write(e)

#=============================== Copy Right ==============================
st.text("""""")
st.text("""""")
st.text("""""")
st.text("""""")
st.text("""""")
st.text("""""")
st.text("""""")
st.text("""""")
st.text("""""")
st.text("""""")
st.write("""
### ©️ Arnav
	""")

