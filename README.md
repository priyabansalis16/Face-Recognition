# Face-Recognition
Face-Recognition using Deep Learning

Installing face recognition libraries
In order to perform face recognition, we need to install two additional libraries:
• dlib
• face recognition

Google-Colaboratory Commands:
!pip install face_recognition
!python encode_faces.py --dataset dataset --encodings encodings.pickle
!python recognize_faces_image.py --encodings encodings.pickle --image examples/example_01.jpeg

FACE RECOGNITION PROJECT STRUCTURE
• dataset/: Contains face images for five characters organized into subdirectories based on their
respective names.
• examples/: It has images for testing that are not in the dataset.
• Resulg.jpeg/: This is where we can store our processed face recognition images. I’m leaving one
of mine in the folder — the classic “lunch scene” from the original Jurassic Park movie.
We also have 3 files in the root directory:
• encode_faces.py : Encodings (128-d vectors) for faces are built with this script.
• recognize_faces_image.py : Recognize faces in a single image (based on encodings from our
dataset).
• encodings. pickle : Facial recognitions encodings are generated from our dataset
via encode_faces.py and then serialized to disk.

ARCHITECTURE AND WORKING
Before we can recognize faces in images, we first need to quantify the faces in our training set.
Keep in mind that we are not actually training a network here — the network has already been
trained to create 128-d embeddings on a dataset of ~3 million images.
We certainly could train a network from scratch or even fine-tune the weights of an existing
model but that is more than likely overkill for many projects. Furthermore, we would need
a lot of images to train the network from scratch.
Instead, it’s easier to use the pre-trained network and then use it to construct 128-d embeddings
for each of the 218 faces in our dataset.
Then, during classification, we can use a simple k-NN model + votes to make the final face
classification. Other traditional machine learning models can be used here as well.
• KNN stores the entire training dataset which it uses as its representation.
• KNN does not learn any model.
• KNN makes predictions just-in-time by calculating the similarity between an input sample and
each training instance.
• There are many distance measures to choose from to match the structure of the input data.
• That it is a good idea to rescale the data, such as using normalization, when using KNN.
