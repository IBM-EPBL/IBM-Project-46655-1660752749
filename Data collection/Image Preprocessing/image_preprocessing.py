"""Importing Libraries """
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

"""Configuring the ImageDataGenerator Class"""

train_data_generator1 = ImageDataGenerator(rescale=1.0/255,horizontal_flip=True)
test_data_generator1 = ImageDataGenerator(rescale=1.0/255, horizontal_flip=True)

train_data_generator2 = ImageDataGenerator(rescale=1.0/255,rotation_range=90)
test_data_generator2 = ImageDataGenerator(rescale=1.0/255,rotation_range=90)

train_data_generator3 = ImageDataGenerator(rescale=1.0/255,brightness_range=[0.2,1.0])
test_data_generator3 = ImageDataGenerator(rescale=1.0/255,brightness_range=[0.2,1.0])

train_data_generator4 = ImageDataGenerator(rescale=1.0/255,zoom_range=[0.5,1.0])
test_data_generator4 = ImageDataGenerator(rescale=1.0/255,zoom_range=[0.5,1.0])

"""Applying ImageDataGenerator to test dataset and train dataset"""

trdata1 = train_data_generator1.flow_from_directory(r'D:\College\7th semester\ibm\Project Development Phase\Sprint 1\Data collection\Dataset\train',target_size=(64,64),batch_size=(3),color_mode='grayscale',class_mode='categorical')

trdata2 = train_data_generator2.flow_from_directory(r'D:\College\7th semester\ibm\Project Development Phase\Sprint 1\Data collection\Dataset\train',target_size=(64,64),batch_size=(3),color_mode='grayscale',class_mode='categorical')

trdata3 = train_data_generator3.flow_from_directory(r'D:\College\7th semester\ibm\Project Development Phase\Sprint 1\Data collection\Dataset\train',target_size=(64,64),batch_size=(3),color_mode='grayscale',class_mode='categorical')

trdata4 = train_data_generator4.flow_from_directory(r'D:\College\7th semester\ibm\Project Development Phase\Sprint 1\Data collection\Dataset\train',target_size=(64,64),batch_size=(3),color_mode='grayscale',class_mode='categorical')

tsdata1 = test_data_generator1.flow_from_directory(r'D:\College\7th semester\ibm\Project Development Phase\Sprint 1\Data collection\Dataset\test',target_size=(64,64),batch_size=32,class_mode='categorical',color_mode='grayscale')

tsdata2 = test_data_generator2.flow_from_directory(r'D:\College\7th semester\ibm\Project Development Phase\Sprint 1\Data collection\Dataset\test',target_size=(64,64),batch_size=32,class_mode='categorical',color_mode='grayscale')

tsdata3 = test_data_generator3.flow_from_directory(r'D:\College\7th semester\ibm\Project Development Phase\Sprint 1\Data collection\Dataset\test',target_size=(64,64),batch_size=32,class_mode='categorical',color_mode='grayscale')

tsdata4 = test_data_generator4.flow_from_directory(r'D:\College\7th semester\ibm\Project Development Phase\Sprint 1\Data collection\Dataset\test',target_size=(64,64),batch_size=32,class_mode='categorical',color_mode='grayscale')

print(trdata1.class_indices)

print(trdata2.class_indices)

print(trdata3.class_indices)

print(trdata4.class_indices)

print(tsdata1.class_indices)

print(tsdata2.class_indices)

print(tsdata3.class_indices)

print(tsdata4.class_indices)
