
import os
import json

import streamlit as st

from PIL import Image
import torchvision
from torchvision import models, transforms 

import torch
import torch.nn as nn

#predicted_idx = str(round(torch.sigmoid(outputs.max(1)[0]).item()))

def get_prediction(image, model, imagenet_class_index):
    tensor = transform_image(image=image)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx], outputs

def transform_image(image):
    """ Transform image to fit model

    Args:
        image (image): Input image from the user

    Returns:
        tensor: transformed image 
    """
    transformation = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    return transformation(image).unsqueeze(0)


@st.cache
def load_model():
    # Make sure to pass `pretrained` as `True` to use the pretrained weights:
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(512, 1)
    model.load_state_dict(torch.load('/Users/paantur/Documents/GitHub/nn_project/cd_weights.pt'))
    # Since we are using our model only for inference, switch to `eval` mode:
    model.eval()

    imagenet_class_index = {'0' : 'cat', '1' : 'dog'}
    
    return model, imagenet_class_index


def main():
    
    st.title("Predict objects in an image")
    st.write("This application knows the objects in an image , but works best when only one object is in the image")

    
    model, imagenet_class_index = load_model()
    
    image_file  = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

    if image_file:
       
        left_column, right_column = st.columns(2)
        left_column.image(image_file, caption="Uploaded image", use_column_width=True)
        image = Image.open(image_file)

        pred_button = st.button("Predict")
        
        
        if pred_button:

            prediction = get_prediction(image, model, imagenet_class_index)
            right_column.title("Prediction")
            right_column.write(prediction)



if __name__ == '__main__':
    main()