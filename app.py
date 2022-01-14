import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input
from PIL import Image
import requests

st.title('Omdena - Ahmedabad Chapter')
st.header('Anamoly detection on Martian Surface')

#st.balloons()
option = st.sidebar.radio("Anamoly detection on Martian Surface",['Home', 'About'])
 
if option == 'Home':
  st.sidebar.subheader('List of Anamolies')
  
  st.sidebar.subheader("Class 1: Craters")
  #url ='https://www.marsartgallery.com/images/martiancraterplaxco.jpg'
  #image = Image.open(requests.get(url, stream=True).raw)
  image1 = Image.open('Anamolies-images/craters.jpg')
  st.sidebar.image(image1, caption='Craters on Mars')
  st.sidebar.write("Craters are caused when a bolide collides with a planet.The Martian surface contains thousands of impact craters because, unlike Earth, Mars has a stable crust, low erosion rate, and no active sources of lava")
    
  st.sidebar.subheader("Class 2: Dark dunes")
  image2 = Image.open('Anamolies-images/dark-dunes.jpg')
  st.sidebar.image(image2, caption='Dark dunes on Mars')
  st.sidebar.write("The dunes within and around the crater are thought to contain sandy material rich in pyroxene and olivine: rock forming minerals that are mafic(containing Magnesium and Iron)")

  st.sidebar.subheader("Class 3: Slope streaks")
  image3 = Image.open('Anamolies-images/slope-streak.jpg')
  st.sidebar.image(image3, caption='Slope streak on Mars')
  st.sidebar.write("Slope streaks are prevalent on the surface of the Mars,but they come in multitude of shapes and sizes.")
    
  st.sidebar.subheader("Class 4: Bright dunes")
  image4 = Image.open('Anamolies-images/bright-dune.jpg')
  st.sidebar.image(image4, caption='Bright dune on Mars')
  st.sidebar.write("These martian dunes are brighter and are composed of possibly sulphates.")

  st.sidebar.subheader("Class 5: Impact Ejecta")
  image5 = Image.open('Anamolies-images/impact-ejecta.jpg')
  st.sidebar.image(image5, caption='Impact ejecta on Mars')
  st.sidebar.write("Impact ejecta is material that is thrown up and out of the surface of a planet as a result of the impact of an meteorite, asteroid or comet. The material that was originally beneath the surface of the planet then rains down onto the environs of the newly formed impact crater.")
  
  st.sidebar.subheader("Class 6: Swiss cheese")
  image6 = Image.open('Anamolies-images/swiss-cheese.jpg')
  st.sidebar.image(image6, caption='Swiss cheese on Mars')
  st.sidebar.write("The Martian south polar cap is a layer of carbon dioxide ice, full of pits that make it look like Swiss cheese. The pits form when the Sun heats the ice and makes it sublimate (transform from a solid to a gas).")
  
  st.sidebar.subheader("Class 7: Spiders")
  image7 = Image.open('Anamolies-images/spiders.jpg')
  st.sidebar.image(image7, caption='Spiders on Mars')
  st.sidebar.write("Spiders are actually topological troughs formed when dry ice directly sublimates to a gas")
  
  uploaded_file = st.file_uploader("Choose a image file", type="jpg")
  
  if uploaded_file is not None:
      # Convert the file to an opencv image.
      file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
      opencv_image = cv2.imdecode(file_bytes, 1)
      opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
      resized = cv2.resize(opencv_image,(224,224))
      # Now do something with the image! For example, let's display it:
      st.image(opencv_image, channels="RGB")

      resized = mobilenet_v2_preprocess_input(resized)
      img_reshape = resized[np.newaxis,...]
      
      predictn = st.button("Predict")

  def yolo_v4(image, confidence_threshold, overlap_threshold):
      @st.cache(allow_output_mutation=True)
      def load_network(config_path, weights_path):
          net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
          output_layer_names = net.getLayerNames()
          output_layer_names = [output_layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
          return net, output_layer_names
      net, output_layer_names = load_network("yolov4-custom.cfg", "yolov4.weights")

      # Run the YOLO neural net.
      blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
      net.setInput(blob)
      layer_outputs = net.forward(output_layer_names)

      boxes, confidences, class_IDs = [], [], []
      H, W = image.shape[:2]
      for output in layer_outputs:
          for detection in output:
              scores = detection[5:]
              classID = np.argmax(scores)
              confidence = scores[classID]
              if confidence > confidence_threshold:
                  box = detection[0:4] * np.array([W, H, W, H])
                  centerX, centerY, width, height = box.astype("int")
                  x, y = int(centerX - (width / 2)), int(centerY - (height / 2))
                  boxes.append([x, y, int(width), int(height)])
                  confidences.append(float(confidence))
                  class_IDs.append(classID)
      indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, overlap_threshold)

      UDACITY_LABELS = {
          0: 'crater',
          1: 'dark dune',
          2: 'slope streak',
          3: 'bright dune',
          4: 'impact ejecta',
          5: 'swiss cheese',
          6: 'spider'
      }
      xmin, xmax, ymin, ymax, labels = [], [], [], [], []
      if len(indices) > 0:
          # loop over the indexes we are keeping
          for i in indices.flatten():
              label = UDACITY_LABELS.get(class_IDs[i], None)
              if label is None:
                  continue

              # extract the bounding box coordinates
              x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]

              xmin.append(x)
              ymin.append(y)
              xmax.append(x+w)
              ymax.append(y+h)
              labels.append(label)

      boxes = pd.DataFrame({"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax, "labels": labels})
      return boxes[["xmin", "ymin", "xmax", "ymax", "labels"]]
    
  }



if option == 'About':
  st.write("Here the dataset description goes")


