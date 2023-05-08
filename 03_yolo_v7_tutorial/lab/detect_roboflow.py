from roboflow import Roboflow

# Get API key:
# Projects > Settings > Roboflow API: Private API Key, Show
# Do not publish this key
# Alternatively, persist in local file, don't commit,
# and load from file
with open('./roboflow.key', 'r') as file:
    api_key = file.read().strip()

# Download model
rf = Roboflow(api_key=api_key)
project = rf.workspace("mikel-sagardia-tknfd").project("basic-object-detection-qkmda")
# Check in the Roboflow web UI rge model version we'd like
# This is a Roboflow model object, which in reality points to the Roboflow API
model = project.version(1).model

# Infer on a local image
img_url = "yolov7/Basic-Object-Detection-1/test/images/img9_png.rf.c3bea63eb9645df2c0d196d74b1550d5.jpg"
print(model.predict(img_url, confidence=40, overlap=30).json())
# Visualize/save the prediction
model.predict(img_url, confidence=40, overlap=30).save("test_prediction.jpg")

# Infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())
