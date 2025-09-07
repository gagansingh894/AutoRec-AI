# 🚗 AutoRecAI - Car Recommendation System

This project is an **end-to-end system** for recommending cars based on an uploaded image.

It combines:
- **Object Detection** → to identify cars in images  
- **Image Classification** → to recognize make and model  
- **Content-Based Recommendation Engine** → to suggest similar cars based on user-specified price range  

The system uses **Python** for model development and training, and **Rust** for high-performance model deployment via gRPC. It also provides a **UI (TBD)** for users to upload images and receive recommendations.

![Status](https://img.shields.io/badge/Status-Updating-blue)
---

## 📖 Overview  

AutoRecAI provides car recommendations based on an uploaded image.  

The system detects the car in the image, classifies its make/model, and recommends similar cars using metadata based on user specified price range.

---

## 📌 Architecture  

TODO: Add architecture diagram and explanation of components

---

### 📊 Data  

The dataset contains approx 60k images of different cars. The file name of each image
contains the make and model followed by other details separated by `-`. 

For example - the following file name `Acura_ILX_2013_28_16_110_15_4_70_55_179_39_FWD_5_4_4dr_Bbw` can be interpreted as

| Field           | Value | Description |
|-----------------|-------|-------------|
| **Make**        | Acura | Manufacturer |
| **Model**       | ILX   | Model Name |
| **Year**        | 2013  | Model Year |
| **MPG City**    | 28    | Miles per Gallon (City) |
| **MPG Highway** | 16    | Miles per Gallon (Highway) |
| **Horsepower**  | 110   | Engine Power (HP) |
| **Torque**      | 15    | Torque (lb-ft) |
| **Weight**      | 4     | Vehicle Weight (1000 lbs) |
| **Length**      | 70    | Length (inches) |
| **Width**       | 55    | Width (inches) |
| **Height**      | 179   | Height (inches) |
| **Wheelbase**   | 39    | Wheelbase (inches) |
| **Drive Type**  | FWD   | Front-Wheel Drive |
| **Doors**       | 5     | Number of Doors |
| **Body Style**  | 4dr   | Four-Door Sedan |
| **Color**       | Bbw   | Black/White/Blue (Color Code) |

The dataset can be downloaded from [here](https://www.kaggle.com/datasets/prondeau/the-car-connection-picture-dataset/data).

---

### ⚙️ Data Processing  

![data processing.jpg](docs/data%20processing.jpg)

#### Data Grouping and Labeling
The initial dataset of 60k images is grouped by `make+model+year` which also acts label for the data.
The name of the folder is the label which contains images of the corresponding car.

#### Image Augmentation
In order to train our custom classification model, we need to generate some artificial data for our images, with the hope
that the model will be able to generalise better. This will be achieved by augmenting our image data. In order to ensure
the data quality is maintained we filter out images which are not of car (example - wheel, side mirror etc) using Yolo based
filtering. This removes noise from our data set.

#### Car Metadata 
As mentioned in the above section, the filename of the image has information about the car. The metadata processing
pipeline extracts this information from the images and uploads it to Qdrant vector database. This information
will be later on used for making recommendations

---

### 🤖Model Development

We have built **2 primary models** for the AutoRec-AI project:


### CarMatch

**Purpose:** Detects cars in images and classifies them into fine-grained categories.

**Architecture:**

- **YOLOv8:** Detects bounding boxes of cars and trucks in input images.
- **MobileNetV3:** Classifies cropped car ROIs into one of 2007 classes.

**Workflow:**

1. Input images are flattened and passed to `CarMatch.forward()`.
2. `_detect()` reshapes images, runs YOLO detection, and extracts the **highest-scoring car/truck ROIs**.
3. ROIs are resized to 224×224 and normalized.
4. `_classify()` feeds ROIs to MobileNet for final classification.

```
---> CarMatch.forward() --> _detect() --> YOLOv8
                                   |--> Extract highest-scoring ROIs
                                   |--> Resize to 224x224
                                   |--> _classify() --> MobileNetV3 --> Output logits
```

**Saving the Model:**

- The `save()` method allows:
  - Saving the **native PyTorch model**.
  - Optional **TorchScript tracing** using an example input tensor of shape `[batch_size, 416*416*3]`.
- **Example input for tracing:** A tensor containing flattened images, e.g., `[batch_size, 416*416*3]`, where some images may be empty (all zeros) if no car is detected.

### CustomMobileNet

**Purpose:** Standalone MobileNetV3 for transfer learning on your dataset.

**Architecture:** MobileNetV3 large backbone with classifier replaced by a linear layer matching the number of classes in your dataset.

**Workflow:**

1. Input images are resized to 224×224, normalized using ImageNet stats, and batched.
2. Forward pass returns raw logits.
3. `train_and_evaluate()` handles training with CrossEntropyLoss and evaluates accuracy on the test set.

**Saving the Model:**

- Native PyTorch model is saved to disk.
- TorchScript tracing uses a dummy input of shape `[1, 3, 224, 224]`.



#### Notes

- For both models, predictions are logits; you need `argmax` or `softmax` to get class indices or probabilities.
- **Class mapping:** `ImageFolder` is used for datasets; `class_to_idx` provides mapping of class names to indices, and `idx_to_class` gives index → class name lookup.

---

## 🛠️ Tech Stack  

- **Programming Languages**: Python, Rust
- **Model Development**: PyTorch, Ultralytics  
- **Object Detection Model**: YOLO  
- **Image Classification Model**: MobileNet (Transfer Learning)  
- **Recommendation System**: Content-Based Filtering  
- **Feature Store**: Qdrant (stores car embeddings for fast similarity search)  
- **Database**: PostgreSQL (used by MLflow for experiment and run metadata)  
- **Experiment Tracking**: MLFlow  
- **Workflow Orchestration**: Argo  
- **Data Versioning**: DVC  
- **Artefact Storage**: S3  
- **Model Serving**: [jams-rs](https://github.com/gagansingh894/jams-rs)  
- **Deployment & Orchestration**: Kubernetes (K8s)

---


## 📖 Project Background  

This project was originally developed as part of my **MSc Data Science dissertation**.  
I am currently updating it to align with **modern MLOps and AI system design best practices**.  

👉 Old code is available in the [old-main branch](https://github.com/gagansingh894/AutoRec-AI/tree/old-main).  