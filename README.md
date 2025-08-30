# 🚗 AutoRecAI - Car Recommendation System

---

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

TODO: Add details about data preprocessing, YOLO detection, MobileNet classification, and how car metadata is integrated for recommendations

---

## 📌 Architecture  

TODO: Add architecture diagram and explanation of components

---

## 🛠️ Tech Stack  

- **Programming Languages**: Python, Rust
- **Model Development**: PyTorch, Ultralytics  
- **Object Detection Model**: YOLO  
- **Image Classification Model**: MobileNet (Transfer Learning)  
- **Recommendation System**: Content-Based Filtering  
- - **Feature Store**: Qdrant (stores car embeddings for fast similarity search)  
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