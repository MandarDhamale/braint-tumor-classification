# Brain Tumor Classification using VGG16 Transfer Learning

This project reproduces the methodology from the paper **"Brain tumor classification using MRI images and deep learning techniques" by Wong et al. (2025), PLoS One**. The goal is to classify brain MRI images into four categories: glioma, meningioma, pituitary, and no tumor, using the VGG16 architecture with transfer learning and a two-stage fine-tuning process.

---

## Environment Setup (Using Conda)

This project relies on Conda for managing the Python environment and its dependencies to ensure reproducibility.

1.  **Prerequisites:**
    * **Conda:** Ensure you have Miniconda or Anaconda installed on your system. You can download it from [https://conda.io/projects/conda/en/latest/user-guide/install/index.html](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).
    * **(Optional but Recommended) NVIDIA GPU Setup:** For significantly faster training, ensure you have:
        * An NVIDIA GPU compatible with CUDA.
        * The latest NVIDIA drivers installed for your operating system.
        * CUDA Toolkit and cuDNN library installed. **Crucially, the versions must match the requirements of the TensorFlow version specified in the `environment.yml` file.** Check the `environment.yml` for the TensorFlow version (e.g., `tensorflow=2.x.x`) and then consult the official [TensorFlow GPU setup guide](https://www.tensorflow.org/install/gpu) for the corresponding compatible CUDA and cuDNN versions.

2.  **Get Project Files:**
    * Clone this repository or download the project files. Ensure the following files are in your project directory:
        * `environment.yml` (Defines the Conda environment)
        * `main_train2.py` (The main training and evaluation script)
        * `main_train2.ipynb` (Jupyter notebook version of the script)
        * `data` folder

3.  **Create Conda Environment:**
    * Open your terminal (Linux/macOS) or Anaconda Prompt (Windows).
    * Navigate (`cd`) to the project directory where you placed the files.
    * Run the following command to create the Conda environment using the provided specification file:
        ```bash
        conda env create -f environment.yml
        ```
    * This command reads `environment.yml`, creates a new environment (typically named `ml-gpu` as specified in the file), and installs all listed packages (Python, TensorFlow, Scikit-learn, etc.) with their specific versions. This process might take a few minutes.

4.  **Activate Environment:**
    * Before running the script, you **must** activate the newly created Conda environment:
        ```bash
        conda activate ml-gpu
        ```
    * Your terminal prompt should now show `(ml-gpu)` at the beginning, indicating the environment is active.

---

## Dataset 

The model requires a dataset of brain MRI images organized into specific training, validation, and testing sets.

1.  **Download Data:**
    * This project requires a 4-class brain tumor MRI dataset with images labeled as `glioma`, `meningioma`, `pituitary`, and `no_tumor`.
    * The specific dataset used to achieve the results in the Milestone II report was sourced from Kaggle, combining data similar to the original paper's sources.
2.  **Data Structure:**
    * The final structure must look like this:
        ```
        project_root/
        ├── main_train2.py
        ├── main_train2.ipynb
        ├── environment.yml
        ├── README.md
        └── data/
            ├── train/
            │   ├── glioma/       (contains glioma training images)
            │   ├── meningioma/   (contains meningioma training images)
            │   ├── no_tumor/     (contains no_tumor training images)
            │   └── pituitary/    (contains pituitary training images)
            ├── validation/
            │   ├── glioma/       (contains glioma validation images)
            │   ├── meningioma/   (...)
            │   ├── no_tumor/     (...)
            │   └── pituitary/    (...)
            └── test/
                ├── glioma/       (contains glioma test images)
                ├── meningioma/   (...)
                ├── no_tumor/     (...)
                └── pituitary/    (...)
        ```
    * **Note:** The script `main_train2.py` relies *exactly* on this directory structure and naming convention to load the data correctly using Keras `ImageDataGenerator`. Ensure the directory and class names match precisely.

---

## Usage 

1.  **Activate Conda Environment:** If not already active, open your terminal/Anaconda Prompt, navigate to the project directory, and run:
    ```bash
    conda activate ml-gpu
    ```
2.  **Run the Script:** Execute the main Python script or use the jupyter notebook:
    ```bash
    python3 main_train2.py
    ```

**What the script does:**
* Loads and preprocesses data from the `data` directory using `ImageDataGenerator` (applies augmentation to training data).
* Builds the VGG16-based transfer learning model.
* Prints the model summary (including layer details and parameter counts).
* Performs the **two-stage training process**:
    * Stage 1: Trains only the classification head for 50 epochs.
    * Stage 2: Unfreezes specified VGG16 layers and fine-tunes the model for another 50 epochs with a lower learning rate.
* Evaluates the final trained model on the `test` dataset.
* Prints the final Test Accuracy and Test Loss to the console.
* Generates and prints a detailed **Classification Report** (Precision, Recall, F1-Score for each class).
* Generates and saves the following plots as PNG files in the project directory:
    * `confusion_matrix.png`
    * `auc_roc_curve.png`
    * `precision_recall_curve.png`

---
