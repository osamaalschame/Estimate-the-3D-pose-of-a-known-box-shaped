# Dataset Download Instructions

This project uses a dataset hosted on Google Drive. Follow the steps below to download and prepare the dataset.

## Download Dataset

Click the link below to download the dataset manually:

🔗 [Download Dataset](https://drive.google.com/file/d/1YeZ4mg_qZ4dBvKKfgGF8RQcyOMNoMp37/view?usp=sharing)

## Setup

1. After downloading the file, extract it using a tool like `unzip` or `7-Zip` if it's in `.zip` or `.rar` format.
2. Move the extracted dataset folder into dataset directory:
    ```
    Estimate-the-3D-pose-of-a-known-box-shaped/
    ├── dataset/
    └── ...
    ```

## 💡 Notes

- Make sure the dataset structure matches what your `data.yaml` file expects.
- If using Google Colab, you can also mount your Google Drive and access the file directly:
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```
