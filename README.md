# Face Recognition System

This project is a face recognition system that can register users and take attendance using facial recognition technology. The system leverages MTCNN for face detection and FaceNet for facial recognition.

## Team Members
- Sandra
- Saniya Sebastian
- Shreya Gem Mathew
- Vinsu Susan Thomas

## Features
- **Face Registration:** Capture multiple photos of a user and apply various augmentations to create robust embeddings.
- **Attendance Taking:** Recognize faces in real-time and record attendance.
- **Attendance Display:** Display the attendance records and identify absentees.

## Requirements
- Python 3.7+
- Streamlit
- OpenCV
- Requests
- Pillow
- Numpy
- Pandas
- Torch
- MTCNN
- Facenet-pytorch
- Scikit-learn

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/SHREYA-GEM/AttendEase.git
    cd AttendEase
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

2. Use the web interface to register users and take attendance.

## Code Structure
- `app.py`: Main application file.
- `attendance.csv`: CSV file to store attendance records.
- `embedd_earlymorning.csv`: CSV file to store face embeddings.

## How It Works
1. **Registration:** 
    - Enter the user's name and roll number.
    - Capture multiple photos and apply augmentations.
    - Extract embeddings and save them to a CSV file.

2. **Attendance:**
    - Capture real-time video.
    - Preprocess and detect faces.
    - Recognize faces and update attendance records.
    - Display the attendance table and list absentees.

## Contributions
Feel free to fork this project and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

