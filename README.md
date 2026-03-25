# Traffic Violation Detection System

## Overview

This project is a traffic violation detection system built using Python. It can detect different types of vehicles and identify common traffic violations like helmet absence and triple riding from images or videos.

The project uses computer vision and a web interface to make it easy to upload and check data.

---

## Features

* Detects vehicles like cars, bikes, and trucks
* Helmet detection (with or without helmet)
* Detects triple riding on bikes
* Works with both images and videos
* Simple web interface using Flask
* Stores detected violations

---

## Technologies Used

* Python
* OpenCV
* YOLOv8
* Flask
* HTML / CSS

---

## Project Structure

```
traffic-violation-detection/
│── app.py
│── detection.py
│── requirements.txt
│── violations.db
│── violations.json
│
├── templates/
├── static/
```

---

## How to Run

1. Clone the repository:

```
git clone https://github.com/your-username/traffic-violation-detection.git
cd traffic-violation-detection
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Run the project:

```
python app.py
```

4. Open in browser:

```
http://127.0.0.1:5000/
```

---


## Notes

* Make sure all required libraries are installed
* Do not upload large files like `.pt` models to GitHub
* You can add your own images/videos for testing

---

## Contributors

* Harshit Bora
* Shaanvi Bisht
* Anshika Garg
* Kanan Malik
---

## License

This project is licensed under the MIT License.
