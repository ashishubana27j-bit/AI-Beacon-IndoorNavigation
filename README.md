# AI-Beacon-IndoorNavigation
Indoor Navigation System using BLE Beacons and XGBoost ML for real-time indoor positioning. The app predicts user location from beacon RSSI and sensor data, then provides Google Maps–style turn-by-turn navigation with shortest-path routing. Designed for malls, hospitals, airports, offices, and smart buildings.

**Indoor Navigation System using BLE Beacons and XGBoost**

An intelligent indoor positioning and navigation solution that leverages Bluetooth Low Energy (BLE) beacons, machine learning (XGBoost), and mobile device sensors to provide real-time indoor location tracking and turn-by-turn navigation inside buildings.

🚀 Features
Real-time indoor positioning using BLE beacon signals
XGBoost-based machine learning model for location prediction
Beacon RSSI fingerprinting and signal processing
Turn-by-turn indoor navigation similar to Google Maps
Shortest path calculation using graph-based routing algorithms
Multi-floor navigation support
Sensor fusion using Accelerometer, Gyroscope, and Magnetometer
Dynamic user position updates while moving
Interactive floor map visualization
Scalable architecture for malls, hospitals, airports, offices, universities, and smart buildings
🏗️ System Architecture
BLE beacons broadcast signals throughout the building.
Mobile devices continuously scan nearby beacon RSSI values.
Sensor data is collected from the device to improve positioning accuracy.
An XGBoost model predicts the user's current location based on beacon fingerprints.
The navigation engine calculates the optimal route from the current location to the selected destination.
The application provides real-time visual navigation and route updates.
🧠 Machine Learning Pipeline
Data Collection from BLE beacons
RSSI Feature Engineering
Data Preprocessing and Normalization
XGBoost Regression Model Training
Real-Time Location Prediction
Continuous Position Refinement using Sensor Fusion
📱 Technologies Used
React Native
BLE Beacon Technology
XGBoost
Python
FastAPI
Indoor Mapping
Graph Pathfinding Algorithms
Accelerometer & Gyroscope Sensors
REST APIs
🎯 Use Cases
Shopping Malls
Hospitals
Airports
Universities
Corporate Offices
Warehouses
Smart Buildings
Exhibition Centers
🔥 Key Benefits
GPS-independent indoor positioning
High-accuracy location prediction
Low infrastructure cost using BLE beacons
Real-time route guidance
Extensible and scalable architecture
Improved user experience in complex indoor environments
📊 Project Goal

The goal of this project is to provide an accurate, scalable, and intelligent indoor navigation system that combines BLE beacon technology, machine learning, and sensor fusion to deliver seamless navigation experiences in environments where GPS signals are unavailable or unreliable.
