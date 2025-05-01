# Nearby Friends Recommendation System

Ever thought how apps like Facebook or Meetup suggest people you might know? This project simulates that smart magic by using **K-Nearest Neighbors (KNN)** and **Geolocation Data** to recommend nearby friends based on shared interests and proximity.

## 🚀 Project Highlights

- 🔍 **Core Algorithm**: K-Nearest Neighbors (KNN)
- 📍 **Location Awareness**: Haversine formula to calculate real-world distances
- 🧠 **Smart Matching**: Combines **location** and **user interests** to suggest relevant connections
- 📊 **Data Processing**: Clean and efficient data preprocessing pipeline
- 🛠️ **Modular Codebase**: Easy to plug into real-world applications

## 🧪 How It Works

1. **Input Data**: User profiles with coordinates (latitude, longitude) and interests
2. **Distance Calculation**: Apply Haversine formula to measure real distance
3. **KNN Algorithm**: Identify the K closest users with similar interests
4. **Output**: List of recommended friends for each user

## 🖥️ Technologies Used

- Python 🐍
- Pandas & NumPy
- Scikit-learn
- Matplotlib (for visualizations)
- Haversine formula (geospatial computation)

## 📸 Sample Output

```plaintext
👤 User: Alice
✅ Recommended Friends:
 - Bob (2.4 km away)
 - Carol (3.1 km away)

git clone https://github.com/yourusername/nearby_friends_knn.git
cd nearby_friends_knn
pip install -r requirements.txt
