# 📱 Mobile Price Predictor – ML Web App

A machine learning-powered web application that predicts the **price range of mobile phones** based on their specifications. Built with **Python**, **Streamlit**, and **Scikit-learn**, this project enables interactive price range predictions using a trained classification model.

---
## 📸 App View

<div align="center">
  <img src="assets/screenshot1.png" width="30%" style="margin-right: 10px;" />
  <img src="assets/screenshot2.png" width="30%" style="margin-right: 10px;" />
  <img src="assets/screenshot3.png" width="30%" />
</div>

---

## 📌 Project Overview

This application allows users to input mobile phone specifications such as RAM, battery, camera quality, screen size, etc., and receive a predicted price category:
- 🟢 **Low**: ₹5,000 – ₹12,000  
- 🔵 **Medium**: ₹12,000 – ₹20,000  
- 🟠 **High**: ₹20,000 – ₹35,000  
- 🔴 **Very High**: ₹35,000+

It is designed to assist consumers, sellers, and manufacturers in making informed pricing decisions.

---

## 🎯 Project Goals

- Build an accurate classification model using mobile phone specs.
- Deploy a responsive web interface using **Streamlit**.
- Visualize feature importance and model confidence.
- Provide an easy-to-use prediction system for non-technical users.

---

## 📊 Dataset Overview

- **Source:** - Restricted by Comapny  
- **Total Records:** 2000
- **Target Column:** `price_range` (0 = Low, 1 = Medium, 2 = High, 3 = Very High)
- **Features Include:**
  - `ram`, `battery_power`, `px_height`, `px_width`, `fc`, `pc`
  - `bluetooth`, `wifi`, `touch_screen`, `dual_sim`, `four_g`, `three_g`
  - `int_memory`, `clock_speed`, `n_cores`, `m_dep`, etc.

---

## 🧠 Machine Learning Model

- **Model Type:** Random Forest Classifier (likely, as inferred from `.feature_importances_`)
- **Preprocessing:** 
  - Feature scaling using `StandardScaler`
  - Derived features: `screen_area`, `resolution`
- **Accuracy:** ~85-90% (based on model quality assumption)

---

## 🧰 Tech Stack

| Layer      | Tools/Technologies |
|------------|-------------------|
| Language   | Python 3.x        |
| ML Library | scikit-learn      |
| UI         | Streamlit         |
| Visualization | Matplotlib, Seaborn |
| Serialization | Pickle          |

---

## 📂 Project Structure

```

├── app.py                      # Streamlit frontend
├── main.ipynb                 # Model training notebook
├── mobile\_price\_model.pkl     # Trained classifier model
├── scaler.pkl                 # Scaler for preprocessing
├── feature\_names.pkl          # List of feature names
├── price\_range\_dict.pkl       # Mapping of labels to price categories
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation

````

---

## 🛠️ Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/AritraOfficial/Mobile-Price-Predictor-ML-App.git
cd Mobile-Price-Predictor-ML-App
````

2. **Create a virtual environment & install dependencies**

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Run the application**

```bash
streamlit run app.py
```

4. **Train model (optional)**
   Open `main.ipynb` and run all cells to retrain the model.

---

## 📈 Feature Importance

![Feature Importance](https://github.com/user-attachments/assets/368d3fac-b60b-46ed-87ef-543bd901dbe3)


The model identifies **RAM**, **battery power**, and **pixel resolution** as top contributors to price prediction.

---

## 🧩 Future Improvements

* Add deep learning or ensemble models (e.g., XGBoost, LightGBM)
* Include brand, OS, release year as input features
* Deploy on cloud platforms (Streamlit Cloud, Render, AWS)
* Add model retraining pipeline with new data

---

## 👨‍💻 Author
For queries or collaborations, feel free to connect:  
<p align="center">
  <a href="https://www.linkedin.com/in/aritramukherjeeofficial/" target="_blank">
    <img src="https://img.shields.io/badge/LinkedIn-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn">
  </a>
  <a href="https://x.com/AritraMofficial" target="_blank">
    <img src="https://img.shields.io/badge/Twitter-%231DA1F2.svg?style=for-the-badge&logo=twitter&logoColor=white" alt="Twitter">
  </a>
  <a href="https://www.instagram.com/aritramukherjee_official/?__pwa=1" target="_blank">
    <img src="https://img.shields.io/badge/Instagram-%23E4405F.svg?style=for-the-badge&logo=instagram&logoColor=white" alt="Instagram">
  </a>
  <a href="https://leetcode.com/u/aritram_official/" target="_blank">
    <img src="https://img.shields.io/badge/LeetCode-%23FFA116.svg?style=for-the-badge&logo=leetcode&logoColor=white" alt="LeetCode">
  </a>
  <a href="https://github.com/AritraOfficial" target="_blank">
    <img src="https://img.shields.io/badge/GitHub-%23181717.svg?style=for-the-badge&logo=github&logoColor=white" alt="GitHub">
  </a>
  <a href="https://discord.com/channels/@me" target="_blank">
    <img src="https://img.shields.io/badge/Discord-%237289DA.svg?style=for-the-badge&logo=discord&logoColor=white" alt="Discord">
  </a>
  <a href="mailto:aritra.work.official@gmail.com" target="_blank">
    <img src="https://img.shields.io/badge/Email-%23D14836.svg?style=for-the-badge&logo=gmail&logoColor=white" alt="Email">
  </a>
</p>

## 📄 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.


---

**If you find this project useful, please consider giving it a ⭐!**
