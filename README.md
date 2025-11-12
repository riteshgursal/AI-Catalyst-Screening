# 🧪 AI-Accelerated Electrocatalyst Screening and Stability Predictor

This project establishes a computational pipeline for **Data-Driven Material Discovery**, demonstrating the application of Machine Learning to accelerate research in electrochemistry and materials science.

It showcases the ability to use predictive modeling to *screen* candidate materials based on simulated intrinsic properties, guiding experimental efforts and saving significant lab time a critical need in **Li-based Battery** and **Electrocatalyst** research.

## 🔬 Alignment with P16 Research (Computational Chemistry)

The core value of this project for experimental labs is acceleration:

* **Predictive Screening:** Uses AI to forecast material performance (Catalytic Activity) based on features like d-band center, focusing experimental efforts only on high-potential candidates.
* **Computational Modeling:** Demonstrates proficiency in building robust regression models (Random Forest) to handle complex, non-linear dependencies often found in materials data.
* **Data Analysis Workflow:** Establishes a clear workflow for ingesting raw data, training a model, and outputting actionable insights for experimental teams.

## 🛠️ How to Execute

### 1. Setup
1.  **Clone the Repository** and navigate into the `ai-catalyst-screening` folder.
2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### 2. Run the Project
The entire analysis, training, and visualization process is executed with a single command:

```bash
python main.py
