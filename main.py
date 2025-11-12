# main.py: AI-Accelerated Electrocatalyst Screening and Stability Predictor

# --- 1. IMPORTS ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import os 
import joblib 

# --- 2. CONFIGURATION ---
MODEL_PATH = 'output/catalyst_model.joblib'
OUTPUT_DIR = 'output'

# --- 3. DATA GENERATION (P16 Research Simulation) ---
def generate_material_data(n_materials=50):
    """
    Generates synthetic data for electrocatalysts (P16 research).
    Features are simplified material properties. Target is Catalytic Activity.
    """
    # Ensures the data is the same every time you run it
    np.random.seed(42) 
    
    # Feature 1: d-band center (Key parameter in catalysis)
    d_band = np.random.uniform(-1.0, 1.0, n_materials) 
    # Feature 2: Surface energy (Affects stability)
    surface_energy = np.random.uniform(1.0, 3.0, n_materials)
    
    # Target: Catalytic Activity is modeled as a function of the features
    activity = 0.5 * d_band - 0.2 * surface_energy + 1.0 + np.random.normal(0, 0.1, n_materials)
    
    df = pd.DataFrame({
        'd_band_center_eV': d_band,
        'Surface_Energy_J/m2': surface_energy,
        'Catalytic_Activity_Target': activity
    })
    
    df.to_csv(os.path.join(OUTPUT_DIR, 'simulated_catalyst_data.csv'), index=False)
    print("✅ Material data generated and saved: Features ready for screening.")
    return df

# --- 4. MODEL TRAINING (ML Core Logic) ---
def train_screening_model(df):
    """
    Trains a Random Forest Regressor to screen and predict material activity.
    """
    X = df[['d_band_center_eV', 'Surface_Energy_J/m2']]
    y = df['Catalytic_Activity_Target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Random Forest Regressor for robust non-linear modeling
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluation
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print(f"\n--- Material Screening Model Evaluation ---")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R-squared Score (R2): {r2:.4f}")
    print("------------------------------------------")
    
    # Identify promising materials (Top 5%)
    all_predictions = model.predict(X)
    top_threshold = np.percentile(all_predictions, 95)
    top_materials = df[all_predictions >= top_threshold].index.tolist()
    
    print(f"🧪 Predicted Top 5% Materials (Ready for Experiment): IDs {top_materials}")
    
    # Save Model
    joblib.dump(model, MODEL_PATH)
    return model, X_test, y_test

# --- 5. VISUALIZATION ---
def visualize_results(model, X_test, y_test):
    """Plots true vs predicted activity to assess model quality."""
    predictions = model.predict(X_test)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predictions, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    
    plt.xlabel("True Catalytic Activity")
    plt.ylabel("Predicted Catalytic Activity")
    plt.title("AI-Accelerated Material Screening (True vs. Predicted)")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plot_path = os.path.join(OUTPUT_DIR, 'catalyst_prediction_plot.png')
    plt.savefig(plot_path)
    print(f"✅ Prediction plot saved to {plot_path}")

# --- 6. WORKFLOW FUNCTION ---
def run_screening_project():
    # Ensure the output folder exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("--- 🧪 AI-Accelerated Electrocatalyst Screening Start ---")
    
    # 1. Data Generation
    df = generate_material_data() 
    
    # 2. Model Training
    model, X_test, y_test = train_screening_model(df)
    
    # 3. Visualization
    visualize_results(model, X_test, y_test)
    
    print("\nProject execution complete. Ready for GitHub.")
    
# --- 7. EXECUTION BLOCK ---
if __name__ == "__main__":
    run_screening_project()