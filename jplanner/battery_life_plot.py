import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import numpy as np
import os
import argparse

def analyze_battery_life(csv_path):
    """
    Reads battery CSV data, performs linear regression,
    and saves a plot predicting total runtime.
    """
    
    # --- 1. Load and Preprocess Data ---
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: File not found. '{csv_path}'")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: File is empty. '{csv_path}'")
        return

    df['battery_percentage'] = pd.to_numeric(df['battery_percentage'], errors='coerce')
    df = df.dropna(subset=['timestamp', 'battery_percentage'])

    if len(df) < 2:
        print("Error: Less than 2 valid data points. Cannot perform regression.")
        return

    # --- 2. Convert Time Data (Relative Time in Minutes) ---
    start_time_ms = df['timestamp'].min()
    df['time_minutes'] = (df['timestamp'] - start_time_ms) / (1000 * 60)

    X = df['time_minutes']
    y = df['battery_percentage']

    # --- 3. Linear Regression Analysis ---
    slope, intercept, r_value, p_value, std_err = linregress(X, y)

    if slope >= 0:
        print("Error: Battery percentage is not decreasing (or data is noisy).")
        print("Stopping prediction.")
        return

    # --- 4. Predict Runtime (Quantitative Metric) ---
    # Calculate total time from 100% to 0%
    # Time = (End % - Start %) / Slope
    total_minutes_from_100 = (0 - 100) / slope
    total_hours_from_100 = total_minutes_from_100 / 60

    # Calculate time from log start (intercept) to 0%
    total_minutes_from_log_start = (0 - intercept) / slope

    # --- 5. Print Results to Console ---
    print("\n--- 🔋 Battery Life Prediction Results ---")
    print(f"Data Collection Duration: {X.min():.2f} min to {X.max():.2f} min (Total {X.max() - X.min():.2f} min)")
    print(f"Battery Change: {y.iloc[0]:.2f}% -> {y.iloc[-1]:.2f}%")
    print(f"Analyzed Slope: {slope:.4f} %/min")
    print(f"R-squared (Accuracy): {r_value**2:.4f}")
    print("-----------------------------------------")
    print("📊 Quantitative Metric (100% -> 0%):")
    is_target_met = total_minutes_from_100 >= 120
    print(f"  > Total Predicted Runtime: {total_minutes_from_100:.2f} minutes ({total_hours_from_100:.2f} hours)")
    print(f"  > 2-Hour (120 min) Target Met: {'✅ Yes' if is_target_met else '❌ No'}")
    print("-----------------------------------------")

    # --- 6. Create and Save Plot ---
    plt.figure(figsize=(12, 7))

    # 1. Actual collected data (scatter plot)
    plt.scatter(X, y, label='Actual Data', alpha=0.7)

    # 2. Fitted line (within data range)
    fit_x = np.array([X.min(), X.max()])
    fit_y = slope * fit_x + intercept
    plt.plot(fit_x, fit_y, color='blue', linestyle='-', linewidth=2, label='Fitted Line')

    # 3. Extrapolation (from log start to 0%)
    extrap_x = np.array([0, total_minutes_from_log_start])
    extrap_y = slope * extrap_x + intercept
    plt.plot(extrap_x, extrap_y, color='red', linestyle='--', linewidth=2, label='Extrapolation to 0%')

    # Plot Styling
    plt.title('Battery Discharge Prediction', fontsize=16)
    plt.xlabel('Time (Minutes)', fontsize=12)
    plt.ylabel('Battery Percentage (%)', fontsize=12)
    plt.ylim(0, 105)
    plt.xlim(left=0)
    plt.grid(True, linestyle=':')
    plt.legend(fontsize=10)

    # Text box (prediction results)
    text_content = (
        f"--- Prediction (100% -> 0%) ---\n"
        f"Total Runtime: {total_minutes_from_100:.2f} min\n"
        f"({total_hours_from_100:.2f} hours)\n"
        f"2-Hour Target Met: {'YES' if is_target_met else 'NO'}"
    )
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
    plt.text(0.95, 0.95, text_content, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', horizontalalignment='right', bbox=props)

    # --- 7. Save Plot to File ---
    output_dir = os.path.dirname(csv_path)
    if not output_dir:
        output_dir = "."
        
    plot_filename = "battery_prediction_plot2.png"
    plot_save_path = os.path.join(output_dir, plot_filename)
    
    plt.savefig(plot_save_path, dpi=150)
    print(f"\n✅ Prediction plot saved to:\n{os.path.abspath(plot_save_path)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyzes battery log CSV, predicts operating time, and saves a plot.")
    parser.add_argument(
        "csv_file", 
        nargs='?',              # Makes the argument optional
        default="battery_data.csv",  # Default value if no argument is given
        type=str, 
        help="Path to the battery_data.csv file to analyze (default: battery.csv)"
    )
    
    args = parser.parse_args()
    
    # Run the analysis function
    analyze_battery_life(args.csv_file)
