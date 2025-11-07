import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

def plot_battery_percentage(csv_path):
    """
    Reads battery CSV data and saves a line plot of
    'battery_percentage' over time.
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

    # Convert 'nan' strings or empty values to NaN and drop them
    df['battery_percentage'] = pd.to_numeric(df['battery_percentage'], errors='coerce')
    df = df.dropna(subset=['timestamp', 'battery_percentage'])

    if len(df) < 2:
        print("Error: Less than 2 valid data points. Cannot create plot.")
        return

    # --- 2. Convert Time Data (Relative Time in Minutes) ---
    df['timestamp'] = pd.to_numeric(df['timestamp'])
    start_time_ms = df['timestamp'].min()
    df['time_minutes'] = (df['timestamp'] - start_time_ms) / (1000 * 60)

    X = df['time_minutes']
    y = df['battery_percentage']

    # --- 3. Print Basic Info to Console ---
    print("\n--- 🔋 Battery Data Summary ---")
    print(f"Data Collection Duration: {X.min():.2f} min to {X.max():.2f} min (Total {X.max() - X.min():.2f} min)")
    print(f"Battery Change: {y.iloc[0]:.2f}% -> {y.iloc[-1]:.2f}%")
    print("---------------------------------")

    # --- 4. Create and Save Plot ---
    plt.figure(figsize=(12, 7))

    # Plot 'battery_percentage' as a line plot
    plt.plot(X, y, label='Battery Percentage', color='blue', linewidth=2)

    # Plot Styling
    plt.title('Battery Percentage Over Time', fontsize=16)
    plt.xlabel('Time (Minutes)', fontsize=12)
    plt.ylabel('Battery Percentage (%)', fontsize=12)
    plt.ylim(0, 105) # Set y-axis limits
    plt.xlim(left=0) # Set x-axis start to 0
    plt.grid(True, linestyle=':')
    plt.legend(fontsize=10)

    # --- 5. Save Plot to File ---
    output_dir = os.path.dirname(csv_path)
    if not output_dir:
        output_dir = "."
        
    plot_filename = "battery_percentage_plot.png"
    plot_save_path = os.path.join(output_dir, plot_filename)
    
    plt.savefig(plot_save_path, dpi=150)
    print(f"\n✅ Battery percentage plot saved to:\n{os.path.abspath(plot_save_path)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyzes battery log CSV and saves a percentage plot.")
    parser.add_argument(
        "csv_file", 
        nargs='?',              # Makes the argument optional
        default="battery_data.csv",  # Default value if no argument is given
        type=str, 
        help="Path to the battery_data.csv file to analyze (default: battery.csv)"
    )
    
    args = parser.parse_args()
    
    # Run the analysis function
    plot_battery_percentage(args.csv_file)
