import os
import csv
import argparse
import statistics

def check_data(base_dir, csv_filename):
    """
    base_dir: The parent directory where data is stored (e.g., ../data/20230223_1015)
    csv_filename: The name of the CSV file inside base_dir (e.g., data.csv)
    """

    csv_path = os.path.join(base_dir, csv_filename)

    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return

    # 1) Open the CSV file
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)

    # Parse the header (first line)
    header = rows[0]
    data_rows = rows[1:] # Actual data

    # 2) Prepare to check timestamp/missing values/image & depth files for each row
    timestamps = []
    missing_counts_per_column = [0] * len(header) # Missing count for each column
    image_front_missing_count = 0
    depth_rs2_missing_count = 0
    # ----------------- ADDED -----------------
    realsense_color_missing_count = 0
    realsense_depth_missing_count = 0
    # -----------------------------------------

    # Assume timestamps are integers in MS (milliseconds)
    # Skip or handle exceptions if conversion from string to int fails
    valid_timestamps = []

    for row in data_rows:
        # Example row: [timestamp, odom_pos_x, odom_pos_y, ...]
        if not row:
            continue

        ts_str = row[0] # timestamp (string)
        timestamps.append(ts_str)

        # Check for missing values
        for i, val in enumerate(row):
            # Consider 'nan' or an empty string as missing
            if val == 'nan' or val.strip() == '':
                missing_counts_per_column[i] += 1

        # Check for image file existence
        front_img_path = os.path.join(base_dir, "images", "front", f"{ts_str}.jpg")
        if not os.path.isfile(front_img_path):
            image_front_missing_count += 1

        # Depth RS2
        rs2_depth_path = os.path.join(base_dir, "depth", "rs2", f"{ts_str}.npy")
        if not os.path.isfile(rs2_depth_path):
            depth_rs2_missing_count += 1

        # ----------------- ADDED -----------------
        # RealSense Color
        realsense_color_path = os.path.join(base_dir, "images", "realsense_color", f"{ts_str}.jpg")
        if not os.path.isfile(realsense_color_path):
            realsense_color_missing_count += 1

        # RealSense Depth
        realsense_depth_path = os.path.join(base_dir, "depth", "realsense_depth", f"{ts_str}.npy")
        if not os.path.isfile(realsense_depth_path):
            realsense_depth_missing_count += 1
        # -----------------------------------------

        # Convert timestamp to integer
        try:
            ts_int = int(ts_str)
            valid_timestamps.append(ts_int)
        except (ValueError, IndexError):
            pass # Ignore (or log) if conversion fails

    row_count = len(data_rows)
    if row_count == 0:
        print("No data rows found in the CSV.")
        return

    # 3) Calculate missing count/ratio per column
    print("\n=== Column Missing Value Information ===")
    for i, col_name in enumerate(header):
        missing_count = missing_counts_per_column[i]
        missing_ratio = (missing_count / row_count * 100) if row_count > 0 else 0
        print(f"Column [{col_name}] => Missing: {missing_count}/{row_count} "
              f"({missing_ratio:.2f}%)")

    # 4) Image/Depth file missing info
    print("\n=== Image/Depth File Missing Information ===")
    print(f"- Front Image Missing: {image_front_missing_count}/{row_count} "
          f"({(image_front_missing_count / row_count) * 100:.2f}%)")
    print(f"- RS2 Depth Missing: {depth_rs2_missing_count}/{row_count} "
          f"({(depth_rs2_missing_count / row_count) * 100:.2f}%)")
    # ----------------- ADDED -----------------
    print(f"- RealSense Color Missing: {realsense_color_missing_count}/{row_count} "
          f"({(realsense_color_missing_count / row_count) * 100:.2f}%)")
    print(f"- RealSense Depth Missing: {realsense_depth_missing_count}/{row_count} "
          f"({(realsense_depth_missing_count / row_count) * 100:.2f}%)")
    # -----------------------------------------

    # 5) Timestamp interval statistics (to check if it's near 10Hz = 100ms)
    # Sort valid_timestamps and calculate differences between consecutive values
    valid_timestamps.sort()
    time_diffs = [valid_timestamps[i] - valid_timestamps[i-1] for i in range(1, len(valid_timestamps))]

    if not time_diffs:
        print("\nNot enough data to calculate timestamp intervals.")
        return

    avg_diff = statistics.mean(time_diffs)
    stdev_diff = statistics.pstdev(time_diffs) # Population standard deviation
    min_diff = min(time_diffs)
    max_diff = max(time_diffs)

    print("\n=== Sample Interval Statistics (based on timestamp) ===")
    print(f"- Row Count: {row_count}")
    print(f"- Average Interval: {avg_diff:.1f} ms (Theoretical: 100ms)")
    print(f"- Standard Deviation: {stdev_diff:.1f} ms")
    print(f"- Min/Max Interval: {min_diff} ms / {max_diff} ms")


def find_latest_data_dir(parent_dir="../data"):
    """Finds the most recently created directory in the parent data directory."""
    try:
        all_subdirs = [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]
        if not all_subdirs:
            return None
        # The directory names (YYYYMMDD_HHMM) can be sorted alphabetically
        latest_dir = max(all_subdirs)
        return os.path.join(parent_dir, latest_dir)
    except FileNotFoundError:
        return None

def main():
    parser = argparse.ArgumentParser(description="Check the integrity of collected ROS2 data.")
    parser.add_argument('--base_dir', type=str,
                        help="Parent directory where data is stored. Defaults to the most recent one.")
    parser.add_argument('--csv_file', type=str, default="data.csv",
                        help="Name of the CSV file to check. Defaults to 'data.csv'.")
    args = parser.parse_args()

    base_dir_to_check = args.base_dir
    if base_dir_to_check is None:
        print("No base directory specified. Searching for the latest directory...")
        base_dir_to_check = find_latest_data_dir()
        if base_dir_to_check is None:
            print("Error: Could not find any data directories in '../data/'.")
            return

    print(f"\n--- Checking data in directory: {os.path.abspath(base_dir_to_check)} ---")
    check_data(base_dir_to_check, args.csv_file)


if __name__ == "__main__":
    main()
