import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

def plot_battery_percentage(csv_path):
    """
    배터리 CSV 데이터를 읽어 'battery_percentage'를
    시간에 따라 라인 플롯으로 그리고 저장합니다.
    """
    
    # --- 1. 데이터 불러오기 및 전처리 ---
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"오류: 파일을 찾을 수 없습니다. '{csv_path}'")
        return
    except pd.errors.EmptyDataError:
        print(f"오류: 파일이 비어있습니다. '{csv_path}'")
        return

    # 'nan' 문자열이나 비어있는 값을 NaN으로 변환 후 제거
    df['battery_percentage'] = pd.to_numeric(df['battery_percentage'], errors='coerce')
    df = df.dropna(subset=['timestamp', 'battery_percentage'])

    if len(df) < 2:
        print("오류: 유효한 데이터가 2개 미만이라 플롯을 그릴 수 없습니다.")
        return

    # --- 2. 시간 데이터 변환 (상대 시간, 분 단위) ---
    df['timestamp'] = pd.to_numeric(df['timestamp'])
    start_time_ms = df['timestamp'].min()
    df['time_minutes'] = (df['timestamp'] - start_time_ms) / (1000 * 60)

    X = df['time_minutes']
    y = df['battery_percentage']

    # --- 3. 콘솔에 기본 정보 출력 ---
    print("\n--- 🔋 배터리 데이터 요약 ---")
    print(f"데이터 수집 시간: {X.min():.2f} min ~ {X.max():.2f} min (총 {X.max() - X.min():.2f} 분)")
    print(f"배터리 변화: {y.iloc[0]:.2f}% -> {y.iloc[-1]:.2f}%")
    print("---------------------------------")

    # --- 4. 플롯 생성 및 저장 ---
    plt.figure(figsize=(12, 7))

    # 'battery_percentage'를 라인 플롯으로 그리기
    plt.plot(X, y, label='배터리 잔량 (Battery Percentage)', color='blue', linewidth=2)

    # 플롯 스타일링
    plt.title('배터리 잔량 변화 (Battery Percentage Over Time)', fontsize=16)
    plt.xlabel('운용 시간 (Minutes)', fontsize=12)
    plt.ylabel('배터리 잔량 (%)', fontsize=12)
    plt.ylim(0, 105) # y축을 0% ~ 105%로 고정
    plt.xlim(left=0) # x축 시작을 0으로 고정
    plt.grid(True, linestyle=':')
    plt.legend(fontsize=10)

    # --- 5. 플롯 파일로 저장 ---
    output_dir = os.path.dirname(csv_path)
    if not output_dir:
        output_dir = "."
        
    plot_filename = "battery_percentage_plot.png"
    plot_save_path = os.path.join(output_dir, plot_filename)
    
    plt.savefig(plot_save_path, dpi=150)
    print(f"\n✅ 배터리 잔량 플롯이 저장되었습니다:\n{os.path.abspath(plot_save_path)}")


if __name__ == "__main__":
    # 사용자가 CSV 파일 경로를 터미널 인자로 전달하도록 설정
    parser = argparse.ArgumentParser(description="배터리 로그 CSV를 분석하여 잔량 변화 플롯을 저장합니다.")
    parser.add_argument(
        "csv_file", 
        nargs='?',              # 인자가 없으면 기본값을 사용
        default="battery_data.csv",  # 기본값 "battery.csv"
        type=str, 
        help="분석할 battery_data.csv 파일의 경로 (기본값: battery.csv)"
    )
    
    args = parser.parse_args()
    
    # 분석 함수 실행
    plot_battery_percentage(args.csv_file)
