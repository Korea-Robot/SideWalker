import os
import csv
import argparse
import statistics

def check_data(base_dir, csv_filename):
    """
    base_dir: 데이터가 저장된 상위 디렉토리 (예: ../data/20230223_1015)
    csv_filename: base_dir 내부의 CSV 파일 이름 (예: data_20230223_1015.csv)
    """

    csv_path = os.path.join(base_dir, csv_filename)

    # 1) CSV 파일 열기
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)

    # 첫 줄(헤더) 파싱
    header = rows[0]
    data_rows = rows[1:]  # 실제 데이터

    # 2) 각 행의 timestamp / 결측값 / 이미지·뎁스 파일 체크할 준비
    timestamps = []
    missing_counts_per_column = [0] * len(header)  # 각 컬럼별 결측 카운트
    image_front_missing_count = 0
    depth_rs2_missing_count = 0

    # 타임스탬프(문자열) -> 정수 변환 실패하면 스킵, 혹은 예외처리
    # 여기서는 MS(밀리초) 단위의 int 형식이라고 가정
    valid_timestamps = []

    for row in data_rows:
        # row 예시: [timestamp, odom_pos_x, odom_pos_y, ...]
        if not row:
            continue

        ts_str = row[0]  # timestamp (문자열)
        timestamps.append(ts_str)

        # 결측값 체크
        for i, val in enumerate(row):
            # 'nan'이거나 빈 문자열이면 결측으로 간주
            if val == 'nan' or val.strip() == '':
                missing_counts_per_column[i] += 1

        # 이미지 파일 존재 여부
        front_img_path = os.path.join(base_dir, "images", "front", f"{ts_str}.jpg")
        if not os.path.isfile(front_img_path):
            image_front_missing_count += 1

        # 뎁스 RS2
        rs2_depth_path = os.path.join(base_dir, "depth", "rs2", f"{ts_str}.npy")
        if not os.path.isfile(rs2_depth_path):
            depth_rs2_missing_count += 1

        # 타임스탬프 정수 변환
        try:
            ts_int = int(ts_str)
            valid_timestamps.append(ts_int)
        except:
            pass  # 변환 실패시 무시(또는 로그)

    row_count = len(data_rows)
    if row_count == 0:
        print("CSV에 데이터 행이 없습니다.")
        return

    # 3) 칼럼별 결측 개수/비율 계산
    print("\n=== 칼럼별 결측 정보 ===")
    for i, col_name in enumerate(header):
        missing_count = missing_counts_per_column[i]
        missing_ratio = missing_count / row_count * 100
        print(f"Column [{col_name}] => 결측: {missing_count}/{row_count} "
              f"({missing_ratio:.2f}%)")

    # 4) 이미지/뎁스 누락
    print("\n=== 이미지/뎁스 누락 정보 ===")
    print(f"- Front 이미지 누락: {image_front_missing_count}/{row_count} "
          f"({(image_front_missing_count / row_count) * 100:.2f}%)")
    print(f"- RS2 뎁스 누락: {depth_rs2_missing_count}/{row_count} "
          f"({(depth_rs2_missing_count / row_count) * 100:.2f}%)")

    # 5) 타임스탬프 간격(10Hz=100ms 근처인지) 통계
    # valid_timestamps를 정렬 후, 연속된 차이 계산
    valid_timestamps.sort()
    time_diffs = []
    for i in range(1, len(valid_timestamps)):
        diff = valid_timestamps[i] - valid_timestamps[i - 1]
        time_diffs.append(diff)

    if not time_diffs:
        print("\n타임스탬프 간격을 계산할 데이터가 부족합니다.")
        return

    avg_diff = statistics.mean(time_diffs)
    stdev_diff = statistics.pstdev(time_diffs)  # 모표준편차(pstdev)
    min_diff = min(time_diffs)
    max_diff = max(time_diffs)

    print("\n=== 샘플 간격(타임스탬프 기준) 통계 ===")
    print(f"- 행 개수: {row_count}")
    print(f"- 평균 간격: {avg_diff:.1f} ms (이론 100ms)")
    print(f"- 표준편차: {stdev_diff:.1f} ms")
    print(f"- 최소/최대: {min_diff} ms / {max_diff} ms")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, required=True,
                        help="데이터가 저장된 상위 디렉토리 (예: ",default="../data/sample")
    parser.add_argument('--csv_file', type=str, required=True,
                        help="체크할 CSV 파일 이름 (예: ",default = "test1.csv")
    args = parser.parse_args()

    check_data(args.base_dir, args.csv_file)


if __name__ == "__main__":
    main()
