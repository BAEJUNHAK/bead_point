#!/usr/bin/env python3
"""
추론 실행 예제 스크립트
다양한 사용 사례를 보여줍니다.
"""
import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """명령어 실행 및 결과 출력"""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"📝 명령어: {' '.join(cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("stderr:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 실행 실패: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False

def main():
    """메인 실행 함수"""
    print("🚀 비드 포인트 검출 추론 예제 실행")
    
    # 기본 경로 설정
    base_dir = Path.cwd()
    train_img_dir = base_dir / "data" / "train" / "img"
    left_img_dir = base_dir / "data" / "Left" / "191412" / "img"
    right_img_dir = base_dir / "data" / "Right" / "191013" / "img"
    
    # 사용 가능한 데이터 디렉토리 찾기
    available_dirs = []
    test_dirs = [
        ("훈련 데이터", train_img_dir),
        ("Left 191412", left_img_dir), 
        ("Right 191013", right_img_dir),
        ("Left 191532", base_dir / "data" / "Left" / "191532" / "img"),
        ("Right 191412", base_dir / "data" / "Right" / "191412" / "img"),
        ("Right 191532", base_dir / "data" / "Right" / "191532" / "img"),
    ]
    
    for name, path in test_dirs:
        if path.exists() and any(path.glob("*.png")):
            available_dirs.append((name, path))
            print(f"✅ 발견: {name} ({path})")
        else:
            print(f"❌ 없음: {name} ({path})")
    
    if not available_dirs:
        print("❌ 사용 가능한 이미지 디렉토리를 찾을 수 없습니다.")
        return
    
    print(f"\n📂 총 {len(available_dirs)}개의 데이터 디렉토리 발견")
    
    # 예제 1: 단일 이미지 추론
    print("\n" + "="*80)
    print("📄 예제 1: 단일 이미지 추론")
    print("="*80)
    
    # 첫 번째 사용 가능한 디렉토리에서 첫 번째 이미지 선택
    first_dir_name, first_dir = available_dirs[0]
    first_image = None
    for img_file in first_dir.glob("*.png"):
        first_image = img_file
        break
    
    if first_image:
        cmd = [
            "python", "inference_complete.py",
            "--input", str(first_image),
            "--output", "runs/single_image_test",
            "--save_overlay", "--save_csv"
        ]
        run_command(cmd, f"단일 이미지 추론: {first_image.name}")
    
    # 예제 2: 폴더 배치 추론 (작은 샘플)
    print("\n" + "="*80)
    print("📂 예제 2: 폴더 배치 추론 (처음 5개 이미지)")
    print("="*80)
    
    # 테스트용 작은 폴더 생성
    test_folder = base_dir / "runs" / "test_batch_small"
    test_folder.mkdir(parents=True, exist_ok=True)
    
    # 처음 5개 이미지 복사
    import shutil
    copied_count = 0
    for img_file in first_dir.glob("*.png"):
        if copied_count >= 5:
            break
        dest_file = test_folder / img_file.name
        shutil.copy2(img_file, dest_file)
        copied_count += 1
        print(f"📋 복사: {img_file.name}")
    
    if copied_count > 0:
        cmd = [
            "python", "inference_complete.py", 
            "--input", str(test_folder),
            "--output", "runs/batch_test_results",
            "--save_overlay", "--save_csv",
            "--max_images", "5"  # 예제에서는 5개로 제한
        ]
        run_command(cmd, f"배치 추론: {copied_count}개 이미지")
    
    # 예제 3: 다른 디렉토리 추론 (가능한 경우)
    if len(available_dirs) > 1:
        print("\n" + "="*80)
        print("📂 예제 3: 다른 데이터셋 추론")
        print("="*80)
        
        second_dir_name, second_dir = available_dirs[1]
        
        # 테스트용 폴더 생성 (3개 이미지만)
        test_folder2 = base_dir / "runs" / "test_different_dataset"
        test_folder2.mkdir(parents=True, exist_ok=True)
        
        copied_count = 0
        for img_file in second_dir.glob("*.png"):
            if copied_count >= 3:
                break
            dest_file = test_folder2 / img_file.name
            shutil.copy2(img_file, dest_file)
            copied_count += 1
        
        if copied_count > 0:
            cmd = [
                "python", "inference_complete.py",
                "--input", str(test_folder2), 
                "--output", f"runs/{second_dir_name.lower().replace(' ', '_')}_results",
                "--save_overlay", "--save_csv",
                "--max_images", "3"  # 3개로 제한
            ]
            run_command(cmd, f"{second_dir_name} 데이터셋 추론: {copied_count}개 이미지")
    
    # 예제 4: CPU 추론 (GPU가 있더라도 CPU로 강제)
    print("\n" + "="*80)
    print("💻 예제 4: CPU 추론 테스트")
    print("="*80)
    
    if first_image:
        cmd = [
            "python", "inference_complete.py",
            "--input", str(first_image),
            "--output", "runs/cpu_inference_test", 
            "--device", "cpu",
            "--save_overlay"
        ]
        run_command(cmd, "CPU 디바이스로 추론")
    
    # 예제 5: 모든 사용 가능한 데이터셋 요약
    print("\n" + "="*80)
    print("📊 예제 5: 전체 데이터셋 요약 (각 1개씩)")
    print("="*80)
    
    summary_folder = base_dir / "runs" / "summary_test"
    summary_folder.mkdir(parents=True, exist_ok=True)
    
    for dir_name, dir_path in available_dirs:
        print(f"\n🔍 {dir_name} 처리 중...")
        
        # 각 디렉토리에서 첫 번째 이미지 하나씩
        first_img = None
        for img_file in dir_path.glob("*.png"):
            first_img = img_file
            break
        
        if first_img:
            # 고유한 이름으로 복사
            safe_name = dir_name.lower().replace(' ', '_').replace('/', '_')
            dest_file = summary_folder / f"{safe_name}_{first_img.name}"
            shutil.copy2(first_img, dest_file)
    
    # 요약 폴더 배치 추론
    summary_files = list(summary_folder.glob("*.png"))
    if summary_files:
        cmd = [
            "python", "inference_complete.py",
            "--input", str(summary_folder),
            "--output", "runs/summary_results", 
            "--save_overlay", "--save_csv",
            "--max_images", "10"  # 요약에서는 10개로 제한
        ]
        run_command(cmd, f"전체 데이터셋 요약: {len(summary_files)}개 이미지")
    
    # 완료 메시지
    print("\n" + "="*80)
    print("🎉 모든 추론 예제 실행 완료!")
    print("="*80)
    print("\n📁 결과 확인:")
    result_dirs = [
        "runs/single_image_test",
        "runs/batch_test_results", 
        "runs/cpu_inference_test",
        "runs/summary_results"
    ]
    
    for result_dir in result_dirs:
        if os.path.exists(result_dir):
            files = list(Path(result_dir).glob("*"))
            print(f"  📂 {result_dir}: {len(files)}개 파일")
        else:
            print(f"  ❌ {result_dir}: 생성되지 않음")
    
    print("\n💡 개별 실행 방법:")
    print("  python inference_complete.py --input [이미지/폴더] --output [결과폴더]")
    print("  python inference_complete.py --help  # 도움말")


if __name__ == "__main__":
    main()
