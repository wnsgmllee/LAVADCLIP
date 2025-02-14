1. UCF-Crime 다운로드 후 unzip
2. dataset에 보면 README 있는데, 거기서 1.Anomaly Detection Experiment에 대한 데이터를 제외한 나머지 파일들은 삭제
3. 삭제하고 남은 각 파일에 대해 unzip
4. README에 있는대로 여러 개 나눠진 디렉토리들 하나로 합치기 (합치면 Anomaly Video, Training Normal, Testing Normal Video 이렇게 세 폴더로 나뉘게 될거임)
5. vid2img.py 가서 가장 아래에 file path에다 다음과 같이 설정하면 됨 (ucf_crime_path: 위의 세 폴더를 포함한 상위폴더 (ucf), output_path: (ucf)와 같은 위치에 (ucf_frame)하나 만들고 이 경로로 설정)
6. run_cpu.sh에서 빈 칸에 본인 username 입력
7. 새 conda env 만들고 requirements 설치
8. sbatch로 run_cpu job 올리기
