# 환경 설정
- Python + Jupyter

# 실행 파일
[test/squat-realtime.ipynb](https://github.com/yunjikwak/capston-design-mpp/blob/15cdec28b45b844d5da48fc90c76c029464a9cf2/test/squat-realtime.ipynb)
- 처음 실행 시 Init 꼭! 실행하기

# 주요 설정
코드 안에서 변경
### 카메라 선택
```python
CAM_INDEX = 0   # 내장 카메라
# CAM_INDEX = 1 # 외장 USB 웹캠 (필요 시 값 변경)
```
### 출력 화면 크기 지정
`MAX_WIDTH = 1000 # 480, 720 등등 변경 가능`

### 목표 스쿼트 횟수
`TARGET_REPS = 3`

# 실행 방법
1. Init 한 번 실행하기
2. 이후 순차 실행
3. 마지막 행 실행 시 cv 창 열림
4. 재시작 원할 시
     - Run All (권장)
      - '실시간으로 MPP 동작' 하위 셀 전체 실행(Run Cells In Section)
      - 마지막 셀만 재실행(결과 중복될 수 있음 -> 추후 수정 예정)
   
# 동작 방법
cv창이 열린 이후

프로그램 시작 : s키 입력

프로그램 정지 : e키 입력

프로그램 정상 종료 : esc or q키 입력

### 프로그램 실행 중
서 있는 상태에서 시작(START) -> 천천히 내려가기(START -> SIT) -> 천천히 올라오기(SIT -> RISING) -> 올라오기 (RISING -> STAND)

- 과정이 끝나면 Count 값에 1 추가됨

# 출력 결과물
스쿼트 횟수

평균 점수

항목
- 무릎 각도
- 허리 일직선 유지
- 엉덩이 깊이
- 앉은 자세 유지

각 스쿼트 자세 횟수별 정확도 평균
