# 데이터 준비 안내

이 폴더는 데이터 파일 목록과 준비 안내만 보관합니다. 원음(MP3), 라벨 원본(JSON), ZIP 압축파일은 업로드하지 않았습니다.

## 공식 자료

- AI Hub: 비언어적 음향 데이터
- https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=644
- 이용정책: https://www.aihub.or.kr/intrcn/guid/usagepolicy.do?currMenu=151&topMenu=105
- 한국지능정보사회진흥원의 인공지능 학습용 데이터 구축 사업 결과물을 활용합니다.

승인된 계정으로 공식 자료를 직접 내려받아 압축을 해제한 뒤, `datasets/extracted/1.Training/`와 `datasets/extracted/2.Validation/` 아래에 원천데이터와 라벨링데이터의 기존 구조를 유지합니다.

프로그램은 공작기계음(M_1_01), 금속 충돌음(O_3_03), 비명(H_3_03)만 사용합니다. Training을 원본 녹음 ID 기준으로 학습/내부 검증으로 나누고, Validation은 최종 테스트에 사용합니다.

- `DATA_FILES.csv`: 이번 저장 시점에 존재한 로컬 자료의 상대경로·바이트 크기 목록입니다. ZIP 내부 항목 전체가 아닌 실제 파일 목록입니다.
- `DATA_SUMMARY.json`: 자료 수, 사용 대상, 코드 체크섬과 원본을 업로드하지 않았다는 기록입니다.

뒤에 추가했다가 취소한 외부 비명 52개는 이 프로젝트의 활성 학습 자료에 포함되지 않습니다.
