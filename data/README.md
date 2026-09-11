# 입력 데이터

원본에서 **실제로 복사한 데이터**를 포함한다. `python3 scripts/restore_artifacts.py`가 압축본을 복원하고 SHA-256을 확인한다. 다른 서버나 기존 연구 폴더의 심볼릭 링크를 요구하지 않는다. 저장소의 결과 CSV 일부도 같은 방법으로 복원한다. `provenance/bundles.json`에 각 압축본과 원래 파일의 해시·크기가 있다.

| 파일 | 용도 |
|---|---|
| `cache/raw/LAKE_V2_DART_AUTO_CALL.csv.gz` | 상품·발행/만기·쿠폰·상품형태 원문 |
| `cache/raw/LAKE_V2_DART_SCHD_INFO.csv.gz` | 조기상환·월 쿠폰 일정, 행사가·지급률·리자드·KI 원문 |
| `cache/raw/LAKE_V2_DART_UDLY_INFO.csv.gz` | 상품별 기초자산 원문 |
| `cache/udly_ticker_map.json` | 기초자산 ID와 시장 가격 티커의 당시 매핑 |
| `cache/px_*.parquet` | 당시 사용한 기초자산 가격 스냅샷. 종가에서 발행 전 역사 변동성·상관 계산 |
| `cache/krw_curve.parquet` | 당시 사용한 콜·3개월·10년 KRW 금리 스냅샷 |
| `els3_dataset.parquet` | 기존 58,790개 표본의 상품 ID만 추출한 스냅샷. 현재 계약/MC 라벨을 담지 않음 |
| `bundles/holidays-0.104.zip` | 실제 사용한 고정 한국 은행 달력 라이브러리와 포함 라이선스 |

새 서버에서 시장 데이터를 다시 내려받아 최신 값으로 바꾸면 동일 실험의 재현이 아니다. 이 저장소는 원래 사용한 스냅샷을 고정한다. 월지급형 템플릿과 공시 조건은 원문에서 다시 추출하고, **현재 학습용 MC 가격은 최종 pricer로 새로 계산한다.**

원본 전체 feature 표에 있던 과거 지급률 근사·과거 MC 라벨을 새 정답으로 사용하지 않는다. `scripts/rebuild_source_inputs.py`로 원문부터 표본 ID와 공시 6개 시장 상태가 일치하는지 검사한다. 합성군 시장은 실험 프로토콜의 분포와 고정 seed로 생성한다.

일반형 참고 계약 86개와 공시 6개 시장 상태는 각각 `analysis/mc_contract_v2_20260910/results/families.json`, `analysis/contract_value_20260910/results/mc_selected_contracts.json`에 있다. 과거 분석 폴더 이름을 유지한 이유는 검증된 계약 생성 코드를 수정하지 않기 위해서다. 해당 폴더에는 과거 MC 가격·모델·분석 코드를 넣지 않았다.

공시 6개의 발행사 문서는 `analysis/mc_monthly_v3_20260910/evidence/`에 원문 URL·HTML·텍스트와 함께 보관한다. 데이터와 문서의 원래 출처 표시는 유지했다. 별도의 데이터 이용허락을 새로 부여하는 저장소는 아니다.

`iv_daily_atm`은 현재 **역사 변동성/상관을 사용한 MC 설정**의 입력이 아니므로 포함하지 않았다. IV 기반 pricer 실험을 현재 결과와 혼동하지 않는다.
