"""Build a side-by-side HTML index using the actual images embedded in the latest PPT."""
from common import *
import html

def main():
 man=json.loads((HERE/'ppt_reference/ppt_manifest.json').read_text());slides={s['slide']:s for s in man['slides']}
 pairs=[(7,0,'sample200_by_segment.png','200개 복원 표본: 공정가 6분위별 차이·MAPE'),(8,0,'fair_mc_histogram.png','전체 공정가−MC 분포'),(9,0,'curve_comparison.png','금리곡선: 원본 예시 날짜 일치, 200개 ID 전체 동일성은 미확정'),(10,0,'sample200_curve_history.png','200개 복원 표본의 3Y·CD91 시계열'),(11,0,'sample200_by_year.png','200개 복원 표본: 발행연도별 차이'),(12,0,'stage1_r2.png','MC 타깃 Stage 1 R²'),(13,0,'stage1_error_histogram.png','Stage 1 예측 오차 분포'),(13,1,'stage1_mape.png','Stage 1 MAPE')]
 # Use embedded image dimensions/labels to identify slide13 ordering from its XML, not assumed file names.
 body=['<!doctype html><html lang="ko"><meta charset="utf-8"><title>PPT와 전체 재실행 결과 비교</title><style>body{font-family:Arial,sans-serif;margin:30px;color:#17212b;background:#f3f5f8}section{background:white;padding:20px;margin:25px 0;border-radius:8px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px}img{width:100%;background:white}h1{font-size:26px}h2{font-size:20px}p{line-height:1.6}</style><h1>최신 PPT ↔ 전체 MC 재계산 결과</h1><p>왼쪽은 첨부 PPT에서 추출한 원본 그림, 오른쪽은 새 MC 가격으로 생성한 그림임. 그림 디자인이나 난수 결과의 픽셀 일치가 기준은 아님. 세부 조건·표본·수치는 REPORT.md와 CSV에 기록함.</p>']
 for slide,i,new,title in pairs:
  figs=slides[slide]['figures']
  if i>=len(figs):continue
  old=figs[i]['file'];body.append(f'<section><h2>{slide}p. {html.escape(title)}</h2><div class="pair"><div><p>PPT 원본</p><img src="ppt_reference/{old}"></div><div><p>새 MC 기반 재실행</p><img src="figures/{new}"></div></div></section>')
 body.append('</html>');(HERE/'PPT_COMPARISON.html').write_text('\n'.join(body));print('PPT comparison created')
if __name__=='__main__':main()
