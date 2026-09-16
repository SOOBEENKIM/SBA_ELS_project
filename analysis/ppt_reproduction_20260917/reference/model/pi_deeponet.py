# -*- coding: utf-8 -*-
"""PI-DeepONet 하이브리드 — 물리정보 stage-1 앵커 + 기존 stage-2 잔차.

 stage-1 앵커만 물리정보(PDE 잔차 + ELS 이벤트 경계)로 바뀌고, stage-2(시장마진 잔차)는
 다른 모델과 **동일한** model.stage2.ml_resid 를 쓴다 → PI 효과가 stage-1 에 국소화돼 분리 측정된다.
   Fair_hat = MC_hat(PI 앵커) + recent_margin + Residual_hat(stage-2)

 두 개를 함께 학습한다 (순서 고정):
   pi_deeponet_hybrid_data_only  물리항 0 — PI 구조의 대조군
   pi_deeponet_hybrid_pde        + PDE 잔차. 같은 폴드의 data_only 체크포인트에서 커리큘럼 초기화

 PDE 를 scratch 에서 학습하면 물리 스케일이 데이터항을 압도해 발산한다. data_only 로 데이터 적합을
 먼저 잡고 그 위에 물리를 얹는 순서가 릴리스에서 검증된 레시피이므로 이 순서를 강제한다."""
from pathlib import Path

from module.pi_train import PREDECESSOR, train_pi_curve
from module.pipeline import predict_hybrid

NAME = "pi_deeponet_hybrid"


def _anchor(variant):
    """variant 별 stage-1 앵커 콜백 생성 (pipeline 규약: anchor_fn(D,cfg,tr,va,te,save_path))."""
    def anchor(D, cfg, tr, va, te, save_path=None):
        init_path = None
        prev = PREDECESSOR.get(variant)
        if prev is not None and save_path is not None:
            # 같은 폴드의 선행 variant 체크포인트 (파일명에서 variant 만 치환)
            cand = Path(str(save_path).replace(f"{NAME}_{variant}_", f"{NAME}_{prev}_") + ".pt")
            if not cand.exists():
                raise FileNotFoundError(
                    f"커리큘럼 선행 체크포인트가 없습니다: {cand}\n"
                    f"  → {NAME}_{prev} 를 먼저 학습해야 합니다 (run() 이 그 순서를 보장합니다)")
            init_path = str(cand)
        predict, _ = train_pi_curve(D, cfg, tr, va, variant=variant,
                                    save_path=save_path, init_path=init_path)
        return predict
    return anchor


def _run_retired(D, cfg):
    """**은퇴(registry 미발견)**: 이름이 'run' 이 아니므로 model.registry 가 이 모듈을 건너뛴다.

     물리를 전 구조에 일괄 적용한 초기 버전이다. 5시드 ablation 결과 data_only 와 deeponet_hybrid 의
     차이가 노이즈 안(paired t=0.42, p=0.70, 5시드 중 2승)이었고, pde 는 STEP 에서 stage-1 을 떨어뜨렸다.
     구조별로 라우팅하는 model/pi_route.py 가 이를 대체한다. 재현이 필요하면 이 함수를 run 으로 되돌린다."""
    T = dict(target=D.FAIR, use_margin=True)   # 공정가 2단계 (다른 하이브리드와 동일)
    out = {}
    # 순서 중요: data_only 4폴드가 모두 저장된 뒤에야 pde 가 커리큘럼 초기화를 찾을 수 있다.
    for variant in ("data_only", "pde"):
        name = f"{NAME}_{variant}"
        out[name] = predict_hybrid(D, cfg, _anchor(variant), name=name, **T)
    return out
