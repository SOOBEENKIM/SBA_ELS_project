# -*- coding: utf-8 -*-
"""PI-DeepONet 물리항 — 3자산 worst-of 경로의존 ELS 의 미분가능 상태제약.

 stage-1 앵커(MC 이론가 근사)에만 쓰인다. stage-2 잔차(시장마진)는 물리 대상이 아니다.

 페이오프 규약은 실측 MC(`module.mc_engine.payoff_from_path_summary_t`) 와 동일해야 한다:
   - 조기상환 판정은 **정규 상환이 리자드보다 우선**
   - 만기 미상환 시 낙인(B<1) 이면 KI 여부에 따라 worst / 원금, 노낙인(B>=1) 이면 항상 worst
 (일치 여부는 tests/test_pi_physics.py 가 MC 페이오프 블록과 직접 대조한다.)

 상태벡터(state, 길이 10) 규약 — PIStateDeepONet 과 공유:
   [0:3] S1..S3 (스팟/기준가)          [3:6] 러닝최소 M1..M3
   [6]   tau (잔존만기, 연)            [7]   KI 지시자 (0/1)
   [8]   이벤트 직전 여부 (1=pre, 0=post)
   [9]   관측 진행률 (0=발행, 1=만기)
"""
import torch


# ===== 경계조건 (만기·조기상환 이벤트) =====
def terminal_payoff(spot, strike, pmt, ki_state):
    """만기 페이오프. worst >= 만기행사가면 (1+pmt), 실패 시 KI 면 worst, 아니면 원금 1.
     노낙인(B>=1) 은 collocation 단계에서 ki_state=1 로 들어와 항상 worst 가 된다(MC 규약과 동일)."""
    worst = spot.amin(dim=1)
    hit = worst >= strike
    survival = torch.where(ki_state > .5, worst.clamp(max=1.0), torch.ones_like(worst))
    return torch.where(hit, 1.0 + pmt, survival)


def event_payoff(spot, running_min, strike, pmt, lz_barrier, lz_pmt):
    """조기상환 관측일 페이오프. 반환 (value, active, regular, lizard).
     regular = worst >= 행사가(정규 상환, 우선). lizard = 정규 실패 + 리자드 조항 유효 + 러닝최소 >= 리자드배리어.
     active=False(둘 다 실패) 인 경로는 상환되지 않으므로 continuation_relation 이 담당한다."""
    regular = spot.amin(dim=1) >= strike
    has_lz = torch.isfinite(lz_barrier)
    lizard = (~regular) & has_lz & (running_min.amin(dim=1) >= lz_barrier)
    active = regular | lizard
    value = torch.where(regular, 1.0 + pmt, 1.0 + lz_pmt)
    return value, active, regular, lizard


def event_tau(tenor, nobs, event_index):
    """관측 j(0-based) 직후의 잔존만기. 균등 관측 가정: elapsed = tenor*(j+1)/nobs."""
    elapsed = tenor * (event_index.to(tenor.dtype) + 1) / nobs.to(tenor.dtype)
    return (tenor - elapsed).clamp_min(0.)


def ki_transition_mask(running_min, barrier):
    """러닝최소가 배리어를 깬 경로(=KI 발생). 상태 전이 판정용."""
    return running_min.amin(dim=1) < barrier


def feasible_ki_states(running_min, barrier, proposed=None):
    """물리적으로 가능한 KI 지시자만 남긴다. 배리어를 이미 깬 경로는 KI=0 이 될 수 없다.
     (collocation 이 '배리어 아래인데 KI=0' 같은 모순 상태를 만들지 않게 하는 교정 장치)"""
    breached = running_min.amin(1) < barrier
    if proposed is None:
        proposed = torch.zeros_like(barrier)
    return torch.where(breached, torch.ones_like(proposed), proposed)


def continuation_relation(value_before, value_after, non_redemption_mask):
    """상환 조건을 만족하지 못한 관측일에서는 가치가 연속이어야 한다(V_before == V_after).
     상환된 경로는 값이 점프하므로 마스크로 제외한다."""
    if not non_redemption_mask.any():
        return value_before.sum() * 0
    return ((value_before[non_redemption_mask] - value_after[non_redemption_mask]) ** 2).mean()


# ===== PDE 잔차 =====
def pde_residual(value_fn, spot, tau, sigma, corr, rate):
    """3자산 백워드 Black-Scholes 잔차: -V_tau + Σ r·Si·V_Si + ½ΣΣ ρij·σi·σj·Si·Sj·V_SiSj - r·V.

     value_fn(spot, tau) -> (n,) 가치. 2차 도함수를 위해 create_graph=True 로 두 번 미분하므로
     반환 잔차는 그대로 손실에 넣어 역전파할 수 있다. (3×3 = 9 회 autograd → 스텝 비용이 큰 항)"""
    spot = spot.requires_grad_(True)
    tau = tau.requires_grad_(True)
    value = value_fn(spot, tau).reshape(-1)
    ones = torch.ones_like(value)
    dspot = torch.autograd.grad(value, spot, ones, create_graph=True)[0]
    dtau = torch.autograd.grad(value, tau, ones, create_graph=True)[0].reshape(-1)
    drift = ((rate[:, None] * spot) * dspot).sum(1)
    diffusion = torch.zeros_like(value)
    for i in range(3):
        for j in range(3):
            gij = torch.autograd.grad(dspot[:, i], spot, torch.ones_like(value),
                                      create_graph=True, retain_graph=True)[0][:, j]
            diffusion = diffusion + .5 * corr[:, i, j] * sigma[:, i] * sigma[:, j] \
                * spot[:, i] * spot[:, j] * gij
    return -dtau + drift + diffusion - rate * value


def finite_or_raise(name, tensor):
    """물리손실은 발산하기 쉬우므로 NaN/Inf 를 조용히 흘리지 않고 즉시 실패시킨다."""
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"{name} 에 NaN/Inf 가 있습니다")
    return tensor
