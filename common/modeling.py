# common/modeling.py
# -----------------------------------------------------------------------------
# Team Factors مع وزن زمني + Gamma-smoothing + انكماش هرمي للموسم السابق،
# تقدير rho ببحث شبكي مرجّح زمنياً، ومصفوفة Poisson-DC بقطع ديناميكي.
# -----------------------------------------------------------------------------
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import math
import numpy as np

from common.utils import parse_date_safe, parse_score

def _safe_goals(match: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    hg, ag = parse_score(match)
    if hg is None or ag is None:
        return None, None
    return int(hg), int(ag)

def calculate_league_averages(matches: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_h, total_a, n, draws = 0.0, 0.0, 0, 0
    for m in matches:
        hg, ag = _safe_goals(m)
        if hg is None:
            continue
        total_h += hg
        total_a += ag
        n += 1
        if hg == ag:
            draws += 1
    if n == 0:
        return {"avg_home_goals": 1.40, "avg_away_goals": 1.10, "draw_rate": 0.26, "matches_count": 0}
    return {
        "avg_home_goals": total_h / n,
        "avg_away_goals": total_a / n,
        "draw_rate": draws / n,
        "matches_count": n
    }

def _time_decay_weight(match_date: Optional[datetime], season_end: Optional[datetime], half_life_days: int = 180) -> float:
    if not (match_date and season_end) or half_life_days <= 0:
        return 1.0
    days = max(0, (season_end - match_date).days)
    return math.exp(-math.log(2.0) * days / float(half_life_days))

def _collect_observations(matches: List[Dict[str, Any]]) -> List[Tuple[int, int, int, int, Optional[datetime]]]:
    obs = []
    for m in matches:
        dt = parse_date_safe(m.get("utcDate"))
        h_id = m.get("homeTeam", {}).get("id")
        a_id = m.get("awayTeam", {}).get("id")
        hg, ag = _safe_goals(m)
        if hg is None or not h_id or not a_id:
            continue
        obs.append((int(h_id), int(a_id), int(hg), int(ag), dt))
    return obs

def build_team_factors(
    matches: List[Dict[str, Any]],
    league_avgs: Dict[str, Any],
    season_end_date: datetime,
    decay_halflife_days: int = 180,
    prior_strength: float = 3.0,
    max_iter: int = 60,
    damping: float = 0.5,
    eps: float = 1e-9,
    # انكماش هرمي نحو الموسم السابق (اختياري)
    prior_attack: Optional[Dict[str, float]] = None,
    prior_defense: Optional[Dict[str, float]] = None,
    team_prior_weight: float = 0.0,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    عوامل الهجوم/الدفاع بمخطط تكراري مستقر:
        λ_home = avg_home * A_home * D_away
        λ_away = avg_away * A_away * D_home
    - أوزان زمنية داخل الموسم
    - Gamma-smoothing نحو 1.0
    - انكماش هرمي نحو عوامل الموسم السابق (إن توفرت) بوزن team_prior_weight
    """
    avg_home = float(league_avgs.get("avg_home_goals", 1.40))
    avg_away = float(league_avgs.get("avg_away_goals", 1.10))

    obs = _collect_observations(matches)
    if not obs:
        return {}, {}

    team_ids = sorted({h for h, _, _, _, _ in obs} | {a for _, a, _, _, _ in obs})
    idx = {tid: i for i, tid in enumerate(team_ids)}
    T = len(team_ids)

    A = np.ones(T, dtype=np.float64)
    D = np.ones(T, dtype=np.float64)

    # priors (هرمي)
    A0 = np.ones(T, dtype=np.float64)
    D0 = np.ones(T, dtype=np.float64)
    if prior_attack:
        for tid, val in prior_attack.items():
            if tid.isdigit() and int(tid) in idx:
                A0[idx[int(tid)]] = max(1e-6, float(val))
    if prior_defense:
        for tid, val in prior_defense.items():
            if tid.isdigit() and int(tid) in idx:
                D0[idx[int(tid)]] = max(1e-6, float(val))

    for _iter in range(max_iter):
        GF = np.zeros(T, dtype=np.float64)
        GA = np.zeros(T, dtype=np.float64)

        E_home_A = np.zeros(T, dtype=np.float64)
        E_away_A = np.zeros(T, dtype=np.float64)
        E_def_home = np.zeros(T, dtype=np.float64)
        E_def_away = np.zeros(T, dtype=np.float64)

        for h, a, hg, ag, dt in obs:
            w = _time_decay_weight(dt, season_end_date, decay_halflife_days)
            ih, ia = idx[h], idx[a]
            GF[ih] += w * hg
            GF[ia] += w * ag
            GA[ih] += w * ag
            GA[ia] += w * hg

            E_home_A[ih] += w * D[ia]
            E_away_A[ia] += w * D[ih]
            E_def_home[ih] += w * A[ia]
            E_def_away[ia] += w * A[ih]

        denom_A = avg_home * E_home_A + avg_away * E_away_A
        denom_D = avg_away * E_def_home + avg_home * E_def_away

        A_target = (GF + prior_strength) / (denom_A + prior_strength + eps)
        D_target = (GA + prior_strength) / (denom_D + prior_strength + eps)

        # انكماش هرمي نحو الموسم السابق (جيومتري) بحسب التعرضات
        if team_prior_weight > 0.0:
            wA = denom_A + eps
            wD = denom_D + eps
            alphaA = team_prior_weight / (team_prior_weight + wA)
            alphaD = team_prior_weight / (team_prior_weight + wD)
            A_target = (A_target ** (1.0 - alphaA)) * (A0 ** alphaA)
            D_target = (D_target ** (1.0 - alphaD)) * (D0 ** alphaD)

        A_new = (A ** (1.0 - damping)) * (A_target ** damping)
        D_new = (D ** (1.0 - damping)) * (D_target ** damping)

        delta = max(
            float(np.max(np.abs(np.log(A_new + eps) - np.log(A + eps)))),
            float(np.max(np.abs(np.log(D_new + eps) - np.log(D + eps))))
        )
        A, D = A_new, D_new
        if delta < 1e-4:
            break

    attack_factors = {str(tid): float(A[idx[tid]]) for tid in team_ids}
    defense_factors = {str(tid): float(D[idx[tid]]) for tid in team_ids}
    return attack_factors, defense_factors

def build_elo_ratings(
    matches: List[Dict[str, Any]],
    start_rating: float = 1500.0,
    k_base: float = 24.0,
    hfa_elo: float = 60.0,
    scale: float = 400.0,
    decay_halflife_days: int = 365
) -> Dict[str, float]:
    ratings: Dict[int, float] = {}
    def get_r(t: int) -> float:
        return ratings.get(t, start_rating)

    for m in matches:
        hg, ag = _safe_goals(m)
        if hg is None:
            continue
        h = m.get("homeTeam", {}).get("id")
        a = m.get("awayTeam", {}).get("id")
        if not h or not a:
            continue
        h, a = int(h), int(a)
        dt = parse_date_safe(m.get("utcDate"))
        season_end = parse_date_safe(m.get('season', {}).get('endDate')) or dt
        w_time = _time_decay_weight(dt, season_end, decay_halflife_days)

        Rh = get_r(h)
        Ra = get_r(a)
        exp_home = 1.0 / (1.0 + 10.0 ** (-(Rh + hfa_elo - Ra) / scale))

        if hg > ag:
            score_home = 1.0
        elif hg == ag:
            score_home = 0.5
        else:
            score_home = 0.0

        gd = abs(hg - ag)
        g_factor = math.log(1 + gd) if gd > 0 else 1.0

        K = k_base * w_time * g_factor
        Rh_new = Rh + K * (score_home - exp_home)
        Ra_new = Ra + K * ((1.0 - score_home) - (1.0 - exp_home))
        ratings[h] = Rh_new
        ratings[a] = Ra_new

    return {str(k): float(v) for k, v in ratings.items()}

def _dc_correction(lambda_h: float, lambda_a: float, rho: float, i: int, j: int) -> float:
    # تصحيح Dixon–Coles للخلايا الصغيرة
    if i == 0 and j == 0:
        return max(1e-12, 1.0 - rho * lambda_h * lambda_a)
    if i == 1 and j == 0:
        return max(1e-12, 1.0 + rho * lambda_a)
    if i == 0 and j == 1:
        return max(1e-12, 1.0 + rho * lambda_h)
    if i == 1 and j == 1:
        return max(1e-12, 1.0 - rho)
    return 1.0

def suggest_goal_cutoff(lambda_h: float, lambda_a: float, min_cap: int = 8, tail_eps: float = 1e-7, hard_cap: int = 16) -> int:
    # قصّ ديناميكي بناءً على ذيل كل بواسون
    def cutoff_for(lam: float) -> int:
        p = math.exp(-lam)
        csum = p
        k = 0
        while (1.0 - csum) > tail_eps and k < hard_cap:
            k += 1
            p = p * lam / k
            csum += p
        return k
    g_h = cutoff_for(max(1e-6, lambda_h))
    g_a = cutoff_for(max(1e-6, lambda_a))
    return max(min_cap, g_h, g_a)

def poisson_matrix_dc(lambda_h: float, lambda_a: float, rho: float, max_goals: Optional[int] = None, tail_eps: float = 1e-7) -> np.ndarray:
    if max_goals is None:
        max_goals = suggest_goal_cutoff(lambda_h, lambda_a, tail_eps=tail_eps)
    max_goals = int(max(6, min(20, max_goals)))

    def poisson_pmf_vector(lam: float, K: int) -> np.ndarray:
        probs = np.zeros(K + 1, dtype=np.float64)
        probs[0] = math.exp(-lam)
        for k in range(1, K + 1):
            probs[k] = probs[k - 1] * lam / k
        s = probs.sum()
        if s > 0:
            probs /= s
        return probs

    p_h = poisson_pmf_vector(lambda_h, max_goals)
    p_a = poisson_pmf_vector(lambda_a, max_goals)
    base = np.outer(p_h, p_a)

    for i in (0, 1):
        for j in (0, 1):
            base[i, j] *= _dc_correction(lambda_h, lambda_a, rho, i, j)

    total = base.sum()
    if total > 0:
        base /= total
    return base

def matrix_to_outcomes(mat: np.ndarray) -> Tuple[float, float, float]:
    p_home = float(np.triu(mat, k=1).sum())
    p_draw = float(np.trace(mat))
    p_away = float(np.tril(mat, k=-1).sum())
    s = p_home + p_draw + p_away
    if s > 0:
        p_home, p_draw, p_away = p_home / s, p_draw / s, p_away / s
    return p_home, p_draw, p_away

def top_scorelines(mat: np.ndarray, top_k: int = 5) -> List[Tuple[int, int, float]]:
    rows = []
    K = mat.shape[0]
    for i in range(K):
        for j in range(K):
            rows.append((i, j, float(mat[i, j])))
    rows.sort(key=lambda x: x[2], reverse=True)
    return rows[:max(0, top_k)]

def fit_dc_rho_mle(
    matches: List[Dict[str, Any]],
    factors_A: Dict[str, float],
    factors_D: Dict[str, float],
    league_avgs: Dict[str, Any],
    decay_halflife_days: int = 180,
    rho_min: float = -0.2,
    rho_max: float = 0.2,
    rho_step: float = 0.001,
) -> float:
    """
    تقدير ρ عبر بحث شبكي مرجّح زمنياً لتعظيم الدالة الاحتمالية.
    """
    avg_home = float(league_avgs.get("avg_home_goals", 1.40))
    avg_away = float(league_avgs.get("avg_away_goals", 1.10))

    obs = []
    end_date = None
    for m in matches:
        dt = parse_date_safe(m.get("utcDate"))
        end_date = max(end_date, dt) if (end_date and dt) else (dt or end_date)
        hg, ag = _safe_goals(m)
        if hg is None:
            continue
        h = m.get("homeTeam", {}).get("id")
        a = m.get("awayTeam", {}).get("id")
        if not h or not a:
            continue
        h, a = str(int(h)), str(int(a))
        Ah = float(factors_A.get(h, 1.0))
        Aa = float(factors_A.get(a, 1.0))
        # توقعات الأهداف
        lam_h = avg_home * Ah * float(factors_D.get(a, 1.0))
        lam_a = avg_away * Aa * float(factors_D.get(h, 1.0))
        obs.append((lam_h, lam_a, int(hg), int(ag), dt))

    if not obs:
        return 0.0
    if end_date is None:
        end_date = datetime.now()

    def nll(rho: float) -> float:
        total = 0.0
        for lam_h, lam_a, hg, ag, dt in obs:
            w = _time_decay_weight(dt, end_date, decay_halflife_days)
            if lam_h <= 0 or lam_a <= 0:
                continue
            log_p_h = -lam_h + (hg * math.log(lam_h)) - math.lgamma(hg + 1.0)
            log_p_a = -lam_a + (ag * math.log(lam_a)) - math.lgamma(ag + 1.0)
            c = _dc_correction(lam_h, lam_a, rho, hg, ag)
            log_c = math.log(max(1e-12, c))
            total += -w * (log_p_h + log_p_a + log_c)
        return total

    best_rho, best_val = 0.0, float("inf")
    grid = np.arange(rho_min, rho_max + 1e-12, rho_step, dtype=np.float64)
    for r in grid:
        val = nll(float(r))
        if val < best_val:
            best_val, best_rho = val, float(r)
    return float(best_rho)
