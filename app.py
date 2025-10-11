# common/modeling.py (النسخة الكاملة والمصححة)

import math
from datetime import datetime
from typing import Dict, List, Tuple, Any

# يتطلب تثبيت: pip install scipy
from scipy.optimize import minimize

# استيراد الوحدات المشتركة بالطريقة الصحيحة
from common import config
from common.utils import parse_date_safe, parse_score, poisson_pmf


def ewma_weight(delta_days: float, half_life_days: float) -> float:
    """
    يحسب الوزن باستخدام المتوسط المتحرك الأسي الموزون (EWMA).
    """
    if delta_days <= 0:
        return 1.0
    return 0.5 ** (delta_days / max(half_life_days, 1.0))

def calculate_league_averages(matches: List[Dict]) -> Dict[str, float]:
    """
    يحسب متوسط الأهداف المسجلة للفريق المضيف والضيف في الدوري.
    """
    hg_sum = ag_sum = n = 0
    for m in matches:
        hg, ag = parse_score(m)
        if hg is None: continue
        hg_sum += hg
        ag_sum += ag
        n += 1
    if n == 0:
        return {"avg_home_goals": 1.4, "avg_away_goals": 1.1, "home_adv": 0.3}
    avg_home = hg_sum / n
    avg_away = ag_sum / n
    home_adv = max(0.0, min(0.8, avg_home - avg_away))
    return {"avg_home_goals": avg_home, "avg_away_goals": avg_away, "home_adv": home_adv}

def build_team_factors(
    matches: List[Dict], league_avgs: Dict, cutoff: datetime
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    يبني عوامل قوة الهجوم والدفاع لكل فريق.
    """
    avg_home = league_avgs["avg_home_goals"]
    avg_away = league_avgs["avg_away_goals"]
    hl = float(config.HALF_LIFE_DAYS)
    prior_games = float(config.PRIOR_GAMES)
    scored_w, expected_w, conceded_w, expectedc_w = {}, {}, {}, {}

    for m in matches:
        dt = parse_date_safe(m.get("utcDate"))
        if not dt or dt > cutoff: continue
        hg, ag = parse_score(m)
        h_id, a_id = m.get("homeTeam", {}).get("id"), m.get("awayTeam", {}).get("id")
        if hg is None or not h_id or not a_id: continue

        w = ewma_weight((cutoff - dt).days, hl)
        scored_w[h_id] = scored_w.get(h_id, 0.0) + w * hg
        expected_w[h_id] = expected_w.get(h_id, 0.0) + w * avg_home
        conceded_w[h_id] = conceded_w.get(h_id, 0.0) + w * ag
        expectedc_w[h_id] = expectedc_w.get(h_id, 0.0) + w * avg_away
        scored_w[a_id] = scored_w.get(a_id, 0.0) + w * ag
        expected_w[a_id] = expected_w.get(a_id, 0.0) + w * avg_away
        conceded_w[a_id] = conceded_w.get(a_id, 0.0) + w * hg
        expectedc_w[a_id] = expectedc_w.get(a_id, 0.0) + w * avg_home

    A: Dict[str, float] = {}
    D: Dict[str, float] = {}
    mean_rate = (avg_home + avg_away) / 2.0
    all_team_ids = set(list(scored_w.keys()) + list(conceded_w.keys()))

    for tid in all_team_ids:
        s = scored_w.get(tid, 0.0) + prior_games * mean_rate
        e = expected_w.get(tid, 0.0) + prior_games * mean_rate
        a_factor = (s / e) if e > 0 else 1.0
        sc = conceded_w.get(tid, 0.0) + prior_games * mean_rate
        ec = expectedc_w.get(tid, 0.0) + prior_games * mean_rate
        d_factor = (sc / ec) if ec > 0 else 1.0
        A[str(tid)] = max(0.3, min(3.0, a_factor))
        D[str(tid)] = max(0.3, min(3.0, d_factor))
    return A, D

def build_elo_ratings(matches: List[Dict]) -> Dict[str, float]:
    """
    يحسب تصنيف Elo الديناميكي للفرق.
    """
    elo: Dict[int, float] = {}
    matches_sorted = sorted(
        [m for m in matches if parse_date_safe(m.get("utcDate"))],
        key=lambda m: parse_date_safe(m["utcDate"])
    )
    for m in matches_sorted:
        h, a = m.get("homeTeam", {}).get("id"), m.get("awayTeam", {}).get("id")
        hg, ag = parse_score(m)
        if not h or not a or hg is None: continue
        eh, ea = elo.get(h, 1500.0), elo.get(a, 1500.0)
        exp_home = 1.0 / (1.0 + 10 ** (-(eh + float(config.ELO_HFA) - ea) / 400.0))
        res_home = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
        delta = float(config.ELO_K) * (res_home - exp_home)
        elo[h], elo[a] = eh + delta, ea - delta
    return {str(tid): rating for tid, rating in elo.items()}

def fit_dc_rho_mle(matches: List[Dict], A: Dict, D: Dict, league_avgs: Dict) -> float:
    """
    يجد معامل الارتباط (rho) باستخدام خوارزمية تحسين من SciPy.
    """
    if len(matches) < 20 or not A or not D: return 0.0
    rho_max = float(config.DC_RHO_MAX)
    avg_home, avg_away = league_avgs["avg_home_goals"], league_avgs["avg_away_goals"]
    memo = {}
    for m in matches:
        h, a = m.get("homeTeam", {}).get("id"), m.get("awayTeam", {}).get("id")
        hg, ag = parse_score(m)
        if not (h and a and hg is not None): continue
        lh = max(0.1, avg_home * A.get(str(h), 1.0) * D.get(str(a), 1.0))
        la = max(0.1, avg_away * A.get(str(a), 1.0) * D.get(str(h), 1.0))
        memo[m['id']] = {'lh': lh, 'la': la, 'hg': hg, 'ag': ag}

    def log_likelihood(rho: float) -> float:
        ll = 0.0
        for data in memo.values():
            lh, la, hg, ag = data['lh'], data['la'], data['hg'], data['ag']
            tau = 1.0
            if hg == 0 and ag == 0:   tau = 1.0 - rho * lh * la
            elif hg == 0 and ag == 1: tau = 1.0 + rho * lh
            elif hg == 1 and ag == 0: tau = 1.0 + rho * la
            elif hg == 1 and ag == 1: tau = 1.0 - rho
            if tau <= 1e-6: return -1e9
            ll += (hg * math.log(lh) - lh) + (ag * math.log(la) - la) + math.log(tau)
        return ll

    result = minimize(lambda r: -log_likelihood(r[0]), x0=[0.0], bounds=[(-rho_max, rho_max)])
    return float(result.x[0]) if result.success else 0.0

def poisson_matrix_dc(lh: float, la: float, rho: float, max_goals: int = 8) -> List[List[float]]:
    pX = [poisson_pmf(i, lh) for i in range(max_goals + 1)]
    pY = [poisson_pmf(j, la) for j in range(max_goals + 1)]
    M = [[pX[i] * pY[j] for j in range(max_goals + 1)] for i in range(max_goals + 1)]
    if max_goals >= 1:
        M[0][0] *= max(1e-6, 1.0 - rho * lh * la)
        M[0][1] *= max(1e-6, 1.0 + rho * lh)
        M[1][0] *= max(1e-6, 1.0 + rho * la)
        M[1][1] *= max(1e-6, 1.0 - rho)
    s = sum(sum(row) for row in M)
    if s > 0:
        for i in range(max_goals + 1):
            for j in range(max_goals + 1): M[i][j] /= s
    return M

def matrix_to_outcomes(M: List[List[float]]) -> Tuple[float, float, float]:
    p_home = sum(M[i][j] for i in range(len(M)) for j in range(len(M[0])) if i > j)
    p_draw = sum(M[i][i] for i in range(min(len(M), len(M[0]))))
    p_away = 1.0 - p_home - p_draw
    return p_home, p_draw, p_away

# --- ✅ تم إضافة الدالة المفقودة هنا ---
def top_scorelines(matrix: List[List[float]], top_k: int = 5) -> List[Tuple[int, int, float]]:
    """
    يستخرج أعلى K نتائج متوقعة من مصفوفة الاحتمالات.
    """
    flat_list = []
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            flat_list.append((i, j, matrix[i][j]))
    flat_list.sort(key=lambda x: x[2], reverse=True)
    return flat_list[:top_k]
# ------------------------------------
