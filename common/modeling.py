# common/modeling.py
import math
from datetime import datetime
from typing import Dict, List, Tuple
from .config import config
from .utils import parse_date_safe, parse_score, poisson_pmf

def ewma_weight(delta_days: float, half_life_days: float) -> float:
    if delta_days <= 0:
        return 1.0
    return 0.5 ** (delta_days / max(half_life_days, 1.0))

def calculate_league_averages(matches: List[Dict]) -> Dict[str, float]:
    hg_sum = ag_sum = n = 0
    for m in matches:
        hg, ag = parse_score(m)
        if hg is None or ag is None:
            continue
        hg_sum += hg
        ag_sum += ag
        n += 1
    if n == 0:
        return {"avg_home_goals": 1.4, "avg_away_goals": 1.1, "home_adv": 0.3}
    avg_home = hg_sum / n
    avg_away = ag_sum / n
    # تقدير ميزة الأرضية بشكل مبسط
    home_adv = max(0.0, min(0.8, avg_home - avg_away))
    return {"avg_home_goals": avg_home, "avg_away_goals": avg_away, "home_adv": home_adv}

def build_team_factors(matches: List[Dict], league_avgs: Dict, cutoff: datetime) -> Tuple[Dict[str, float], Dict[str, float]]:
    avg_home = league_avgs["avg_home_goals"]
    avg_away = league_avgs["avg_away_goals"]
    hl = float(config.HALF_LIFE_DAYS)
    prior_games = float(config.PRIOR_GAMES)

    scored_w = {}     # team -> weighted goals scored
    expected_w = {}   # team -> weighted expected goals scored (league rate)
    conceded_w = {}   # team -> weighted goals conceded
    expectedc_w = {}  # team -> weighted expected conceded (league rate)

    for m in matches:
        dt = parse_date_safe(m.get("utcDate"))
        if not dt or dt > cutoff:
            continue
        hg, ag = parse_score(m)
        if hg is None or ag is None:
            continue
        h = m.get("homeTeam", {}).get("id")
        a = m.get("awayTeam", {}).get("id")
        if not h or not a:
            continue
        w = ewma_weight((cutoff - dt).days, hl)

        # سجل الفريق المُضيف
        scored_w[h] = scored_w.get(h, 0.0) + w * hg
        expected_w[h] = expected_w.get(h, 0.0) + w * avg_home
        conceded_w[h] = conceded_w.get(h, 0.0) + w * ag
        expectedc_w[h] = expectedc_w.get(h, 0.0) + w * avg_away

        # سجل الفريق الضيف
        scored_w[a] = scored_w.get(a, 0.0) + w * ag
        expected_w[a] = expected_w.get(a, 0.0) + w * avg_away
        conceded_w[a] = conceded_w.get(a, 0.0) + w * hg
        expectedc_w[a] = expectedc_w.get(a, 0.0) + w * avg_home

    # حساب عوامل الهجوم/الدفاع مع تمهيد (shrinkage)
    A: Dict[str, float] = {}
    D: Dict[str, float] = {}
    for tid in set(list(scored_w.keys()) + list(conceded_w.keys())):
        # smoothing: أضف prior_games مباريات عند المتوسط
        s = scored_w.get(tid, 0.0) + prior_games * (avg_home + avg_away) / 2.0
        e = expected_w.get(tid, 0.0) + prior_games * (avg_home + avg_away) / 2.0
        a_factor = (s / e) if e > 0 else 1.0

        sc = conceded_w.get(tid, 0.0) + prior_games * (avg_home + avg_away) / 2.0
        ec = expectedc_w.get(tid, 0.0) + prior_games * (avg_home + avg_away) / 2.0
        # ملاحظة: D>1 يعني دفاع أضعف (يتلقى أكثر من المتوقع)
        d_factor = (sc / ec) if ec > 0 else 1.0

        A[str(tid)] = max(0.3, min(3.0, a_factor))
        D[str(tid)] = max(0.3, min(3.0, d_factor))

    return A, D

def build_elo_ratings(matches: List[Dict]) -> Dict[str, float]:
    # إعداد بسيط لـELO بميزة أرضية ثابتة
    K = 20.0
    HFA = 60.0  # home field advantage pts
    elo: Dict[int, float] = {}

    # فرز المباريات زمنيًا
    matches_sorted = sorted(
        [m for m in matches if parse_date_safe(m.get("utcDate"))],
        key=lambda m: parse_date_safe(m["utcDate"])
    )

    for m in matches_sorted:
        h = m.get("homeTeam", {}).get("id")
        a = m.get("awayTeam", {}).get("id")
        hg, ag = parse_score(m)
        if not h or not a or hg is None or ag is None:
            continue

        eh = elo.get(h, 1500.0)
        ea = elo.get(a, 1500.0)

        # توقع النتيجة للبيت مع HFA
        exp_home = 1.0 / (1.0 + 10 ** (-(eh + HFA - ea) / 400.0))
        res_home = 1.0 if hg > ag else 0.5 if hg == ag else 0.0

        delta = K * (res_home - exp_home)
        elo[h] = eh + delta
        elo[a] = ea - delta

    # تحويل المفاتيح إلى str للحفظ
    return {str(tid): rating for tid, rating in elo.items()}

def fit_dc_rho_mle(matches: List[Dict], A: Dict, D: Dict, league_avgs: Dict) -> float:
    if len(matches) < 20 or not A or not D:
        return 0.0

    rho_max = float(config.DC_RHO_MAX)
    avg_home, avg_away = league_avgs["avg_home_goals"], league_avgs["avg_away_goals"]

    memo = {}
    for m in matches:
        h, a = m.get("homeTeam", {}).get("id"), m.get("awayTeam", {}).get("id")
        hg, ag = parse_score(m)
        if not (h and a and hg is not None and ag is not None):
            continue
        lh = max(0.1, avg_home * A.get(str(h), 1.0) * D.get(str(a), 1.0))
        la = max(0.1, avg_away * A.get(str(a), 1.0) * D.get(str(h), 1.0))
        memo[m['id']] = {'lh': lh, 'la': la, 'hg': hg, 'ag': ag}

    def loglik(rho: float) -> float:
        ll = 0.0
        for data in memo.values():
            lh, la, hg, ag = data['lh'], data['la'], data['hg'], data['ag']
            # معاملات Dixon-Coles لتعديل احتمالات (0,0), (0,1), (1,0), (1,1)
            tau = 1.0
            if hg == 0 and ag == 0:
                tau = 1.0 - rho * lh * la
            elif hg == 0 and ag == 1:
                tau = 1.0 + rho * lh
            elif hg == 1 and ag == 0:
                tau = 1.0 + rho * la
            elif hg == 1 and ag == 1:
                tau = 1.0 - rho
            if tau <= 1e-6:
                return -float('inf')
            ll += (hg * math.log(lh) - lh) + (ag * math.log(la) - la) + math.log(tau)
        return ll

    best_rho, best_ll = 0.0, -float('inf')
    # مسح شبكي بسيط 0.01 خطوة
    for r_int in range(int(-rho_max * 100), int(rho_max * 100) + 1):
        r = r_int / 100.0
        ll = loglik(r)
        if ll > best_ll:
            best_ll, best_rho = ll, r

    return best_rho

def poisson_matrix_dc(lh: float, la: float, rho: float, max_goals: int = 8) -> List[List[float]]:
    pX = [poisson_pmf(i, lh) for i in range(max_goals + 1)]
    pY = [poisson_pmf(j, la) for j in range(max_goals + 1)]

    M = [[pX[i] * pY[j] for j in range(max_goals + 1)] for i in range(max_goals + 1)]

    if max_goals >= 1:
        M[0][0] *= max(1e-6, 1.0 - rho * lh * la)
        M[0][1] *= max(1e-6, 1.0 + rho * lh)
        M[1][0] *= max(1e-6, 1.0 + rho * la)
        M[1][1] *= max(1e-6, 1.0 - rho)

    # إعادة التطبيع
    s = sum(sum(row) for row in M)
    if s > 0:
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                M[i][j] /= s
    return M

def matrix_to_outcomes(M: List[List[float]]) -> Tuple[float, float, float]:
    # p_home, p_draw, p_away
    p_home = sum(M[i][j] for i in range(len(M)) for j in range(len(M[0])) if i > j)
    p_draw = sum(M[i][i] for i in range(min(len(M), len(M[0]))))
    p_away = 1.0 - p_home - p_draw
    return p_home, p_draw, p_away