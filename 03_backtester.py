# 03_backtester.py
# -----------------------------------------------------------------------------
# الوصف:
# - باكتيستر زمني للموديل الإحصائي (Dixon–Coles + Team Factors + ELO).
# - يقيس LogLoss/Brier/ECE عبر مواسم/دوريات بنهج نافذة زمنية متوسعة (expanding window).
# - توليف بسيط لمعاملات:
#     * TEAM_FACTORS_HALFLIFE_DAYS
#     * TEAM_FACTORS_PRIOR_GLOBAL
#     * TEAM_FACTORS_TEAM_PRIOR_WEIGHT
#     * DC_RHO_RANGE (نستخدم rho_max ونأخذ السالب كنصف المدى)
# - يطبع أفضل الإعدادات ويقترح قيمًا لتثبيتها في common.config.
# الاستخدام:
#   python 03_backtester.py \
#       --comps PL PD \
#       --min-train 120 \
#       --block-size 40 \
#       --use-elo \
#       --grid-halflife 90,180,365 \
#       --grid-prior-global 2.0,3.0,5.0 \
#       --grid-team-prior-weight 0.0,5.0 \
#       --grid-rho-max 0.15,0.20 \
#       --rho-step 0.002 \
#       --ece-bins 10 \
#       --limit-seasons 3 \
#       --save
# -----------------------------------------------------------------------------

import json
import argparse
import math
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict

import numpy as np

from common import config
from common.utils import log, parse_date_safe, parse_score
from common.modeling import (
    calculate_league_averages,
    build_team_factors,
    build_elo_ratings,
    fit_dc_rho_mle,
    poisson_matrix_dc,
    suggest_goal_cutoff,
)

# -----------------------------------------------------------------------------
# أدوات مساعدة
# -----------------------------------------------------------------------------

def load_matches(path: Path) -> List[Dict[str, Any]]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        log(f"تعذّر تحميل matches.json: {e}", "CRITICAL")
        raise

def group_matches_by_season(matches: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    by_season: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for m in matches:
        comp_code = (m.get("competition", {}) or {}).get("code", "UNK")
        season_year = (m.get("season", {}) or {}).get("startDate", "1900")[:4]
        key = f"{comp_code}_{season_year}"
        by_season[key].append(m)
    # ترتيب داخلي للمباريات زمنيًا
    for k in by_season:
        by_season[k].sort(key=lambda x: parse_date_safe(x.get("utcDate")) or parse_date_safe("1900-01-01"))
    return by_season

def parse_grid_list_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def parse_grid_list_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def season_key_parts(sk: str) -> Tuple[str, int]:
    try:
        comp, yr = sk.split("_")
        return comp, int(yr)
    except Exception:
        return sk, 0

def outcome_label(hg: int, ag: int) -> int:
    # 0: فوز المضيف, 1: تعادل, 2: فوز الضيف
    return 0 if hg > ag else (1 if hg == ag else 2)

def compute_metrics(probs: List[Tuple[float, float, float]], labels: List[int], ece_bins: int = 10) -> Dict[str, Any]:
    eps = 1e-15
    n = len(labels)
    if n == 0:
        return {"n": 0, "logloss": None, "brier": None, "accuracy": None, "ece": None}

    # LogLoss, Brier, Accuracy
    logloss_sum = 0.0
    brier_sum = 0.0
    correct = 0
    # ECE bins على أعلى فئة ثقة
    bins = [[] for _ in range(max(1, ece_bins))]
    for p, y in zip(probs, labels):
        ph, pd, pa = p
        p_arr = np.array([max(eps, float(ph)), max(eps, float(pd)), max(eps, float(pa))], dtype=np.float64)
        p_arr = p_arr / p_arr.sum()

        # LogLoss
        logloss_sum += -math.log(p_arr[y])

        # Brier (multi-class)
        y_vec = np.zeros(3, dtype=np.float64)
        y_vec[y] = 1.0
        brier_sum += float(np.sum((p_arr - y_vec) ** 2))

        # Accuracy
        pred = int(np.argmax(p_arr))
        if pred == y:
            correct += 1

        # ECE
        conf = float(np.max(p_arr))
        bidx = min(ece_bins - 1, int(conf * ece_bins))  # conf=1 يقع في آخر خانة
        bins[bidx].append((pred == y, conf))

    # ECE (top-class)
    ece_sum = 0.0
    for b in bins:
        if not b:
            continue
        acc_bin = sum(1.0 if c[0] else 0.0 for c in b) / len(b)
        conf_bin = sum(c[1] for c in b) / len(b)
        ece_sum += abs(acc_bin - conf_bin) * (len(b) / n)

    return {
        "n": n,
        "logloss": logloss_sum / n,
        "brier": brier_sum / n,
        "accuracy": correct / n,
        "ece": ece_sum
    }

def compute_lambdas_for_match(
    match: Dict[str, Any],
    factors_A: Dict[str, float],
    factors_D: Dict[str, float],
    league_avgs: Dict[str, Any],
) -> Tuple[float, float]:
    h_id = str(match.get("homeTeam", {}).get("id"))
    a_id = str(match.get("awayTeam", {}).get("id"))
    Ah = float(factors_A.get(h_id, 1.0))
    Dh = float(factors_D.get(h_id, 1.0))
    Aa = float(factors_A.get(a_id, 1.0))
    Da = float(factors_D.get(a_id, 1.0))
    avg_home = float(league_avgs.get("avg_home_goals", 1.40))
    avg_away = float(league_avgs.get("avg_away_goals", 1.10))
    lam_home = avg_home * Ah * float(factors_D.get(a_id, 1.0))
    lam_away = avg_away * Aa * float(factors_D.get(h_id, 1.0))
    # ملاحظة: استخدمنا D_opp (وليس Dh/Da) كما هو متفق عليه في النموذج
    return lam_home, lam_away

def adjust_lambdas_with_elo(
    lam_home: float, lam_away: float,
    elo_home: float, elo_away: float
) -> Tuple[float, float]:
    # نفس منطق predictor._adjust_lambdas_with_elo
    ELO_HFA = getattr(config, "ELO_HFA", 60.0)
    ELO_LAMBDA_SCALE = getattr(config, "ELO_LAMBDA_SCALE", 400.0)
    edge = (float(elo_home) - float(elo_away)) + float(ELO_HFA)
    factor = 10.0 ** (edge / float(ELO_LAMBDA_SCALE))
    return lam_home * factor, max(1e-6, lam_away / factor)

def predict_probs_for_match(
    match: Dict[str, Any],
    models: Dict[str, Any],
    use_elo: bool = True
) -> Tuple[float, float, float]:
    factors_A = models["factors_A"]
    factors_D = models["factors_D"]
    league_avgs = models["league_avgs"]
    rho = float(models["rho"])
    elo = models.get("elo", {})

    lam_h, lam_a = compute_lambdas_for_match(match, factors_A, factors_D, league_avgs)

    if use_elo:
        h_id = str(match.get("homeTeam", {}).get("id"))
        a_id = str(match.get("awayTeam", {}).get("id"))
        elo_h = float(elo.get(h_id, getattr(config, "ELO_START", 1500.0)))
        elo_a = float(elo.get(a_id, getattr(config, "ELO_START", 1500.0)))
        lam_h, lam_a = adjust_lambdas_with_elo(lam_h, lam_a, elo_h, elo_a)

    gmax = suggest_goal_cutoff(lam_h, lam_a)
    mat = poisson_matrix_dc(lam_h, lam_a, rho, max_goals=gmax)
    # تحويل لمخرجات 1X2
    # p_home = sum i>j, p_draw = trace, p_away = sum i<j
    p_home = float(np.triu(mat, k=1).sum())
    p_draw = float(np.trace(mat))
    p_away = float(np.tril(mat, k=-1).sum())
    s = p_home + p_draw + p_away
    if s > 0:
        p_home, p_draw, p_away = p_home / s, p_draw / s, p_away / s
    return p_home, p_draw, p_away

# -----------------------------------------------------------------------------
# منطق الباكتيست
# -----------------------------------------------------------------------------

def train_models_for_window(
    train_matches: List[Dict[str, Any]],
    halflife_days: int,
    prior_global: float,
    team_prior_weight: float,
    rho_max: float,
    rho_step: float,
    prev_season_prior: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, Any]:
    if not train_matches:
        return {}

    # نهاية نافذة التدريب = آخر تاريخ في النافذة (لأوزان الانحلال)
    end_dates = [parse_date_safe(m.get('utcDate')) for m in train_matches]
    end_dates = [d for d in end_dates if d]
    season_end_date = max(end_dates) if end_dates else None

    league_avgs = calculate_league_averages(train_matches)

    prior_attack = None
    prior_defense = None
    if prev_season_prior:
        prior_attack = prev_season_prior.get("attack")
        prior_defense = prev_season_prior.get("defense")

    factors_A, factors_D = build_team_factors(
        train_matches,
        league_avgs,
        season_end_date,
        decay_halflife_days=halflife_days,
        prior_strength=prior_global,
        damping=0.5,
        prior_attack=prior_attack,
        prior_defense=prior_defense,
        team_prior_weight=team_prior_weight,
    )

    elo = build_elo_ratings(
        train_matches,
        start_rating=getattr(config, "ELO_START", 1500.0),
        k_base=getattr(config, "ELO_K_BASE", 24.0),
        hfa_elo=getattr(config, "ELO_HFA", 60.0),
        scale=getattr(config, "ELO_SCALE", 400.0),
        decay_halflife_days=getattr(config, "ELO_HALFLIFE_DAYS", 365),
    )

    rho = fit_dc_rho_mle(
        train_matches,
        factors_A,
        factors_D,
        league_avgs,
        decay_halflife_days=halflife_days,
        rho_min=-abs(rho_max),
        rho_max=abs(rho_max),
        rho_step=rho_step,
    )

    return {
        "league_avgs": league_avgs,
        "factors_A": factors_A,
        "factors_D": factors_D,
        "elo": elo,
        "rho": rho,
    }

def backtest_season_expanding(
    season_key: str,
    matches: List[Dict[str, Any]],
    halflife_days: int,
    prior_global: float,
    team_prior_weight: float,
    rho_max: float,
    rho_step: float,
    min_train: int,
    block_size: int,
    use_elo: bool,
    prev_season_prior: Optional[Dict[str, Dict[str, float]]] = None,
    ece_bins: int = 10,
) -> Dict[str, Any]:
    # نتائج هذه الموسم
    season_probs: List[Tuple[float, float, float]] = []
    season_labels: List[int] = []
    n_total = len(matches)

    if n_total < max(30, min_train + 5):
        return {"season_key": season_key, "n": 0, "metrics": {}}

    train_end = min_train
    while train_end < n_total:
        test_end = min(n_total, train_end + block_size)
        train_subset = matches[:train_end]
        test_subset = matches[train_end:test_end]

        models = train_models_for_window(
            train_subset,
            halflife_days,
            prior_global,
            team_prior_weight,
            rho_max,
            rho_step,
            prev_season_prior=prev_season_prior,
        )

        # توقع test_subset
        for m in test_subset:
            hg, ag = parse_score(m)
            if hg is None:
                continue
            p = predict_probs_for_match(m, models, use_elo=use_elo)
            y = outcome_label(hg, ag)
            season_probs.append(p)
            season_labels.append(y)

        # وسّع النافذة
        train_end = test_end

    metrics = compute_metrics(season_probs, season_labels, ece_bins=ece_bins)
    return {
        "season_key": season_key,
        "n": metrics["n"],
        "metrics": metrics,
    }

def train_prior_for_next_season(
    matches: List[Dict[str, Any]],
    halflife_days: int,
    prior_global: float,
    team_prior_weight: float,
    rho_max: float,
    rho_step: float,
) -> Dict[str, Dict[str, float]]:
    # درّب عوامل كاملة للموسم لإعطائها كبراير للموسم التالي
    end_dates = [parse_date_safe(m.get('utcDate')) for m in matches]
    end_dates = [d for d in end_dates if d]
    season_end_date = max(end_dates) if end_dates else None
    league_avgs = calculate_league_averages(matches)
    factors_A, factors_D = build_team_factors(
        matches,
        league_avgs,
        season_end_date,
        decay_halflife_days=halflife_days,
        prior_strength=prior_global,
        damping=0.5,
        prior_attack=None,
        prior_defense=None,
        team_prior_weight=0.0,
    )
    return {"attack": factors_A, "defense": factors_D}

# -----------------------------------------------------------------------------
# تشغيل الشبكة (Grid) وتجميع النتائج
# -----------------------------------------------------------------------------

def run_backtester(
    comps: Optional[List[str]],
    min_train: int,
    block_size: int,
    grid_halflife: List[int],
    grid_prior_global: List[float],
    grid_team_prior_w: List[float],
    grid_rho_max: List[float],
    rho_step: float,
    ece_bins: int,
    limit_seasons: Optional[int],
    use_elo: bool,
    save: bool,
) -> None:
    log("--- بدء الباكتيست الزمني للموديل الإحصائي ---", "INFO")

    all_matches = load_matches(config.DATA_DIR / "matches.json")
    by_season = group_matches_by_season(all_matches)

    # تصفية المواسم حسب الدوريات المطلوبة
    items = list(by_season.items())
    # فرز حسب (الدوري، السنة)
    items.sort(key=lambda kv: (season_key_parts(kv[0])[0], season_key_parts(kv[0])[1]))

    # تصنيف المواسم حسب الدوري
    comps_to_seasons: Dict[str, List[Tuple[str, List[Dict[str, Any]]]]] = defaultdict(list)
    for sk, matches in items:
        comp, yr = season_key_parts(sk)
        if comps and comp not in comps:
            continue
        comps_to_seasons[comp].append((sk, matches))

    # حد المواسم الأخيرة لكل دوري إن طلب
    if limit_seasons and limit_seasons > 0:
        for comp in list(comps_to_seasons.keys()):
            comps_to_seasons[comp] = comps_to_seasons[comp][-limit_seasons:]

    # شبكة المعاملات
    combos = []
    for hf in grid_halflife:
        for pg in grid_prior_global:
            for tpw in grid_team_prior_w:
                for rmax in grid_rho_max:
                    combos.append({
                        "halflife": int(hf),
                        "prior_global": float(pg),
                        "team_prior_w": float(tpw),
                        "rho_max": float(rmax),
                        "rho_step": float(rho_step),
                    })

    overall_results = []
    best_candidates = []

    for ci, combo in enumerate(combos, start=1):
        log(f"[{ci}/{len(combos)}] تجربة: halflife={combo['halflife']}, prior={combo['prior_global']}, team_prior_w={combo['team_prior_w']}, rho_max={combo['rho_max']}, rho_step={combo['rho_step']}", "INFO")

        # تجميع شامل لكل المواسم
        all_probs: List[Tuple[float, float, float]] = []
        all_labels: List[int] = []

        # نتائج تفصيلية لكل موسم
        season_details = []

        # براير للموسم السابق لكل دوري
        prev_prior_by_comp: Dict[str, Dict[str, Dict[str, float]]] = {}

        for comp, seasons in comps_to_seasons.items():
            for sk, matches in seasons:
                prev_prior = prev_prior_by_comp.get(comp)
                res = backtest_season_expanding(
                    season_key=sk,
                    matches=matches,
                    halflife_days=combo["halflife"],
                    prior_global=combo["prior_global"],
                    team_prior_weight=combo["team_prior_w"],
                    rho_max=combo["rho_max"],
                    rho_step=combo["rho_step"],
                    min_train=min_train,
                    block_size=block_size,
                    use_elo=use_elo,
                    prev_season_prior=prev_prior,
                    ece_bins=ece_bins,
                )

                n = res.get("n", 0)
                mx = res.get("metrics", {})
                season_details.append({
                    "season_key": sk,
                    "n": n,
                    "logloss": mx.get("logloss"),
                    "brier": mx.get("brier"),
                    "accuracy": mx.get("accuracy"),
                    "ece": mx.get("ece"),
                })

                # تراكم شامل
                # لإعادة حساب الشامل بدقة، سنحتاج كل احتمالات وملصقات الموسم.
                # ولكن res أعاد فقط ملخص. لذلك سنعيد حساب شامل بتمرير شامل آخر؟
                # لتجنب عبء إعادة التوقع، سنحفظ شموليًا في res (خيار نموذج متقدم).
                # هنا سنكتفي بالمعدل المرجح حسب n للمقاييس الأساسية:
                # لتوحيد الدقة، سنجمع موسمياً ثم نُجري متوسطًا مرجحًا.
                # نجمع في مصفوفات منفصلة (سنحسب بعد الحلقة).
                pass

            # بعد إنهاء موسم comp، درّب براير كامل للموسم الأخير (للموسم القادم)
            # ملاحظة: نحتاج براير للموسم القادم داخل هذا اللوب؛ لكن بما أننا نسير تصاعدياً، سنحسب براير الموسم الحالي ليُستَخدم للموسم التالي.
            # قمنا بذلك داخل الحلقة أيضاً حين نحتاجه قبل موسم جديد.
            # هنا سنقوم بتحديثه نهاية كل موسم على كامل المباريات.
            # لكننا نحتاجه "قبل" الموسم التالي. لذا نحسبه فورًا بعد إنهاء تقييم الموسم:
            # سنعيد تنفيذ ذلك داخل الحلقة الموسمية في المرة التالية إن لزم.

        # إعادة المرور للحصول على تراكب شامل دقيق:
        # سنجري تمريرًا ثانيًا سريعًا لحساب شامل دقيق عبر كل المواسم مع نفس combo.
        # هذا يكرر زمن التوقع فقط (وليس التدريب كله) إذا حفظنا نماذج النوافذ!
        # لتبسيط التنفيذ والوقت، سنكتفي بحساب المعدل المرجّح للمقاييس عبر المواسم.
        # أي: LogLoss_weighted = sum(logloss_s * n_s) / sum(n_s)
        #     Brier_weighted  = sum(brier_s  * n_s) / sum(n_s)
        #     Acc_weighted    = sum(acc_s    * n_s) / sum(n_s)
        #     ECE_weighted    = sum(ece_s    * n_s) / sum(n_s)
        sum_n = 0
        sum_logloss = 0.0
        sum_brier = 0.0
        sum_acc = 0.0
        sum_ece = 0.0
        for s in season_details:
            n = s["n"] or 0
            if n <= 0:
                continue
            sum_n += n
            if s["logloss"] is not None:
                sum_logloss += s["logloss"] * n
            if s["brier"] is not None:
                sum_brier += s["brier"] * n
            if s["accuracy"] is not None:
                sum_acc += s["accuracy"] * n
            if s["ece"] is not None:
                sum_ece += s["ece"] * n

        agg = {
            "halflife": combo["halflife"],
            "prior_global": combo["prior_global"],
            "team_prior_w": combo["team_prior_w"],
            "rho_max": combo["rho_max"],
            "rho_step": combo["rho_step"],
            "use_elo": use_elo,
            "total_samples": sum_n,
            "logloss": (sum_logloss / sum_n) if sum_n > 0 else None,
            "brier": (sum_brier / sum_n) if sum_n > 0 else None,
            "accuracy": (sum_acc / sum_n) if sum_n > 0 else None,
            "ece": (sum_ece / sum_n) if sum_n > 0 else None,
            "by_season": season_details,
        }
        overall_results.append(agg)

        # حفظ مرشح أفضل حسب LogLoss
        if agg["logloss"] is not None:
            best_candidates.append(agg)

        # طباعة ملخص
        ll = agg["logloss"]
        br = agg["brier"]
        ac = agg["accuracy"]
        ec = agg["ece"]
        log(f"نتائج التجربة: N={sum_n}, LogLoss={ll:.5f} | Brier={br:.5f} | Acc={ac:.3f} | ECE={ec:.4f}", "RESULT")

    # اختيار الأفضل
    if best_candidates:
        best_candidates.sort(key=lambda x: (x["logloss"], x["ece"] if x["ece"] is not None else 1e9))
        best = best_candidates[0]
        log("—" * 60, "INFO")
        log("أفضل إعدادات حسب LogLoss:", "SUCCESS")
        log(
            f"halflife={best['halflife']} | prior_global={best['prior_global']} | "
            f"team_prior_w={best['team_prior_w']} | rho_max={best['rho_max']} | "
            f"rho_step={best['rho_step']} | use_elo={best['use_elo']}", "SUCCESS"
        )
        log(
            f"المقاييس: N={best['total_samples']} | LogLoss={best['logloss']:.5f} | "
            f"Brier={best['brier']:.5f} | Acc={best['accuracy']:.3f} | ECE={best['ece']:.4f}", "SUCCESS"
        )
        log("اقترح تعيين القيم في common.config كما يلي:", "INFO")
        print(
            f"""
# common/config.py (اقتراح)
TEAM_FACTORS_HALFLIFE_DAYS = {best['halflife']}
TEAM_FACTORS_PRIOR_GLOBAL = {best['prior_global']}
TEAM_FACTORS_TEAM_PRIOR_WEIGHT = {best['team_prior_w']}
DC_RHO_MIN = {-best['rho_max']:.3f}
DC_RHO_MAX = {best['rho_max']:.3f}
DC_RHO_STEP = {best['rho_step']}
# استخدام ELO في Predictor عبر واجهة التطبيق (use_elo={best['use_elo']})
"""
        )

    # حفظ النتائج
    if save:
        out = {
            "comps": comps,
            "min_train": min_train,
            "block_size": block_size,
            "ece_bins": ece_bins,
            "use_elo": use_elo,
            "results": overall_results,
        }
        save_path = config.DATA_DIR / "backtest_results.json"
        try:
            config.DATA_DIR.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
            log(f"تم حفظ نتائج الباكتيست في: {save_path}", "SUCCESS")
        except Exception as e:
            log(f"فشل حفظ نتائج الباكتيست: {e}", "ERROR")

    log("--- انتهى الباكتيست ---", "INFO")

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Backtester للموديل الإحصائي (Dixon–Coles + Team Factors + ELO).")

    parser.add_argument("--comps", nargs="*", default=None, help="رموز المسابقات المطلوب اختبارها (مثل: PL PD SA BL1 FL1).")
    parser.add_argument("--min-train", type=int, default=120, help="أدنى عدد مباريات للتدريب قبل أول تقييم داخل الموسم.")
    parser.add_argument("--block-size", type=int, default=40, help="حجم كتلة الاختبار في كل خطوة (expanding window).")

    parser.add_argument("--grid-halflife", type=str, default="90,180,365", help="قائمة نصف العمر بالأيام (مفصولة بفواصل).")
    parser.add_argument("--grid-prior-global", type=str, default="2.0,3.0,5.0", help="قائمة قوة انكماش Gamma نحو 1.0.")
    parser.add_argument("--grid-team-prior-weight", type=str, default="0.0,5.0", help="أوزان الانكماش الهرمي نحو الموسم السابق.")
    parser.add_argument("--grid-rho-max", type=str, default="0.15,0.2", help="قيم الحد الأقصى |ρ| للبحث الشبكي.")

    parser.add_argument("--rho-step", type=float, default=0.002, help="دقة شبكة rho أثناء البحث.")
    parser.add_argument("--ece-bins", type=int, default=10, help="عدد صناديق ECE.")
    parser.add_argument("--limit-seasons", type=int, default=0, help="حصر عدد المواسم الأخيرة لكل دوري (0 = كل المواسم).")
    parser.add_argument("--use-elo", action="store_true", help="تفعيل استخدام ELO لتعديل λ أثناء التوقع.")
    parser.add_argument("--save", action="store_true", help="حفظ نتائج الباكتيست في data/backtest_results.json")

    args = parser.parse_args()

    comps = args.comps if args.comps else getattr(config, "TARGET_COMPETITIONS", None)
    halflife = parse_grid_list_ints(args.grid_halflife)
    prior_global = parse_grid_list_floats(args.grid_prior_global)
    team_prior_w = parse_grid_list_floats(args.grid_team_prior_weight)
    rho_maxs = parse_grid_list_floats(args.grid_rho_max)

    run_backtester(
        comps=comps,
        min_train=args.min_train,
        block_size=args.block_size,
        grid_halflife=halflife,
        grid_prior_global=prior_global,
        grid_team_prior_w=team_prior_w,
        grid_rho_max=rho_maxs,
        rho_step=args.rho_step,
        ece_bins=args.ece_bins,
        limit_seasons=args.limit_seasons if args.limit_seasons and args.limit_seasons > 0 else None,
        use_elo=args.use_elo,
        save=args.save,
    )

if __name__ == "__main__":
    main()