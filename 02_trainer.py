# 02_trainer.py
import json
from collections import defaultdict
from common.config import config
from common.utils import log, parse_date_safe
from common.modeling import (
    calculate_league_averages, build_team_factors, build_elo_ratings,
    fit_dc_rho_mle
)

def run_trainer():
    log("--- Starting Model Trainer ---", "INFO")

    try:
        with open(config.DATA_DIR / "matches.json", 'r', encoding='utf-8') as f:
            all_matches = json.load(f)
    except IOError:
        log("matches.json not found. Please run 01_pipeline.py first.", "CRITICAL")
        return

    matches_by_season = defaultdict(list)
    for match in all_matches:
        season_year = match.get('season', {}).get('startDate', '1900')[:4]
        comp_code = match.get('competition', {}).get('code', 'UNK')
        season_key = f"{comp_code}_{season_year}"
        matches_by_season[season_key].append(match)

    log(f"Grouped matches into {len(matches_by_season)} unique seasons.")

    models = {
        "team_factors": {}, "elo_ratings": {},
        "league_averages": {}, "rho_values": {}
    }

    for season_key, matches in matches_by_season.items():
        if len(matches) < 30:
            continue
        log(f"Training model for season: {season_key}")

        end_dates = [d for d in (parse_date_safe(m.get('utcDate')) for m in matches) if d]
        if not end_dates:
            continue
        season_end_date = max(end_dates)

        league_avgs = calculate_league_averages(matches)
        factors_A, factors_D = build_team_factors(matches, league_avgs, season_end_date)
        elo = build_elo_ratings(matches)
        rho = fit_dc_rho_mle(matches, factors_A, factors_D, league_avgs)

        models["league_averages"][season_key] = league_avgs
        models["team_factors"][season_key] = {"attack": factors_A, "defense": factors_D}
        models["elo_ratings"][season_key] = elo
        models["rho_values"][season_key] = rho

    for name, data in models.items():
        path = config.MODELS_DIR / f"{name}.json"
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        log(f"Saved {name} model to {path}")

    log("--- Model Trainer Finished ---", "INFO")

if __name__ == "__main__":
    run_trainer()
