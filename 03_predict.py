# 03_predict.py
import json
import argparse
from datetime import datetime
from typing import Dict
from common.config import config
from common.utils import log, enhanced_team_search
from common.modeling import poisson_matrix_dc, matrix_to_outcomes, top_scorelines

def current_season_year(now: datetime) -> int:
    return now.year if now.month >= config.CURRENT_SEASON_START_MONTH else now.year - 1

class Predictor:
    def __init__(self):
        self.models = self._load_models()
        self.teams_map = self._load_teams_map()

    def _load_json(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_models(self) -> Dict:
        log("Loading pre-trained models...")
        return {
            "averages": self._load_json(config.MODELS_DIR / "league_averages.json"),
            "factors": self._load_json(config.MODELS_DIR / "team_factors.json"),
            "elo": self._load_json(config.MODELS_DIR / "elo_ratings.json"),
            "rho": self._load_json(config.MODELS_DIR / "rho_values.json"),
        }

    def _load_teams_map(self) -> Dict:
        log("Loading teams map...")
        return self._load_json(config.DATA_DIR / "teams.json")

    def _adjust_lambdas_with_elo(self, lam_home: float, lam_away: float, elo_home: float, elo_away: float) -> (float, float):
        edge = (elo_home - elo_away) + config.ELO_HFA
        factor = 10 ** (edge / config.ELO_LAMBDA_SCALE)
        return lam_home * factor, max(1e-6, lam_away / factor)

    def predict(self, team1_name: str, team2_name: str, comp_code: str, topk: int = 0, use_elo: bool = False):
        comp_code = comp_code.upper()
        log(f"Prediction request for: {team1_name} vs {team2_name} in {comp_code}")

        home_id = enhanced_team_search(team1_name, self.teams_map, comp_code)
        away_id = enhanced_team_search(team2_name, self.teams_map, comp_code)
        if not home_id or not away_id:
            raise ValueError(f"Could not find one or both teams: '{team1_name}', '{team2_name}'")

        season_year = current_season_year(datetime.now())
        season_key = f"{comp_code}_{season_year}"
        last_season_key = f"{comp_code}_{season_year - 1}"

        if season_key not in self.models["elo"]:
            log(f"Model for current season ({season_key}) not found. Falling back to last season.", "WARNING")
            season_key = last_season_key
            if season_key not in self.models["elo"]:
                raise ValueError(f"No recent model available for competition {comp_code}.")

        elo = self.models["elo"].get(season_key, {})
        elo_home = float(elo.get(str(home_id), 1500))
        elo_away = float(elo.get(str(away_id), 1500))

        factors = self.models["factors"].get(season_key, {})
        home_attack = factors.get("attack", {}).get(str(home_id), 1.0)
        home_defense = factors.get("defense", {}).get(str(home_id), 1.0)
        away_attack = factors.get("attack", {}).get(str(away_id), 1.0)
        away_defense = factors.get("defense", {}).get(str(away_id), 1.0)

        avgs = self.models["averages"].get(season_key, {})
        avg_home = avgs.get("avg_home_goals", 1.4)
        avg_away = avgs.get("avg_away_goals", 1.1)
        rho = float(self.models["rho"].get(season_key, 0.0))

        lam_home = home_attack * away_defense * avg_home
        lam_away = away_attack * home_defense * avg_away

        if use_elo:
            lam_home, lam_away = self._adjust_lambdas_with_elo(lam_home, lam_away, elo_home, elo_away)

        matrix = poisson_matrix_dc(lam_home, lam_away, rho, max_goals=8)
        p_home, p_draw, p_away = matrix_to_outcomes(matrix)

        result = {
            "meta": {"version": config.VERSION, "model_season_used": season_key},
            "match": f"{team1_name} (Home) vs {team2_name} (Away)",
            "competition": comp_code,
            "teams_found": {
                "home": {"name": team1_name, "id": home_id},
                "away": {"name": team2_name, "id": away_id}
            },
            "model_inputs": {
                "lambda_home": round(lam_home, 3),
                "lambda_away": round(lam_away, 3),
                "rho": round(rho, 3),
                "use_elo_adjust": use_elo
            },
            "probabilities": {
                "home_win": f"{p_home * 100:.1f}%",
                "draw": f"{p_draw * 100:.1f}%",
                "away_win": f"{p_away * 100:.1f}%"
            }
        }

        if topk and topk > 0:
            tops = top_scorelines(matrix, top_k=topk)
            result["top_scorelines"] = [
                {"home_goals": i, "away_goals": j, "prob": f"{p * 100:.2f}%"} for i, j, p in tops
            ]

        print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict a football match using pre-trained models.")
    parser.add_argument("--team1", required=True, help="Home team name.")
    parser.add_argument("--team2", required=True, help="Away team name.")
    parser.add_argument("--comp", required=True, help="Competition code (e.g., PL, PD, SA, BL1, FL1, CL, DED, PPL, BSA).")
    parser.add_argument("--topk", type=int, default=0, help="Show top-K most probable scorelines.")
    parser.add_argument("--use-elo", action="store_true", help="Apply ELO-based adjustment to goal rates.")
    args = parser.parse_args()

    try:
        predictor = Predictor()
        predictor.predict(args.team1, args.team2, args.comp, topk=args.topk, use_elo=args.use_elo)
    except Exception as e:
        log(f"Prediction failed: {e}", "CRITICAL")
