# app.py (النسخة النهائية)

import sys
import os
from pathlib import Path

# --- ✅ الحل النهائي لمشكلة المسار (Path) ---
# This block MUST be at the very top of the file.
# It finds the root folder of your project and adds it to the list
# of places Python looks for modules, ensuring 'common' is always found.
APP_ROOT = Path(__file__).resolve().parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))
# --- نهاية الحل ---

import json
import subprocess
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import streamlit as st

from common import config
from common.utils import enhanced_team_search
from common.modeling import poisson_matrix_dc, matrix_to_outcomes, top_scorelines

# ... (The rest of the app code remains exactly the same as the last version I gave you) ...

st.set_page_config(page_title="⚽ Football Predictor", page_icon="⚽", layout="wide")

@st.cache_data
def _load_json(path: Path) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

@st.cache_data
def load_teams_map() -> Optional[Dict]:
    return _load_json(config.DATA_DIR / "teams.json")

@st.cache_data
def load_models() -> Dict[str, dict]:
    return {
        "averages": _load_json(config.MODELS_DIR / "league_averages.json") or {},
        "factors": _load_json(config.MODELS_DIR / "team_factors.json") or {},
        "elo": _load_json(config.MODELS_DIR / "elo_ratings.json") or {},
        "rho": _load_json(config.MODELS_DIR / "rho_values.json") or {},
    }

def current_season_year(now: datetime) -> int:
    return now.year if now.month >= config.CURRENT_SEASON_START_MONTH else now.year - 1

def _primary_name(names: List[str]) -> str:
    names = [n for n in (names or []) if n]
    if not names: return "Unknown"
    def score(n: str) -> Tuple[int, int, int]:
        return (int(" " in n), len(n), -int(n.isupper()))
    return sorted(names, key=score, reverse=True)[0]

def teams_for_comp(teams_map: Dict, comp_code: str) -> List[Tuple[str, int]]:
    out = []
    for t in teams_map.values():
        comps = t.get("competitions", [])
        if comp_code in comps:
            out.append((_primary_name(t.get("names", [])), t.get("id")))
    out = [(name, tid) for name, tid in out if tid]
    out.sort(key=lambda x: x[0].lower())
    return out

def run_pipeline_cli(years: int) -> Tuple[bool, str]:
    cmd = [sys.executable, str(APP_ROOT / "01_pipeline.py"), "--years", str(years)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        ok = (result.returncode == 0)
        output = (result.stdout or "") + "\n" + (result.stderr or "")
        return ok, output
    except Exception as e:
        return False, f"Failed to run pipeline: {e}"

def run_trainer_cli() -> Tuple[bool, str]:
    cmd = [sys.executable, str(APP_ROOT / "02_trainer.py")]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        ok = (result.returncode == 0)
        output = (result.stdout or "") + "\n" + (result.stderr or "")
        return ok, output
    except Exception as e:
        return False, f"Failed to run trainer: {e}"

def compute_prediction(
    team1_name: str, team2_name: str, comp_code: str, use_elo: bool, topk: int
) -> Tuple[dict, List[List[float]]]:
    teams_map = load_teams_map()
    if not teams_map: raise RuntimeError("Teams map not found. Please run the data pipeline first.")
    home_id = enhanced_team_search(team1_name, teams_map, comp_code)
    away_id = enhanced_team_search(team2_name, teams_map, comp_code)
    if not home_id or not away_id: raise ValueError(f"Could not find one or both teams: '{team1_name}', '{team2_name}'")
    
    season_year = current_season_year(datetime.now())
    season_key = f"{comp_code}_{season_year}"
    last_season_key = f"{comp_code}_{season_year - 1}"
    
    models = load_models()
    if season_key not in models["elo"]:
        season_key = last_season_key
        if season_key not in models["elo"]:
            raise RuntimeError(f"No recent model available for competition {comp_code}. Train models first.")
            
    elo = models["elo"].get(season_key, {})
    elo_home = float(elo.get(str(home_id), 1500))
    elo_away = float(elo.get(str(away_id), 1500))
    factors = models["factors"].get(season_key, {})
    home_attack = float(factors.get("attack", {}).get(str(home_id), 1.0))
    home_defense = float(factors.get("defense", {}).get(str(home_id), 1.0))
    away_attack = float(factors.get("attack", {}).get(str(away_id), 1.0))
    away_defense = float(factors.get("defense", {}).get(str(away_id), 1.0))
    avgs = models["averages"].get(season_key, {})
    avg_home = float(avgs.get("avg_home_goals", 1.4))
    avg_away = float(avgs.get("avg_away_goals", 1.1))
    rho = float(models["rho"].get(season_key, 0.0))
    lam_home = home_attack * away_defense * avg_home
    lam_away = away_attack * home_defense * avg_away
    
    if use_elo:
        edge = (elo_home - el_away) + config.ELO_HFA
        factor = 10 ** (edge / config.ELO_LAMBDA_SCALE)
        lam_home = lam_home * factor
        lam_away = max(1e-6, lam_away / factor)
        
    matrix = poisson_matrix_dc(lam_home, lam_away, rho, max_goals=8)
    p_home, p_draw, p_away = matrix_to_outcomes(matrix)
    
    result = {
        "meta": {"version": config.VERSION, "model_season_used": season_key},
        "match": f"{team1_name} (Home) vs {team2_name} (Away)",
        "competition": comp_code,
        "teams_found": {"home": {"name": team1_name, "id": home_id}, "away": {"name": team2_name, "id": away_id}},
        "model_inputs": {"lambda_home": round(lam_home, 3), "lambda_away": round(lam_away, 3), "rho": round(rho, 3), "use_elo_adjust": use_elo},
        "probabilities": {"home_win": p_home, "draw": p_draw, "away_win": p_away}
    }
    
    if topk and topk > 0:
        tops = top_scorelines(matrix, top_k=topk)
        result["top_scorelines"] = [{"home_goals": i, "away_goals": j, "prob": p} for i, j, p in tops]
        
    return result, matrix

st.title("⚽ Football Predictor")
st.caption("Dixon–Coles + ELO Model")

with st.sidebar:
    st.header("Data & Models")
    if st.button("Run Data Pipeline"):
        with st.spinner("Building database... this might take a while."):
            ok, logs = run_pipeline_cli(5)
        st.cache_data.clear()
        if ok: st.success("Data built successfully.")
        else: st.error("Data pipeline failed.")
        with st.expander("Logs"): st.code(logs)
            
    if st.button("Train Models"):
        with st.spinner("Training models..."):
            ok, logs = run_trainer_cli()
        st.cache_data.clear()
        if ok: st.success("Models trained successfully.")
        else: st.error("Model training failed.")
        with st.expander("Logs"): st.code(logs)
            
    st.divider()
    use_elo = st.checkbox("Enable ELO adjustment", value=True)
    topk = st.slider("Top K scorelines", 0, 10, 5)

st.subheader("Match Prediction")
comp_code = st.selectbox("Competition", options=config.TARGET_COMPETITIONS)
teams_map = load_teams_map()
if teams_map:
    comp_teams = teams_for_comp(teams_map, comp_code)
    if comp_teams:
        names = [n for n, _ in comp_teams]
        c1, c2 = st.columns(2)
        team1_name = c1.selectbox("Home Team", options=names, index=0)
        team2_name = c2.selectbox("Away Team", options=names, index=1)
        
        if st.button("Predict"):
            try:
                result, _ = compute_prediction(team1_name, team2_name, comp_code, use_elo, topk)
                st.json(result)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
    else:
        st.warning("No teams found for this competition.")
else:
    st.warning("No local team data found. Run the data pipeline first.")
