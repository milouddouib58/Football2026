# 01_pipeline.py
import json
from datetime import datetime
from common.config import config
from common.api_client import APIClient
from common.utils import log

def run_pipeline(years_to_fetch: int = 15):
    log("--- Starting Data Pipeline ---", "INFO")
    client = APIClient()

    target_comps = client.get_competitions()
    if not target_comps:
        log("Could not fetch competitions. Exiting.", "CRITICAL")
        return

    current_year = datetime.now().year
    years = range(current_year, current_year - years_to_fetch, -1)

    all_matches = {}
    for code, comp_id in target_comps.items():
        for year in years:
            log(f"Fetching matches for {code} in {year}...")
            matches_in_year = client.get_matches_for_year(year, comp_id)
            if matches_in_year:
                log(f"Found {len(matches_in_year)} matches.")
                for match in matches_in_year:
                    all_matches[match['id']] = match

    log(f"Total unique matches collected: {len(all_matches)}")
    output_path = config.DATA_DIR / "matches.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(list(all_matches.values()), f, ensure_ascii=False, indent=2)
    log(f"Match data saved to {output_path}")

    # teams
    teams_data = client.get_teams_for_competitions(list(target_comps.values()))
    teams_path = config.DATA_DIR / "teams.json"
    with open(teams_path, 'w', encoding='utf-8') as f:
        json.dump(teams_data, f, ensure_ascii=False, indent=2)
    log(f"Team data saved to {teams_path}")

    log("--- Data Pipeline Finished ---", "INFO")

if __name__ == "__main__":
    run_pipeline()