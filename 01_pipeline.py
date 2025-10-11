# 01_pipeline.py
import json
import argparse
from common.config import config
from common.api_client import APIClient
from common.utils import log

def run_pipeline():
    log("--- Starting Data Pipeline (Free Tier Compatible) ---", "INFO")
    log("NOTE: This script will only fetch finished matches from the CURRENT season.", "WARNING")
    client = APIClient()

    target_comps = client.get_competitions()
    if not target_comps:
        log("Could not fetch competitions. Exiting.", "CRITICAL")
        return

    all_matches = {}
    # تم تغيير الحلقة لتناسب الخطة المجانية: لا نطلب سنوات ماضية
    for code, comp_id in target_comps.items():
        log(f"Fetching current season matches for {code}...")
        # استدعاء الدالة الجديدة التي تعمل مع الخطة المجانية
        current_matches = client.get_current_matches_for_competition(comp_id)
        if current_matches:
            log(f"Found {len(current_matches)} finished matches.")
            for match in current_matches:
                all_matches[match['id']] = match
        else:
            log(f"No finished matches found for {code} in the current season via API.")

    log(f"Total unique matches collected: {len(all_matches)}")
    if not all_matches:
        log("No match data was collected. This might be because no matches have finished yet in the current season.", "CRITICAL")

    output_path = config.DATA_DIR / "matches.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(list(all_matches.values()), f, ensure_ascii=False, indent=2)
    log(f"Match data saved to {output_path}")
    
    # لا داعي لإيقاف السكريبت، جلب الفرق لا يزال مفيدًا
    teams_data = client.get_teams_for_competitions(list(target_comps.values()))
    teams_path = config.DATA_DIR / "teams.json"
    with open(teams_path, 'w', encoding='utf-8') as f:
        json.dump(teams_data, f, ensure_ascii=False, indent=2)
    log(f"Team data saved to {teams_path}")

    log("--- Data Pipeline Finished ---", "INFO")

if __name__ == "__main__":
    # تم حذف خيار --years لأنه لم يعد له استخدام في هذه الطريقة
    parser = argparse.ArgumentParser(description="Build local dataset from football-data.org (Free Tier Compatible)")
    args = parser.parse_args()
    run_pipeline()
