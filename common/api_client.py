# common/api_client.py
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import List, Dict, Optional

from .config import config
from .utils import log

def _build_retry() -> Retry:
    try:
        return Retry(
            total=config.MAX_RETRIES,
            backoff_factor=1,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET"])
        )
    except TypeError:
        # ملاءمة لإصدارات أقدم
        return Retry(
            total=config.MAX_RETRIES,
            backoff_factor=1,
            status_forcelist=(429, 500, 502, 503, 504),
            method_whitelist=frozenset(["GET"])
        )

class APIClient:
    def __init__(self):
        self._session = requests.Session()
        self._session.headers.update({
            "X-Auth-Token": config.API_KEY,
            "User-Agent": f"FD-Predictor/{config.VERSION}"
        })
        retries = _build_retry()
        self._session.mount("https://", HTTPAdapter(max_retries=retries))
        self._last_call_ts = 0.0

    def _rate_limit(self):
        elapsed = time.time() - self._last_call_ts
        wait_for = config.MIN_INTERVAL_SEC - elapsed
        if wait_for > 0:
            time.sleep(wait_for)
        self._last_call_ts = time.time()

    def _make_request(self, path: str, params: Optional[Dict] = None) -> Optional[Dict]:
        self._rate_limit()
        url = f"{config.BASE_URL}{path}"
        try:
            response = self._session.get(url, params=params, timeout=config.TIMEOUT)
            if response.status_code == 429:
                wait_sec = int(response.headers.get("Retry-After", 60))
                log(f"Rate limit hit. Waiting {wait_sec}s...", "WARNING")
                time.sleep(wait_sec)
                return self._make_request(path, params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            log(f"API request failed for {url}: {e}", "ERROR")
            return None

    def get_competitions(self) -> Dict[str, int]:
        log("Fetching competitions map...")
        data = self._make_request("/competitions")
        if not data or "competitions" not in data:
            return {}
        comp_map = {c['code']: c['id'] for c in data['competitions'] if c.get('code') and c.get('id')}
        target_map = {code: comp_map[code] for code in config.TARGET_COMPETITIONS if code in comp_map}
        log(f"Found IDs for {len(target_map)} target competitions.")
        return target_map

    def get_matches_for_year(self, year: int, competition_id: int) -> List[Dict]:
        params = {
            "competitions": competition_id,
            "dateFrom": f"{year}-01-01",
            "dateTo": f"{year}-12-31",
            "status": "FINISHED"
        }
        data = self._make_request("/matches", params=params)
        return data.get("matches", []) if data else []

    def get_teams_for_competitions(self, comp_ids: List[int]) -> Dict:
        log("Fetching teams for all target competitions...")
        all_teams: Dict[int, Dict] = {}
        for comp_id in comp_ids:
            data = self._make_request(f"/competitions/{comp_id}/teams")
            if data and "teams" in data:
                comp_code = data.get("competition", {}).get("code", "UNK")
                for team in data["teams"]:
                    tid = team["id"]
                    if tid not in all_teams:
                        all_teams[tid] = {
                            "id": tid,
                            "names": list(filter(None, {team.get("name"), team.get("shortName"), team.get("tla")})),
                            "competitions": set()
                        }
                    all_teams[tid]["competitions"].add(comp_code)
        # تحويل المجموعات إلى قوائم لتخزين JSON
        for tid in list(all_teams.keys()):
            all_teams[tid]["competitions"] = list(all_teams[tid]["competitions"])
        log(f"Found {len(all_teams)} unique teams across competitions.")
        return all_teams
