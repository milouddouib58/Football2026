# -----------------------------------------------------------------------------
# common/api_client.py
# -----------------------------------------------------------------------------
# الوصف:
#   يحتوي هذا الملف على الكلاس `APIClient` المسؤول عن كافة عمليات التواصل
#   مع واجهة برمجة التطبيقات (API) الخاصة بموقع football-data.org.
#   يشمل ذلك:
#   - إعداد الطلبات مع الترويسات الصحيحة ومفتاح API.
#   - تطبيق استراتيجية إعادة المحاولة عند حدوث أخطاء في الشبكة أو الخادم.
#   - احترام حدود معدل الطلبات (Rate Limiting) لتجنب الحظر.
#   - تغليف (encapsulation) نقاط النهاية (endpoints) المختلفة في دوال واضحة.
# -----------------------------------------------------------------------------

import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import List, Dict, Optional, Any

from .config import config
from .utils import log


def _build_retry_strategy() -> Retry:
    """
    ينشئ استراتيجية إعادة محاولة مرنة للتعامل مع أخطاء الخادم المؤقتة.
    
    يعيد المحاولة عند رموز الحالة التالية:
    - 429: تم تجاوز حد الطلبات (Too Many Requests).
    - 5xx: أخطاء الخادم (Server Errors).
    """
    try:
        # الإصدارات الحديثة من urllib3
        return Retry(
            total=config.MAX_RETRIES,
            backoff_factor=1,  # يضيف تأخير بسيط بين المحاولات
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
    except TypeError:
        # للحفاظ على التوافقية مع الإصدارات القديمة من urllib3
        return Retry(
            total=config.MAX_RETRIES,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            method_whitelist=["GET"],  # الاسم القديم للمعامل
        )


class APIClient:
    """
    كلاس مخصص لإدارة جميع التفاعلات مع football-data.org API.
    """
    def __init__(self):
        """يقوم بإعداد جلسة (session) طلبات مع الترويسات واستراتيجية إعادة المحاولة."""
        self._session = requests.Session()
        self._session.headers.update({
            "X-Auth-Token": config.API_KEY,
            "User-Agent": f"FD-Predictor/{config.VERSION}",
        })
        retries = _build_retry_strategy()
        adapter = HTTPAdapter(max_retries=retries)
        self._session.mount("https://", adapter)
        self._last_call_timestamp = 0.0

    def _apply_rate_limit(self):
        """
        يضمن احترام حدود معدل الطلبات التي يفرضها الـAPI.

        يستخدم `MAX_CALLS_PER_MINUTE` من ملف الإعدادات إن وجد،
        وإلا يستخدم الفاصل الزمني الافتراضي الآمن (6 ثواني بين كل طلب).
        """
        max_calls = getattr(config, "MAX_CALLS_PER_MINUTE", 10) # الافتراضي 10 طلبات/دقيقة
        min_interval = 60.0 / float(max_calls)

        elapsed_since_last_call = time.time() - self._last_call_timestamp
        wait_duration = min_interval - elapsed_since_last_call

        if wait_duration > 0:
            time.sleep(wait_duration)
        
        self._last_call_timestamp = time.time()

    def _make_request(self, path: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """
        يقوم بتنفيذ طلب GET إلى الـAPI مع معالجة الأخطاء واحترام حدود الطلبات.

        Args:
            path (str): مسار نقطة النهاية (e.g., "/competitions").
            params (Optional[Dict]): المعاملات (parameters) التي سترسل مع الطلب.

        Returns:
            Optional[Dict]: بيانات الـJSON المستلمة من الـAPI، أو None في حالة الفشل.
        """
        self._apply_rate_limit()
        url = f"{config.BASE_URL}{path}"
        
        try:
            response = self._session.get(url, params=params, timeout=config.TIMEOUT)
            response.raise_for_status()  # يطلق استثناء (exception) لرموز الحالة 4xx أو 5xx
            return response.json()
        except requests.exceptions.RequestException as e:
            log(f"فشل طلب الـAPI لـ {url} (السبب: {e})", "ERROR")
            return None

    def get_competitions(self) -> Dict[str, int]:
        """
        يجلب قائمة المسابقات ويقوم بترشيحها بناءً على القائمة المحددة في `config`.

        Returns:
            Dict[str, int]: قاموس يربط رمز المسابقة (e.g., "PL") بالمعرف الرقمي الخاص بها (ID).
        """
        log("جاري جلب معرفات المسابقات المستهدفة...", "INFO")
        data = self._make_request("/competitions")
        if not data or "competitions" not in data:
            return {}

        # إنشاء قاموس كامل يربط الرمز بالمعرف
        all_comps_map = {
            c.get('code'): c.get('id') 
            for c in data['competitions'] if c.get('code') and c.get('id')
        }
        
        # ترشيح القاموس ليحتوي فقط على المسابقات المستهدفة
        target_map = {
            code: all_comps_map[code] 
            for code in config.TARGET_COMPETITIONS if code in all_comps_map
        }
        
        log(f"تم العثور على {len(target_map)} مسابقة مستهدفة.", "INFO")
        return target_map

    def get_matches_for_season(self, season_year: int, competition_id: int) -> List[Dict]:
        """
        يجلب جميع المباريات المنتهية لموسم ومسابقة محددين.

        Args:
            season_year (int): سنة بداية الموسم (e.g., 2023 لموسم 2023/24).
            competition_id (int): المعرف الرقمي للمسابقة.

        Returns:
            List[Dict]: قائمة ببيانات المباريات.
        """
        params = {"season": season_year, "status": "FINISHED"}
        path = f"/competitions/{competition_id}/matches"
        data = self._make_request(path, params=params)
        return data.get("matches", []) if data else []

    def get_teams_for_competitions(self, comp_ids: List[int]) -> Dict[int, Dict]:
        """
        يجلب جميع الفرق المشاركة في قائمة من المسابقات ويجمعها في قاموس فريد.

        Args:
            comp_ids (List[int]): قائمة بالمعرفات الرقمية للمسابقات.

        Returns:
            Dict[int, Dict]: قاموس يحتوي على بيانات الفرق الفريدة، حيث المفتاح هو ID الفريق.
        """
        log("جاري تجميع بيانات الفرق من جميع المسابقات...", "INFO")
        all_teams: Dict[int, Dict[str, Any]] = {}

        for comp_id in comp_ids:
            data = self._make_request(f"/competitions/{comp_id}/teams")
            if not data or "teams" not in data:
                continue

            comp_code = data.get("competition", {}).get("code", "UNKNOWN")
            
            for team in data["teams"]:
                team_id = team.get("id")
                if not team_id:
                    continue

                # إذا لم يكن الفريق موجودًا في القاموس، قم بإضافته
                if team_id not in all_teams:
                    all_teams[team_id] = {
                        "id": team_id,
                        "names": list(filter(None, {team.get("name"), team.get("shortName"), team.get("tla")})),
                        "competitions": set(), # استخدم مجموعة (set) لتجنب التكرار
                    }
                
                # أضف المسابقة الحالية إلى قائمة مسابقات الفريق
                all_teams[team_id]["competitions"].add(comp_code)

        # تحويل مجموعات المسابقات إلى قوائم لتكون متوافقة مع JSON
        for team_id in all_teams:
            all_teams[team_id]["competitions"] = sorted(list(all_teams[team_id]["competitions"]))

        log(f"تم العثور على {len(all_teams)} فريق فريد عبر جميع المسابقات.", "INFO")
        return all_teams

