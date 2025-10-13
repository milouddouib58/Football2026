# 03_predict.py
# -----------------------------------------------------------------------------
# CLI للتنبؤ بمباراة باستخدام الموديلات المدربة عبر كلاس Predictor الموحد.
# -----------------------------------------------------------------------------
import json
import argparse

from common.utils import log
from predictor import Predictor

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
        result = predictor.predict(args.team1, args.team2, args.comp, topk=args.topk, use_elo=args.use_elo)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as e:
        log(f"Prediction failed: {e}", "CRITICAL")
