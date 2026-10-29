"""
FocusAnalyser — Historical Session & Analytics Viewer
CLI utility to inspect stored sessions, export to CSV/Excel, and view detailed metrics.
"""

import sys
import os
import csv
import argparse
from database import SessionDB

def show_history(export_csv=False, detail_id=None):
    db = SessionDB()
    sessions = db.get_all_sessions()
    summary = db.get_analytics_summary()

    # Detail View for a specific session ID
    if detail_id is not None:
        matched = [s for s in sessions if s["id"] == detail_id]
        if not matched:
            print(f"[!] Session #{detail_id} not found in database.")
            return
        s = matched[0]
        print("=" * 65)
        print(f"  FocusAnalyser — Detailed Record for Session #{s['id']}")
        print("=" * 65)
        print(f"  • Date & Time:             {s['date']} at {s['start_time']}")
        print(f"  • Experiment Label:        {s['experiment_label']}")
        print(f"  • Overall Attention Score: {int(s['attention_score'])}%")
        print(f"  • Total Duration:          {s['duration']:.1f} seconds ({s['duration']/60.0:.1f} min)")
        print(f"  • Focused Time:            {s['focused_time']:.1f} seconds")
        print(f"  • Distracted Time:         {s['distracted_time']:.1f} seconds")
        print(f"  • Drowsy Time:             {s['drowsy_time']:.1f} seconds")
        print(f"  • Longest Focus Streak:    {s['longest_focus_span']:.1f} seconds")
        print(f"  • Average Focus Span:      {s['avg_focus_span']:.1f} seconds")
        print(f"  • Distraction Incidents:   {s['distraction_count']}")
        print(f"  • Blinks Recorded:         {s['blink_count']}")
        print(f"  • Yawns Recorded:          {s['yawn_count']}")
        print(f"  • Primary Distractor:      {s['primary_distractor']}")
        print("=" * 65)
        return

    # Export to CSV for Excel
    if export_csv:
        csv_path = os.path.join(os.path.dirname(__file__), "data", "sessions.csv")
        if sessions:
            keys = sessions[0].keys()
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(sessions)
            print(f"[OK] Successfully exported {len(sessions)} sessions to CSV:")
            print(f"     {csv_path}")
            print("     You can now open 'sessions.csv' directly in Microsoft Excel!")
        else:
            print("[!] No sessions to export.")
        return

    # Standard Summary Table View
    print("=" * 90)
    print("  FocusAnalyser — Historical Analytics & Session Log")
    print("=" * 90)
    print(f"  Total Sessions: {summary['total_sessions']}   |   "
          f"Avg Attention: {summary['avg_attention']}%   |   "
          f"Best Session: {summary['best_attention']}%   |   "
          f"Total Focus Time: {summary['total_focus_hours']} hrs")
    print("-" * 90)
    print(f"{'ID':<5} {'Date':<11} {'Start':<9} {'Duration':<10} {'Score':<8} {'Focus(s)':<9} {'Dist(s)':<8} {'Streak(s)':<10} {'Distractor':<16}")
    print("-" * 90)

    if not sessions:
        print("  No sessions recorded yet. Run 'python main.py' to record your first session!")
    else:
        for s in sessions:
            score_str = f"{int(s['attention_score'])}%"
            distractor = (s['primary_distractor'] or 'None').replace('_', ' ').title()
            print(f"#{s['id']:<4} {s['date']:<11} {s['start_time']:<9} {s['duration']:<10.1f} {score_str:<8} {s['focused_time']:<9.1f} {s['distracted_time']:<8.1f} {s['longest_focus_span']:<10.1f} {distractor:<16}")

    print("=" * 90)
    print("  TIP:")
    print("    • View full details of a session:  python history.py --detail <ID>  (e.g. python history.py --detail 20)")
    print("    • Open in Excel / export to CSV:   python history.py --csv")
    print("=" * 90)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View FocusAnalyser session history")
    parser.add_argument("--csv", action="store_true", help="Export database to data/sessions.csv for Excel")
    parser.add_argument("--detail", type=int, default=None, help="Show full metrics for a specific session ID")
    args = parser.parse_args()

    show_history(export_csv=args.csv, detail_id=args.detail)
