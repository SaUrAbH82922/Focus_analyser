"""
FocusAnalyser — Historical Session & Analytics Viewer
CLI utility to inspect stored sessions, compare experiment runs, and track productivity trends.
"""

from database import SessionDB

def show_history():
    db = SessionDB()
    sessions = db.get_all_sessions()
    summary = db.get_analytics_summary()

    print("=" * 85)
    print("  FocusAnalyser — Historical Analytics & Session Log")
    print("=" * 85)
    print(f"  Total Sessions: {summary['total_sessions']}  |  "
          f"Avg Attention: {summary['avg_attention']}%  |  "
          f"Best Session: {summary['best_attention']}%  |  "
          f"Total Focus: {summary['total_focus_hours']} hrs")
    print("-" * 85)
    print(f"{'ID':<4} {'Date':<11} {'Start':<9} {'Duration':<9} {'Score':<7} {'Focus(s)':<9} {'Dist(s)':<8} {'Streak(s)':<10} {'Label':<15}")
    print("-" * 85)

    if not sessions:
        print("  No sessions recorded yet. Run 'python main.py' to record your first session!")
    else:
        for s in sessions:
            print(f"#{s['id']:<3} {s['date']:<11} {s['start_time']:<9} {s['duration']:<9.1f} {s['attention_score']:<6.0f}% {s['focused_time']:<9.1f} {s['distracted_time']:<8.1f} {s['longest_focus_span']:<10.1f} {s['experiment_label']:<15}")

    print("=" * 85)

if __name__ == "__main__":
    show_history()
