"""
Report Generator Module.
Produces publication-quality, 4-panel analytical session reports and charts
combining attention timelines, focus heatmaps, state distributions, and diagnostic breakdowns.
Ensures pixel-perfect grid alignment across all panels with dedicated header spacing.
"""

import os
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from config import DASHBOARD_DIR

class ReportGenerator:
    def __init__(self, output_dir=DASHBOARD_DIR):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate(self,
                 timeline_data: tuple,
                 heatmap_data: list,
                 stats: dict,
                 distraction_summary: dict,
                 show_window: bool = True) -> str:
        """
        Generates and saves the 4-panel post-session analytics report.
        Returns the absolute filepath of the saved PNG image.
        """
        timestamps, scores = timeline_data
        
        # Configure dark aesthetic
        plt.style.use('dark_background')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 11), dpi=130)
        fig.patch.set_facecolor('#181818')

        # Clean, Non-overlapping Header
        session_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        exp_label = stats.get("experiment_label", "Standard")

        fig.text(0.5, 0.955, "FocusAnalyser — Session Productivity & Attention Report",
                 fontsize=16, fontweight='bold', color='#ffffff', ha='center', va='center')
        fig.text(0.5, 0.925, f"Session Date: {session_time_str}   |   Experiment Label: [{exp_label}]",
                 fontsize=10, color='#a0a0a0', ha='center', va='center')

        # -----------------------------------------------------------------
        # Panel 1: Attention Timeline (Top-Left)
        # -----------------------------------------------------------------
        ax1.set_facecolor('#222222')
        for spine in ax1.spines.values():
            spine.set_color('#444444')
            spine.set_linewidth(1.0)

        if timestamps and scores:
            scores_pct = [s * 100 for s in scores]
            ax1.plot(timestamps, scores_pct, color='#3498db', linewidth=2.2, label='Attention (%)')
            
            # Threshold guideline fills
            ax1.axhline(60, color='#2ecc71', linestyle='--', alpha=0.7, label='Focused Threshold (60%)')
            ax1.axhline(38, color='#e74c3c', linestyle='--', alpha=0.7, label='Distraction Threshold (38%)')
            ax1.fill_between(timestamps, scores_pct, 60, where=np.array(scores_pct) >= 60,
                             color='#2ecc71', alpha=0.25, interpolate=True)
            ax1.fill_between(timestamps, scores_pct, 38, where=np.array(scores_pct) < 38,
                             color='#e74c3c', alpha=0.30, interpolate=True)
            
            ax1.set_ylim(0, 105)
            ax1.set_xlim(min(timestamps), max(timestamps) if max(timestamps) > 0 else 1)
        else:
            ax1.text(0.5, 0.5, "No temporal data captured", ha='center', va='center', color='#888888')

        ax1.set_title("1. Continuous Attention Timeline", fontsize=12, fontweight='bold', color='#ffffff', pad=10)
        ax1.set_xlabel("Elapsed Time (seconds)", fontsize=9.5, color='#cccccc')
        ax1.set_ylabel("Attention Score (%)", fontsize=9.5, color='#cccccc')
        ax1.grid(True, linestyle=':', alpha=0.35, color='#555555')
        ax1.legend(loc='lower left', fontsize=8, facecolor='#2b2b2b', edgecolor='#444444')

        # -----------------------------------------------------------------
        # Panel 2: Summary KPI Metrics Card (Top-Right)
        # -----------------------------------------------------------------
        ax2.set_facecolor('#222222')
        for spine in ax2.spines.values():
            spine.set_color('#444444')
            spine.set_linewidth(1.0)
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_title("2. Executive Performance Summary", fontsize=12, fontweight='bold', color='#ffffff', pad=10)

        overall_score = stats.get("attention_score", 0.0)
        total_time = stats.get("duration", 0.0)
        f_time = stats.get("focused_time", 0.0)
        d_time = stats.get("distracted_time", 0.0)
        dr_time = stats.get("drowsy_time", 0.0)
        avg_span = stats.get("avg_focus_span", 0.0)
        longest_streak = stats.get("longest_focus_span", 0.0)
        dist_count = stats.get("distraction_count", 0)
        blinks = stats.get("blink_count", 0)
        yawns = stats.get("yawn_count", 0)
        top_cause = distraction_summary.get("primary_distractor", "None")

        # Top score pill badge
        score_color = '#2ecc71' if overall_score >= 60 else ('#f39c12' if overall_score >= 38 else '#e74c3c')
        ax2.text(0.5, 0.90, f"OVERALL ATTENTION: {int(overall_score)}%", fontsize=11.5, fontweight='bold',
                 color=score_color, ha='center', va='center',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='#2b2b2b', edgecolor=score_color, linewidth=1.5))

        # Precisely aligned two-column metrics table
        metrics = [
            ("Total Duration", f"{total_time:.1f}s ({total_time/60.0:.1f} min)"),
            ("Focused Time", f"{f_time:.1f}s ({(f_time/max(0.1, total_time))*100:.1f}%)"),
            ("Distracted Time", f"{d_time:.1f}s ({(d_time/max(0.1, total_time))*100:.1f}%)"),
            ("Drowsy Time", f"{dr_time:.1f}s"),
            ("Average Focus Span", f"{avg_span:.1f} seconds"),
            ("Longest Unbroken Streak", f"{longest_streak:.1f} seconds"),
            ("Distraction Incidents", f"{dist_count} events"),
            ("Blink Count", f"{blinks} blinks"),
            ("Yawn Count", f"{yawns} yawns"),
            ("Primary Distractor", top_cause.replace('_', ' ').title()),
        ]

        y_start = 0.78
        y_step = 0.065
        for idx, (label, val) in enumerate(metrics):
            y_pos = y_start - (idx * y_step)
            ax2.text(0.08, y_pos, label, fontsize=9.0, color='#a0a0a0', va='center', ha='left')
            ax2.text(0.92, y_pos, val, fontsize=9.0, fontweight='bold', color='#ffffff', va='center', ha='right')
            if idx < len(metrics) - 1:
                ax2.axhline(y_pos - (y_step / 2.0), xmin=0.08, xmax=0.92, color='#333333', linewidth=0.5)

        # -----------------------------------------------------------------
        # Panel 3: Temporal Focus Heatmap (Bottom-Left)
        # -----------------------------------------------------------------
        ax3.set_facecolor('#222222')
        for spine in ax3.spines.values():
            spine.set_color('#444444')
            spine.set_linewidth(1.0)

        ax3.set_title("3. Focus Heatmap (🟢 High | 🟡 Moderate | 🔴 Low)", fontsize=12, fontweight='bold', color='#ffffff', pad=10)

        if heatmap_data:
            for seg in heatmap_data:
                start = seg["start"]
                width = max(0.5, seg["end"] - seg["start"])
                lbl = seg["label"]
                col = '#2ecc71' if lbl == "HIGH" else ('#f39c12' if lbl == "MODERATE" else '#e74c3c')
                ax3.barh(0, width, left=start, height=0.45, color=col, edgecolor='#181818', linewidth=0.6)

            total_h_time = heatmap_data[-1]["end"]
            ax3.set_xlim(0, max(1.0, total_h_time))
            ax3.set_ylim(-0.6, 0.6)
            ax3.set_yticks([])
            ax3.set_xlabel("Session Timeline (seconds)", fontsize=9.5, color='#cccccc')
        else:
            ax3.text(0.5, 0.5, "Insufficient segment duration for heatmap", ha='center', va='center', color='#888888')

        green_patch = mpatches.Patch(color='#2ecc71', label='High (>=70%)')
        amber_patch = mpatches.Patch(color='#f39c12', label='Moderate (40-69%)')
        red_patch = mpatches.Patch(color='#e74c3c', label='Low (<40%)')
        ax3.legend(handles=[green_patch, amber_patch, red_patch], loc='upper right', fontsize=8,
                   facecolor='#2b2b2b', edgecolor='#444444')

        # -----------------------------------------------------------------
        # Panel 4: Distraction Diagnostics / State Allocation (Bottom-Right)
        # -----------------------------------------------------------------
        ax4.set_facecolor('#222222')
        for spine in ax4.spines.values():
            spine.set_color('#444444')
            spine.set_linewidth(1.0)

        reason_counts = dict(distraction_summary.get("reason_counts", {}))

        if reason_counts:
            labels = [k.replace('_', ' ').title() for k in reason_counts.keys()]
            vals = list(reason_counts.values())
            y_pos = np.arange(len(labels))
            
            # Distinct colors for behavioral distraction causes
            bar_colors = []
            for lbl in labels:
                ll = lbl.lower()
                if "eye" in ll or "close" in ll:
                    bar_colors.append('#e74c3c')   # Red for eye closure
                elif "head" in ll:
                    bar_colors.append('#e67e22')   # Amber for head turn
                elif "gaze" in ll:
                    bar_colors.append('#f39c12')   # Yellow/amber for gaze
                elif "yawn" in ll:
                    bar_colors.append('#9b59b6')   # Purple for yawning
                elif "slouch" in ll or "posture" in ll:
                    bar_colors.append('#3498db')   # Cyan/blue for posture
                else:
                    bar_colors.append('#1abc9c')   # Teal for other

            bars = ax4.barh(y_pos, vals, color=bar_colors, edgecolor='#181818', height=0.55)
            ax4.set_yticks(y_pos)
            ax4.set_yticklabels(labels, fontsize=9.0, color='#ffffff')
            ax4.set_xlabel("Occurrences Count", fontsize=9.5, color='#cccccc')
            ax4.set_title("4. Contributing Distraction Causes", fontsize=12, fontweight='bold', color='#ffffff', pad=10)
            ax4.grid(True, linestyle=':', alpha=0.35, axis='x', color='#555555')
            
            for bar, val in zip(bars, vals):
                ax4.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height() / 2,
                         str(val), va='center', color='#ffffff', fontsize=8.5, fontweight='bold')
            ax4.set_xlim(0, max(vals) + 1.5 if vals else 5)
        else:
            labels = ['Focused', 'Distracted', 'Drowsy']
            sizes = [max(0.01, f_time), max(0.01, d_time), max(0.0, dr_time)]
            colors = ['#2ecc71', '#e74c3c', '#9b59b6']
            wedges, texts, autotexts = ax4.pie(
                sizes,
                labels=labels,
                autopct='%1.1f%%',
                colors=colors,
                startangle=140,
                textprops={'color': "#ffffff", 'fontsize': 9.5}
            )
            for at in autotexts:
                at.set_fontsize(8.5)
                at.set_fontweight('bold')
            ax4.set_title("4. Time State Allocation", fontsize=12, fontweight='bold', color='#ffffff', pad=10)

        # Generous vertical separation: subplots start at top=0.84, leaving top 16% for titles
        plt.subplots_adjust(left=0.07, right=0.95, top=0.84, bottom=0.08, wspace=0.22, hspace=0.32)

        # Save image to disk
        filename = f"focus_report_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=130, facecolor=fig.get_facecolor(), edgecolor='none')
        print(f"\n[FocusAnalyser] Report successfully saved to: {filepath}")

        if show_window:
            plt.show()

        plt.close(fig)
        return filepath
