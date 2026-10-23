"""
DefectVision - Background Negative Candidate Interactive Review Tool
Offline Dent Pilot Step 3D

Self-contained local server serving an interactive browser interface
for human evaluation of candidate negative patches.
Automatically updates blank_reviewer_decisions.csv in real time.
"""

import sys
import os
import csv
import json
import shutil
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse

# Paths
BASE_DIR = Path(r"D:\DefectVision\experiments\dent_classification")
FILTERED_DIR = BASE_DIR / "background_audit" / "filtered_review"
BG_AUDIT_DIR = BASE_DIR / "background_audit"
CSV_PATH = FILTERED_DIR / "reviewer_decisions.csv"
TEMPLATE_PATH = FILTERED_DIR / "blank_reviewer_decisions.csv"
BACKUP_PATH = FILTERED_DIR / "reviewer_decisions.backup.csv"
CR7_SOURCE_IMG_DIR = Path(r"C:\Users\susha\Downloads\CR7-DET\CR7-DET\image")
CR7_LABEL_DIR = Path(r"C:\Users\susha\Downloads\CR7-DET\CR7-DET\label")

PORT = 8555


def load_dataset():
    src_path = CSV_PATH if CSV_PATH.exists() else TEMPLATE_PATH
    if not src_path.exists():
        raise FileNotFoundError(f"Source CSV not found at {src_path}")

    rows = []
    with open(src_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # Parse defect annotations for source image
            stem = r["source_image_stem"]
            lbl_file = CR7_LABEL_DIR / f"{stem}.txt"
            defects = []
            if lbl_file.exists():
                for line in lbl_file.read_text(encoding="utf-8").splitlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:5])
                        # convert normalized 0-1 to 400x400
                        x1 = (cx - w / 2.0) * 400.0
                        y1 = (cy - h / 2.0) * 400.0
                        x2 = (cx + w / 2.0) * 400.0
                        y2 = (cy + h / 2.0) * 400.0
                        defects.append({
                            "class_id": cls_id,
                            "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]
                        })

            r["defects"] = defects
            rows.append(r)
    return rows


DATASET = load_dataset()
FIELDNAMES = [
    "candidate_id", "split", "reviewer_decision", "reviewer_notes",
    "source_image_stem", "patch_x1", "patch_y1", "patch_x2", "patch_y2",
    "patch_width", "patch_height", "aspect_ratio", "min_defect_dist_px", "crop_path"
]


def save_decision(candidate_id, decision, notes):
    updated = False
    for r in DATASET:
        if r["candidate_id"] == candidate_id:
            r["reviewer_decision"] = decision
            if notes is not None:
                r["reviewer_notes"] = notes
            updated = True
            break

    if not updated:
        return False

    # Save to primary reviewer_decisions.csv
    with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for r in DATASET:
            row_dict = {k: r.get(k, "") for k in FIELDNAMES}
            writer.writerow(row_dict)

    # Also try to update template if not locked by Excel
    try:
        with open(TEMPLATE_PATH, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            for r in DATASET:
                row_dict = {k: r.get(k, "") for k in FIELDNAMES}
                writer.writerow(row_dict)
    except Exception:
        pass

    return True


def get_stats():
    total = len(DATASET)
    clean = sum(1 for r in DATASET if r.get("reviewer_decision", "").strip().lower() == "clean")
    defect = sum(1 for r in DATASET if r.get("reviewer_decision", "").strip().lower() == "defect_suspected")
    uncertain = sum(1 for r in DATASET if r.get("reviewer_decision", "").strip().lower() == "uncertain")
    unreviewed = sum(1 for r in DATASET if not r.get("reviewer_decision", "").strip())
    return {
        "total": total,
        "clean": clean,
        "defect_suspected": defect,
        "uncertain": uncertain,
        "unreviewed": unreviewed,
        "pct_done": round((clean + defect + uncertain) / max(total, 1) * 100, 1)
    }


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DefectVision — Background Negative Candidate Review</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --bg-main: #0b0f19;
    --bg-card: #151d2e;
    --bg-card-hover: #1c263c;
    --border: #23304a;
    --border-bright: #3b82f6;
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --color-clean: #10b981;
    --color-clean-bg: rgba(16, 185, 129, 0.15);
    --color-clean-hover: #059669;
    --color-defect: #ef4444;
    --color-defect-bg: rgba(239, 68, 68, 0.15);
    --color-defect-hover: #dc2626;
    --color-uncertain: #f59e0b;
    --color-uncertain-bg: rgba(245, 158, 11, 0.15);
    --color-uncertain-hover: #d97706;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background-color: var(--bg-main);
    color: var(--text-primary);
    font-family: 'Inter', sans-serif;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    overflow-x: hidden;
  }

  header {
    background: #0f172a;
    border-bottom: 1px solid var(--border);
    padding: 14px 28px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    position: sticky;
    top: 0;
    z-index: 100;
  }

  .brand {
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .brand-logo {
    width: 32px;
    height: 32px;
    background: linear-gradient(135deg, #3b82f6, #8b5cf6);
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    color: white;
    font-size: 16px;
  }
  .brand-title {
    font-size: 18px;
    font-weight: 700;
    letter-spacing: -0.02em;
  }
  .brand-subtitle {
    font-size: 12px;
    color: var(--text-secondary);
  }

  .header-actions {
    display: flex;
    align-items: center;
    gap: 16px;
  }

  .save-indicator {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 500;
    background: rgba(16, 185, 129, 0.1);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.3);
    transition: opacity 0.3s;
  }

  /* Progress Bar Container */
  .stats-bar {
    background: #111827;
    border-bottom: 1px solid var(--border);
    padding: 10px 28px;
    display: flex;
    align-items: center;
    gap: 20px;
    flex-wrap: wrap;
  }

  .progress-wrap {
    flex: 1;
    min-width: 220px;
  }
  .progress-meta {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    margin-bottom: 6px;
    color: var(--text-secondary);
  }
  .progress-track {
    width: 100%;
    height: 8px;
    background: #1f293d;
    border-radius: 4px;
    overflow: hidden;
    display: flex;
  }
  .prog-clean { background: var(--color-clean); height: 100%; transition: width 0.3s; }
  .prog-defect { background: var(--color-defect); height: 100%; transition: width 0.3s; }
  .prog-uncertain { background: var(--color-uncertain); height: 100%; transition: width 0.3s; }

  .stat-pill {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 6px;
    background: #1e293b;
    border: 1px solid var(--border);
  }
  .dot { width: 8px; height: 8px; border-radius: 50%; }
  .dot-clean { background: var(--color-clean); }
  .dot-defect { background: var(--color-defect); }
  .dot-uncertain { background: var(--color-uncertain); }
  .dot-unreviewed { background: #64748b; }

  /* Controls Bar */
  .filter-bar {
    padding: 8px 28px;
    background: #0f172a;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    flex-wrap: wrap;
  }

  .filter-group {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 13px;
  }
  .btn-filter {
    background: transparent;
    border: 1px solid var(--border);
    color: var(--text-secondary);
    padding: 4px 10px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 12px;
    font-weight: 500;
    transition: all 0.2s;
  }
  .btn-filter:hover {
    background: #1e293b;
    color: var(--text-primary);
  }
  .btn-filter.active {
    background: #3b82f6;
    border-color: #3b82f6;
    color: white;
  }

  .toggle-wrap {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
    color: var(--text-secondary);
    cursor: pointer;
  }
  .toggle-wrap input { cursor: pointer; }

  /* Main Workspace */
  main {
    flex: 1;
    padding: 24px 28px;
    display: grid;
    grid-template-columns: 1.15fr 0.85fr;
    gap: 24px;
    max-width: 1600px;
    width: 100%;
    margin: 0 auto;
  }

  @media (max-width: 1100px) {
    main { grid-template-columns: 1fr; }
  }

  .card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    display: flex;
    flex-direction: column;
  }

  .card-header {
    padding: 14px 18px;
    border-bottom: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: rgba(255,255,255,0.02);
  }

  .card-title {
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary);
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .badge {
    font-size: 11px;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 4px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  .badge-train { background: #0284c7; color: white; }
  .badge-val { background: #8b5cf6; color: white; }
  .badge-test { background: #ec4899; color: white; }

  .crop-viewer-box {
    padding: 24px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 380px;
    background: #080c14;
    position: relative;
  }

  .crop-image-wrapper {
    position: relative;
    border: 2px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
    background: #000;
    box-shadow: 0 8px 30px rgba(0,0,0,0.6);
  }

  #cropImg {
    display: block;
    max-width: 100%;
    height: auto;
    image-rendering: auto;
    transition: transform 0.2s ease;
  }
  #cropImg.pixelated {
    image-rendering: pixelated;
  }

  .crop-controls {
    margin-top: 14px;
    display: flex;
    gap: 12px;
    align-items: center;
  }

  .btn-sm {
    background: #1e293b;
    border: 1px solid var(--border);
    color: var(--text-secondary);
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 12px;
    cursor: pointer;
  }
  .btn-sm:hover {
    color: var(--text-primary);
    background: #283548;
  }

  /* Source Context Overlay */
  .source-viewer-box {
    padding: 20px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    background: #080c14;
    border-bottom: 1px solid var(--border);
  }

  .source-canvas-wrap {
    position: relative;
    width: 320px;
    height: 320px;
    border: 1px solid var(--border);
    border-radius: 6px;
    overflow: hidden;
  }
  #sourceCanvas {
    width: 320px;
    height: 320px;
    display: block;
  }

  /* Right column: Action / Decision Panel */
  .action-panel {
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .status-banner {
    padding: 16px;
    border-radius: 10px;
    border: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 14px;
    font-weight: 600;
    background: #1e293b;
    transition: all 0.2s;
  }
  .status-banner.is-clean {
    background: var(--color-clean-bg);
    border-color: var(--color-clean);
    color: #34d399;
  }
  .status-banner.is-defect {
    background: var(--color-defect-bg);
    border-color: var(--color-defect);
    color: #f87171;
  }
  .status-banner.is-uncertain {
    background: var(--color-uncertain-bg);
    border-color: var(--color-uncertain);
    color: #fbbf24;
  }

  .decision-buttons {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .btn-decision {
    padding: 18px 20px;
    border-radius: 10px;
    font-size: 16px;
    font-weight: 700;
    cursor: pointer;
    border: 2px solid transparent;
    display: flex;
    align-items: center;
    justify-content: space-between;
    transition: all 0.15s ease;
  }

  .btn-decision .key-hint {
    background: rgba(0,0,0,0.3);
    padding: 4px 10px;
    border-radius: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    font-weight: 500;
  }

  .btn-clean {
    background: var(--color-clean-bg);
    border-color: rgba(16, 185, 129, 0.4);
    color: #34d399;
  }
  .btn-clean:hover, .btn-clean.selected {
    background: var(--color-clean);
    border-color: #059669;
    color: #ffffff;
    box-shadow: 0 0 20px rgba(16, 185, 129, 0.4);
    transform: translateY(-1px);
  }

  .btn-defect {
    background: var(--color-defect-bg);
    border-color: rgba(239, 68, 68, 0.4);
    color: #f87171;
  }
  .btn-defect:hover, .btn-defect.selected {
    background: var(--color-defect);
    border-color: #dc2626;
    color: #ffffff;
    box-shadow: 0 0 20px rgba(239, 68, 68, 0.4);
    transform: translateY(-1px);
  }

  .btn-uncertain {
    background: var(--color-uncertain-bg);
    border-color: rgba(245, 158, 11, 0.4);
    color: #fbbf24;
  }
  .btn-uncertain:hover, .btn-uncertain.selected {
    background: var(--color-uncertain);
    border-color: #d97706;
    color: #ffffff;
    box-shadow: 0 0 20px rgba(245, 158, 11, 0.4);
    transform: translateY(-1px);
  }

  .notes-section {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 16px;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  .notes-section label {
    font-size: 12px;
    font-weight: 600;
    color: var(--text-secondary);
  }
  .notes-input {
    background: #0b0f19;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 10px 12px;
    color: var(--text-primary);
    font-family: inherit;
    font-size: 13px;
    outline: none;
    transition: border 0.2s;
  }
  .notes-input:focus {
    border-color: var(--border-bright);
  }

  .nav-buttons {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
  }
  .btn-nav {
    background: #1e293b;
    border: 1px solid var(--border);
    color: var(--text-primary);
    padding: 12px;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    transition: all 0.2s;
  }
  .btn-nav:hover {
    background: #283548;
    border-color: #3b82f6;
  }

  .meta-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    background: #0f172a;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px;
    font-size: 12px;
  }
  .meta-item {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }
  .meta-label { color: var(--text-muted); font-size: 11px; }
  .meta-val { color: var(--text-primary); font-family: 'JetBrains Mono', monospace; font-weight: 500; }

  .shortcut-card {
    background: rgba(255,255,255,0.02);
    border: 1px dashed var(--border);
    border-radius: 8px;
    padding: 12px 14px;
    font-size: 11px;
    color: var(--text-secondary);
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .shortcut-row {
    display: flex;
    justify-content: space-between;
  }
  kbd {
    background: #1e293b;
    border: 1px solid var(--border);
    padding: 1px 6px;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    color: var(--text-primary);
  }
</style>
</head>
<body>

<header>
  <div class="brand">
    <div class="brand-logo">DV</div>
    <div>
      <div class="brand-title">DefectVision Negative Candidate Review</div>
      <div class="brand-subtitle">Offline Dent Pilot Step 3D • Isolated Background Verification</div>
    </div>
  </div>
  <div class="header-actions">
    <div id="saveIndicator" class="save-indicator">
      <span>●</span> <span id="saveText">Synced to CSV</span>
    </div>
  </div>
</header>

<div class="stats-bar">
  <div class="progress-wrap">
    <div class="progress-meta">
      <span id="progressText">Progress: 0 / 285 reviewed (0.0%)</span>
      <span id="remainingText">285 unreviewed</span>
    </div>
    <div class="progress-track">
      <div id="progClean" class="prog-clean" style="width: 0%"></div>
      <div id="progDefect" class="prog-defect" style="width: 0%"></div>
      <div id="progUncertain" class="prog-uncertain" style="width: 0%"></div>
    </div>
  </div>

  <div class="stat-pill"><div class="dot dot-clean"></div> Clean: <span id="countClean">0</span></div>
  <div class="stat-pill"><div class="dot dot-defect"></div> Defect: <span id="countDefect">0</span></div>
  <div class="stat-pill"><div class="dot dot-uncertain"></div> Uncertain: <span id="countUncertain">0</span></div>
  <div class="stat-pill"><div class="dot dot-unreviewed"></div> Unreviewed: <span id="countUnreviewed">285</span></div>
</div>

<div class="filter-bar">
  <div class="filter-group">
    <span style="color: var(--text-muted); font-size: 12px;">Split:</span>
    <button class="btn-filter active" onclick="setSplitFilter('all')">All (285)</button>
    <button class="btn-filter" onclick="setSplitFilter('train')">Train (100)</button>
    <button class="btn-filter" onclick="setSplitFilter('val')">Val (85)</button>
    <button class="btn-filter" onclick="setSplitFilter('test')">Test (100)</button>
  </div>

  <div class="filter-group">
    <span style="color: var(--text-muted); font-size: 12px;">Status:</span>
    <button class="btn-filter active" id="fAll" onclick="setStatusFilter('all')">All</button>
    <button class="btn-filter" id="fUnreviewed" onclick="setStatusFilter('unreviewed')">Unreviewed Only</button>
    <button class="btn-filter" id="fClean" onclick="setStatusFilter('clean')">Clean</button>
    <button class="btn-filter" id="fDefect" onclick="setStatusFilter('defect')">Defect</button>
    <button class="btn-filter" id="fUncertain" onclick="setStatusFilter('uncertain')">Uncertain</button>
  </div>

  <label class="toggle-wrap">
    <input type="checkbox" id="autoAdvance" checked>
    <span>Auto-advance after click</span>
  </label>
</div>

<main>
  <!-- Left column: Visuals -->
  <div style="display: flex; flex-direction: column; gap: 20px;">
    <!-- Crop Card -->
    <div class="card">
      <div class="card-header">
        <div class="card-title">
          <span>Candidate Patch Crop</span>
          <span id="badgeSplit" class="badge badge-train">TRAIN</span>
          <span id="cropIdText" style="font-family: 'JetBrains Mono', monospace; font-size: 12px; color: var(--text-secondary);">bg_...</span>
        </div>
        <div style="font-size: 12px; color: var(--text-secondary);" id="patchDimText">131 × 115 px</div>
      </div>
      <div class="crop-viewer-box">
        <div class="crop-image-wrapper">
          <img id="cropImg" src="" alt="Candidate Patch Crop">
        </div>
        <div class="crop-controls">
          <button class="btn-sm" onclick="setZoom(1)">1×</button>
          <button class="btn-sm" onclick="setZoom(2)">2×</button>
          <button class="btn-sm" onclick="setZoom(3)">3×</button>
          <button class="btn-sm" onclick="togglePixelated()">Toggle Smoothing</button>
        </div>
      </div>
    </div>

    <!-- Source Context Card -->
    <div class="card">
      <div class="card-header">
        <div class="card-title">
          <span>Full Source Steel Image (400×400)</span>
          <span id="srcStemText" style="font-family: 'JetBrains Mono', monospace; font-size: 12px; color: var(--text-secondary);">dents100.jpg</span>
        </div>
        <div style="display: flex; gap: 10px; font-size: 11px;">
          <span style="color: #60a5fa;">■ Patch Box</span>
          <span style="color: #f87171;">■ Known Defect(s)</span>
          <span style="color: #64748b;">┄ 25px Margin</span>
        </div>
      </div>
      <div class="source-viewer-box">
        <div class="source-canvas-wrap">
          <canvas id="sourceCanvas" width="400" height="400"></canvas>
        </div>
      </div>
    </div>
  </div>

  <!-- Right column: Controls & Decision -->
  <div class="action-panel">
    <!-- Current Decision Banner -->
    <div id="statusBanner" class="status-banner">
      <span id="statusBannerText">Status: UNREVIEWED</span>
      <span id="currentIndexText" style="font-size: 12px; color: var(--text-secondary); font-family: 'JetBrains Mono', monospace;">1 / 285</span>
    </div>

    <!-- 3 Big Buttons -->
    <div class="decision-buttons">
      <button class="btn-decision btn-clean" id="btnClean" onclick="recordDecision('clean')">
        <span>✓ 1. Clean Negative Patch</span>
        <span class="key-hint">1</span>
      </button>

      <button class="btn-decision btn-defect" id="btnDefect" onclick="recordDecision('defect_suspected')">
        <span>⚠ 2. Defect Suspected</span>
        <span class="key-hint">2</span>
      </button>

      <button class="btn-decision btn-uncertain" id="btnUncertain" onclick="recordDecision('uncertain')">
        <span>? 3. Uncertain / Artifact</span>
        <span class="key-hint">3</span>
      </button>
    </div>

    <!-- Notes Field -->
    <div class="notes-section">
      <label for="reviewerNotes">Reviewer Notes (optional):</label>
      <input type="text" id="reviewerNotes" class="notes-input" placeholder="e.g. Faint roll groove, specular lighting seam, or borderline mark..." onchange="saveNotes()">
    </div>

    <!-- Navigation -->
    <div class="nav-buttons">
      <button class="btn-nav" onclick="prevItem()">
        <span>← Previous</span>
        <kbd>←</kbd>
      </button>
      <button class="btn-nav" onclick="nextItem()">
        <span>Next →</span>
        <kbd>→</kbd>
      </button>
    </div>

    <div style="display: flex; gap: 8px;">
      <button class="btn-nav" style="flex: 1; padding: 8px; font-size: 12px;" onclick="jumpNextUnreviewed()">
        ⚡ Jump to Next Unreviewed
      </button>
      <button class="btn-nav" style="flex: 1; padding: 8px; font-size: 12px; background: transparent;" onclick="clearDecision()">
        Clear Decision
      </button>
    </div>

    <!-- Metadata Details -->
    <div class="meta-grid">
      <div class="meta-item">
        <span class="meta-label">CANDIDATE ID</span>
        <span class="meta-val" id="metaId">-</span>
      </div>
      <div class="meta-item">
        <span class="meta-label">SOURCE IMAGE</span>
        <span class="meta-val" id="metaSrc">-</span>
      </div>
      <div class="meta-item">
        <span class="meta-label">PATCH COORDS [X1, Y1, X2, Y2]</span>
        <span class="meta-val" id="metaCoords">-</span>
      </div>
      <div class="meta-item">
        <span class="meta-label">NEAREST DEFECT DISTANCE</span>
        <span class="meta-val" id="metaDist">-</span>
      </div>
    </div>

    <!-- Keyboard Shortcuts -->
    <div class="shortcut-card">
      <div style="font-weight: 600; color: var(--text-primary); margin-bottom: 2px;">Keyboard Shortcuts</div>
      <div class="shortcut-row"><span>Mark Clean</span> <kbd>1</kbd></div>
      <div class="shortcut-row"><span>Mark Defect Suspected</span> <kbd>2</kbd></div>
      <div class="shortcut-row"><span>Mark Uncertain</span> <kbd>3</kbd></div>
      <div class="shortcut-row"><span>Previous / Next</span> <div><kbd>←</kbd> <kbd>→</kbd> or <kbd>A</kbd> <kbd>D</kbd></div></div>
      <div class="shortcut-row"><span>Next Unreviewed</span> <kbd>U</kbd></div>
      <div class="shortcut-row"><span>Clear Current Decision</span> <kbd>0</kbd> or <kbd>Backspace</kbd></div>
    </div>
  </div>
</main>

<script>
let items = [];
let currentIndex = 0;
let filteredIndices = [];
let currentSplit = 'all';
let currentStatus = 'all';
let zoomLevel = 1.8;
let isPixelated = false;

const cropImg = document.getElementById('cropImg');
const sourceCanvas = document.getElementById('sourceCanvas');
const ctx = sourceCanvas.getContext('2d');
const notesInput = document.getElementById('reviewerNotes');

async function init() {
  const res = await fetch('/api/items');
  const data = await res.json();
  items = data.items;
  updateStats(data.stats);
  applyFilters();
  renderCurrent();
}

function updateStats(stats) {
  document.getElementById('countClean').innerText = stats.clean;
  document.getElementById('countDefect').innerText = stats.defect_suspected;
  document.getElementById('countUncertain').innerText = stats.uncertain;
  document.getElementById('countUnreviewed').innerText = stats.unreviewed;
  
  document.getElementById('progressText').innerText = `Progress: ${stats.clean + stats.defect_suspected + stats.uncertain} / ${stats.total} reviewed (${stats.pct_done}%)`;
  document.getElementById('remainingText').innerText = `${stats.unreviewed} unreviewed`;
  
  document.getElementById('progClean').style.width = `${(stats.clean / stats.total) * 100}%`;
  document.getElementById('progDefect').style.width = `${(stats.defect_suspected / stats.total) * 100}%`;
  document.getElementById('progUncertain').style.width = `${(stats.uncertain / stats.total) * 100}%`;
}

function applyFilters() {
  filteredIndices = [];
  items.forEach((item, idx) => {
    if (currentSplit !== 'all' && item.split !== currentSplit) return;
    const dec = (item.reviewer_decision || '').trim().toLowerCase();
    if (currentStatus === 'unreviewed' && dec) return;
    if (currentStatus === 'clean' && dec !== 'clean') return;
    if (currentStatus === 'defect' && dec !== 'defect_suspected') return;
    if (currentStatus === 'uncertain' && dec !== 'uncertain') return;
    filteredIndices.push(idx);
  });

  if (filteredIndices.length === 0) {
    // empty filter fallback
    currentIndex = -1;
  } else if (!filteredIndices.includes(currentIndex)) {
    currentIndex = filteredIndices[0];
  }
}

function setSplitFilter(split) {
  currentSplit = split;
  document.querySelectorAll('.filter-group:first-child .btn-filter').forEach(btn => {
    btn.classList.toggle('active', btn.innerText.toLowerCase().startsWith(split));
  });
  applyFilters();
  renderCurrent();
}

function setStatusFilter(status) {
  currentStatus = status;
  document.querySelectorAll('#fAll, #fUnreviewed, #fClean, #fDefect, #fUncertain').forEach(btn => {
    btn.classList.remove('active');
  });
  if (status === 'all') document.getElementById('fAll').classList.add('active');
  if (status === 'unreviewed') document.getElementById('fUnreviewed').classList.add('active');
  if (status === 'clean') document.getElementById('fClean').classList.add('active');
  if (status === 'defect') document.getElementById('fDefect').classList.add('active');
  if (status === 'uncertain') document.getElementById('fUncertain').classList.add('active');
  applyFilters();
  renderCurrent();
}

function renderCurrent() {
  if (currentIndex < 0 || currentIndex >= items.length) {
    document.getElementById('statusBannerText').innerText = 'No items match current filters';
    cropImg.src = '';
    return;
  }

  const item = items[currentIndex];
  const posInFiltered = filteredIndices.indexOf(currentIndex) + 1;
  document.getElementById('currentIndexText').innerText = `${posInFiltered} of ${filteredIndices.length} (Global: #${currentIndex + 1} / ${items.length})`;

  // Split badge
  const badge = document.getElementById('badgeSplit');
  badge.className = `badge badge-${item.split}`;
  badge.innerText = item.split.toUpperCase();

  document.getElementById('cropIdText').innerText = item.candidate_id;
  document.getElementById('patchDimText').innerText = `${item.patch_width} × ${item.patch_height} px (AR: ${item.aspect_ratio})`;
  document.getElementById('srcStemText').innerText = `${item.source_image_stem}.jpg`;

  // Crop image
  cropImg.src = `/crop/${item.crop_path}?t=${Date.now()}`;
  updateZoom();

  // Metadata
  document.getElementById('metaId').innerText = item.candidate_id;
  document.getElementById('metaSrc').innerText = `${item.source_image_stem}.jpg`;
  document.getElementById('metaCoords').innerText = `[${item.patch_x1}, ${item.patch_y1}, ${item.patch_x2}, ${item.patch_y2}]`;
  document.getElementById('metaDist').innerText = `${item.min_defect_dist_px} px away`;

  // Notes
  notesInput.value = item.reviewer_notes || '';

  // Decision UI state
  const dec = (item.reviewer_decision || '').trim().toLowerCase();
  const banner = document.getElementById('statusBanner');
  banner.className = 'status-banner';
  
  document.getElementById('btnClean').classList.remove('selected');
  document.getElementById('btnDefect').classList.remove('selected');
  document.getElementById('btnUncertain').classList.remove('selected');

  if (dec === 'clean') {
    banner.classList.add('is-clean');
    document.getElementById('statusBannerText').innerText = '✓ DECISION: CLEAN NEGATIVE';
    document.getElementById('btnClean').classList.add('selected');
  } else if (dec === 'defect_suspected') {
    banner.classList.add('is-defect');
    document.getElementById('statusBannerText').innerText = '⚠ DECISION: DEFECT SUSPECTED';
    document.getElementById('btnDefect').classList.add('selected');
  } else if (dec === 'uncertain') {
    banner.classList.add('is-uncertain');
    document.getElementById('statusBannerText').innerText = '? DECISION: UNCERTAIN / ARTIFACT';
    document.getElementById('btnUncertain').classList.add('selected');
  } else {
    document.getElementById('statusBannerText').innerText = 'Status: UNREVIEWED';
  }

  // Draw source image context on canvas
  drawSourceContext(item);
}

function drawSourceContext(item) {
  const img = new Image();
  img.onload = () => {
    ctx.clearRect(0, 0, 400, 400);
    ctx.drawImage(img, 0, 0, 400, 400);

    // Draw 25px margin guide
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.4)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.strokeRect(25, 25, 350, 350);
    ctx.setLineDash([]);

    // Draw known defects in red
    if (item.defects && item.defects.length > 0) {
      ctx.lineWidth = 2;
      ctx.strokeStyle = '#ef4444';
      ctx.fillStyle = 'rgba(239, 68, 68, 0.2)';
      item.defects.forEach(d => {
        const [x1, y1, x2, y2] = d.box;
        ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      });
    }

    // Draw candidate patch box in vibrant blue
    const px1 = parseFloat(item.patch_x1);
    const py1 = parseFloat(item.patch_y1);
    const px2 = parseFloat(item.patch_x2);
    const py2 = parseFloat(item.patch_y2);

    ctx.lineWidth = 2.5;
    ctx.strokeStyle = '#3b82f6';
    ctx.fillStyle = 'rgba(59, 130, 246, 0.25)';
    ctx.fillRect(px1, py1, px2 - px1, py2 - py1);
    ctx.strokeRect(px1, py1, px2 - px1, py2 - py1);
  };
  img.src = `/source/${item.source_image_stem}`;
}

async function recordDecision(decision) {
  if (currentIndex < 0 || currentIndex >= items.length) return;
  const item = items[currentIndex];
  item.reviewer_decision = decision;
  const notes = notesInput.value.trim();
  item.reviewer_notes = notes;

  flashSaving();

  // Send to backend
  const res = await fetch('/api/save', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      candidate_id: item.candidate_id,
      decision: decision,
      notes: notes
    })
  });
  const data = await res.json();
  updateStats(data.stats);

  renderCurrent();

  // Auto-advance if checked
  if (document.getElementById('autoAdvance').checked) {
    nextItem();
  }
}

async function saveNotes() {
  if (currentIndex < 0 || currentIndex >= items.length) return;
  const item = items[currentIndex];
  item.reviewer_notes = notesInput.value.trim();

  flashSaving();
  const res = await fetch('/api/save', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      candidate_id: item.candidate_id,
      decision: item.reviewer_decision || '',
      notes: item.reviewer_notes
    })
  });
  const data = await res.json();
  updateStats(data.stats);
}

async function clearDecision() {
  if (currentIndex < 0 || currentIndex >= items.length) return;
  const item = items[currentIndex];
  item.reviewer_decision = '';

  flashSaving();
  const res = await fetch('/api/save', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      candidate_id: item.candidate_id,
      decision: '',
      notes: item.reviewer_notes || ''
    })
  });
  const data = await res.json();
  updateStats(data.stats);
  renderCurrent();
}

function flashSaving() {
  const el = document.getElementById('saveIndicator');
  const txt = document.getElementById('saveText');
  txt.innerText = 'Saving...';
  el.style.opacity = '1';
  setTimeout(() => {
    txt.innerText = 'Saved to CSV ✓';
  }, 180);
}

function nextItem() {
  if (filteredIndices.length === 0) return;
  const pos = filteredIndices.indexOf(currentIndex);
  if (pos < filteredIndices.length - 1) {
    currentIndex = filteredIndices[pos + 1];
    renderCurrent();
  }
}

function prevItem() {
  if (filteredIndices.length === 0) return;
  const pos = filteredIndices.indexOf(currentIndex);
  if (pos > 0) {
    currentIndex = filteredIndices[pos - 1];
    renderCurrent();
  }
}

function jumpNextUnreviewed() {
  const nextIdx = items.findIndex((it, idx) => idx > currentIndex && !it.reviewer_decision);
  if (nextIdx !== -1) {
    currentIndex = nextIdx;
    renderCurrent();
    return;
  }
  const firstUnreviewed = items.findIndex(it => !it.reviewer_decision);
  if (firstUnreviewed !== -1) {
    currentIndex = firstUnreviewed;
    renderCurrent();
  } else {
    alert('All 285 candidates have been reviewed!');
  }
}

function setZoom(lvl) {
  zoomLevel = lvl;
  updateZoom();
}

function updateZoom() {
  cropImg.style.transform = `scale(${zoomLevel})`;
  cropImg.style.transformOrigin = 'center center';
}

function togglePixelated() {
  isPixelated = !isPixelated;
  cropImg.classList.toggle('pixelated', isPixelated);
}

// Keyboard shortcuts
window.addEventListener('keydown', (e) => {
  // If typing in notes input, do not capture shortcuts
  if (document.activeElement === notesInput) {
    if (e.key === 'Enter') {
      notesInput.blur();
    }
    return;
  }

  if (e.key === '1') {
    recordDecision('clean');
  } else if (e.key === '2') {
    recordDecision('defect_suspected');
  } else if (e.key === '3') {
    recordDecision('uncertain');
  } else if (e.key === 'ArrowRight' || e.key === 'd' || e.key === 'D') {
    nextItem();
  } else if (e.key === 'ArrowLeft' || e.key === 'a' || e.key === 'A') {
    prevItem();
  } else if (e.key === 'u' || e.key === 'U') {
    jumpNextUnreviewed();
  } else if (e.key === '0' || e.key === 'Backspace') {
    clearDecision();
  }
});

init();
</script>
</body>
</html>
"""


class ReviewRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Quiet log to keep terminal clean
        pass

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
            return

        elif path == "/api/items":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            data = {
                "items": DATASET,
                "stats": get_stats()
            }
            self.wfile.write(json.dumps(data).encode("utf-8"))
            return

        elif path.startswith("/crop/"):
            rel_crop_path = path[len("/crop/"):]
            crop_file = BG_AUDIT_DIR / rel_crop_path
            if crop_file.exists() and crop_file.is_file():
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.end_headers()
                self.wfile.write(crop_file.read_bytes())
                return
            else:
                self.send_error(404, "Crop not found")
                return

        elif path.startswith("/source/"):
            stem = path[len("/source/"):]
            img_file = CR7_SOURCE_IMG_DIR / f"{stem}.jpg"
            if img_file.exists() and img_file.is_file():
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.end_headers()
                self.wfile.write(img_file.read_bytes())
                return
            else:
                self.send_error(404, "Source image not found")
                return

        self.send_error(404, "File not found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/save":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            data = json.loads(body)

            cand_id = data.get("candidate_id")
            decision = data.get("decision", "")
            notes = data.get("notes", "")

            ok = save_decision(cand_id, decision, notes)
            self.send_response(200 if ok else 400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "status": "ok" if ok else "error",
                "stats": get_stats()
            }).encode("utf-8"))
            return

        self.send_error(404)


def main():
    server = HTTPServer(("127.0.0.1", PORT), ReviewRequestHandler)
    print(f"===============================================================")
    print(f" DefectVision Background Negative Review Server Started")
    print(f" URL: http://localhost:{PORT}")
    print(f" Serving 285 candidates from {CSV_PATH.name}")
    print(f" Press Ctrl+C to stop.")
    print(f"===============================================================")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping review server.")
        server.server_close()


if __name__ == "__main__":
    main()
