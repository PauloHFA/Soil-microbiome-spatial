# -*- coding: utf-8 -*-
"""Prepare docs/ directory for GitHub Pages deployment."""
from pathlib import Path

docs_dir = Path("docs")
docs_dir.mkdir(parents=True, exist_ok=True)

# 1. Update docs/index.html
index_path = docs_dir / "index.html"
if index_path.exists():
    content = index_path.read_text(encoding="utf-8")
    target_idx = """      <div class="meta-item"><strong>Validação:</strong> <span class="badge-success">Spatial Group K-Fold (Zero Leakage)</span></div>
    </div>"""

    nav_bar = """      <div class="meta-item"><strong>Validação:</strong> <span class="badge-success">Spatial Group K-Fold (Zero Leakage)</span></div>
    </div>
    <div class="nav-actions" style="margin-top: 22px; display: flex; flex-wrap: wrap; gap: 12px;">
      <a href="dashboard.html" style="background:#2563eb; color:#ffffff; text-decoration:none; padding:10px 18px; border-radius:6px; font-weight:600; font-size:14px; display:inline-flex; align-items:center; gap:8px; box-shadow:0 2px 4px rgba(0,0,0,0.2);">📊 Dashboard Analítico Interativo</a>
      <a href="RELATORIO_TECNICO_CIENTIFICO.pdf" target="_blank" style="background:#334155; color:#ffffff; text-decoration:none; padding:10px 18px; border-radius:6px; font-weight:600; font-size:14px; display:inline-flex; align-items:center; gap:8px;">📥 Baixar Relatório em PDF</a>
      <a href="tabelas_manuscrito.xlsx" download style="background:#166534; color:#ffffff; text-decoration:none; padding:10px 18px; border-radius:6px; font-weight:600; font-size:14px; display:inline-flex; align-items:center; gap:8px;">📗 Baixar Tabelas (Excel)</a>
      <a href="https://github.com/PauloHFA/Soil-microbiome-spatial" target="_blank" style="background:#1e293b; color:#94a3b8; text-decoration:none; padding:10px 18px; border-radius:6px; font-weight:600; font-size:14px; border:1px solid #475569; display:inline-flex; align-items:center; gap:8px;">💻 Código no GitHub</a>
    </div>"""

    if target_idx in content:
        content = content.replace(target_idx, nav_bar, 1)
        index_path.write_text(content, encoding="utf-8")
        (docs_dir / "RELATORIO_TECNICO_CIENTIFICO.html").write_text(content, encoding="utf-8")
        print("Updated docs/index.html and docs/RELATORIO_TECNICO_CIENTIFICO.html")
    else:
        print("Target not found in docs/index.html")

# 2. Update docs/dashboard.html
dash_path = docs_dir / "dashboard.html"
if dash_path.exists():
    content = dash_path.read_text(encoding="utf-8")
    target_dash = """<header>
    <a href="RELATORIO_TECNICO_CIENTIFICO.html" style="float:right; background:#1e3a5f; color:#ffffff; text-decoration:none; padding:9px 18px; border-radius:6px; font-weight:600; font-size:13px; box-shadow:0 1px 3px rgba(0,0,0,0.1);">📄 Relatório Técnico-Científico →</a>"""

    dash_nav = """<header>
    <div style="float:right; display:flex; gap:10px; flex-wrap:wrap; margin-bottom:10px;">
      <a href="index.html" style="background:#1e3a5f; color:#ffffff; text-decoration:none; padding:8px 14px; border-radius:6px; font-weight:600; font-size:13px; box-shadow:0 1px 3px rgba(0,0,0,0.1);">📄 Relatório Completo</a>
      <a href="RELATORIO_TECNICO_CIENTIFICO.pdf" target="_blank" style="background:#475569; color:#ffffff; text-decoration:none; padding:8px 14px; border-radius:6px; font-weight:600; font-size:13px;">📥 Baixar PDF</a>
      <a href="https://github.com/PauloHFA/Soil-microbiome-spatial" target="_blank" style="background:#ffffff; color:#0f172a; text-decoration:none; padding:8px 14px; border-radius:6px; font-weight:600; font-size:13px; border:1px solid #cbd5e1;">💻 GitHub</a>
    </div>"""

    if target_dash in content:
        content = content.replace(target_dash, dash_nav, 1)
        dash_path.write_text(content, encoding="utf-8")
        print("Updated docs/dashboard.html")
    else:
        print("Target not found in docs/dashboard.html")

# 3. Create .nojekyll
(docs_dir / ".nojekyll").write_text("", encoding="utf-8")
print("Created docs/.nojekyll")
