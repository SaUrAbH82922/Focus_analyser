"""
Markdown to PDF Converter using Microsoft Edge Headless.
Generates an academic-grade, beautifully formatted PDF report from README.md.
"""

import os
import re
import subprocess
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MD_PATH = os.path.join(BASE_DIR, "README.md")
HTML_PATH = os.path.join(BASE_DIR, "README_preview.html")
PDF_PATH = os.path.join(BASE_DIR, "FocusAnalyser_Documentation.pdf")
EDGE_PATH = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

def parse_markdown_to_html(md_text):
    """Simple and robust Markdown to HTML converter."""
    # Convert code blocks first
    code_blocks = []
    def save_code_block(match):
        lang = match.group(1) or ""
        code = match.group(2)
        # HTML escape
        code = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        code_blocks.append(f'<pre class="code-block"><code class="language-{lang}">{code}</code></pre>')
        return f"%%CODEBLOCK_{len(code_blocks)-1}%%"

    md = re.sub(r'```([a-zA-Z0-9_-]*)\n(.*?)```', save_code_block, md_text, flags=re.DOTALL)

    # Inline code
    md = re.sub(r'`([^`]+)`', r'<code class="inline-code">\1</code>', md)

    # Blockquotes
    md = re.sub(r'^>\s*(.*?)$', r'<blockquote>\1</blockquote>', md, flags=re.MULTILINE)

    # Horizontal rules
    md = re.sub(r'^---+$', r'<hr/>', md, flags=re.MULTILINE)

    # Headers
    md = re.sub(r'^######\s+(.*?)$', r'<h6>\1</h6>', md, flags=re.MULTILINE)
    md = re.sub(r'^#####\s+(.*?)$', r'<h5>\1</h5>', md, flags=re.MULTILINE)
    md = re.sub(r'^####\s+(.*?)$', r'<h4>\1</h4>', md, flags=re.MULTILINE)
    md = re.sub(r'^###\s+(.*?)$', r'<h3>\1</h3>', md, flags=re.MULTILINE)
    md = re.sub(r'^##\s+(.*?)$', r'<h2>\1</h2>', md, flags=re.MULTILINE)
    md = re.sub(r'^#\s+(.*?)$', r'<h1>\1</h1>', md, flags=re.MULTILINE)

    # Bold and Italic
    md = re.sub(r'\*\*\*(.*?)\*\*\*', r'<strong><em>\1</em></strong>', md)
    md = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', md)
    md = re.sub(r'\*(.*?)\*', r'<em>\1</em>', md)

    # Tables
    def parse_table(match):
        lines = [l.strip() for l in match.group(0).strip().split('\n') if l.strip()]
        if len(lines) < 2:
            return match.group(0)
        
        headers = [c.strip() for c in lines[0].strip('|').split('|')]
        # line 1 is separator
        html = '<div class="table-container"><table><thead><tr>'
        for h in headers:
            html += f'<th>{h}</th>'
        html += '</tr></thead><tbody>'
        
        for row_line in lines[2:]:
            cells = [c.strip() for c in row_line.strip('|').split('|')]
            html += '<tr>'
            for c in cells:
                html += f'<td>{c}</td>'
            html += '</tr>'
        html += '</tbody></table></div>'
        return html

    table_pattern = re.compile(r'(\|[^\n]+\|\n\|[-:| ]+\|\n(?:\|[^\n]+\|\n?)+)', re.MULTILINE)
    md = re.sub(table_pattern, parse_table, md)

    # Lists
    lines = md.split('\n')
    in_ul = False
    in_ol = False
    new_lines = []

    for line in lines:
        ul_match = re.match(r'^\s*[-*]\s+(.*)$', line)
        ol_match = re.match(r'^\s*(\d+)\.\s+(.*)$', line)

        if ul_match:
            if not in_ul:
                new_lines.append('<ul>')
                in_ul = True
            new_lines.append(f'<li>{ul_match.group(1)}</li>')
        elif ol_match:
            if not in_ol:
                new_lines.append('<ol>')
                in_ol = True
            new_lines.append(f'<li>{ol_match.group(2)}</li>')
        else:
            if in_ul:
                new_lines.append('</ul>')
                in_ul = False
            if in_ol:
                new_lines.append('</ol>')
                in_ol = False
            
            # Paragraph wrapper if line is not tag or empty
            l_strip = line.strip()
            if l_strip and not l_strip.startswith('<') and not l_strip.startswith('%%CODEBLOCK'):
                new_lines.append(f'<p>{line}</p>')
            else:
                new_lines.append(line)

    if in_ul:
        new_lines.append('</ul>')
    if in_ol:
        new_lines.append('</ol>')

    html_body = '\n'.join(new_lines)

    # Restore code blocks
    for i, cb in enumerate(code_blocks):
        html_body = html_body.replace(f"%%CODEBLOCK_{i}%%", cb)

    return html_body

def build_full_html(body_html):
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>FocusAnalyser: System Documentation</title>
    <!-- KaTeX for rendering LaTeX math formulas -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"
            onload="renderMathInElement(document.body, {{
                delimiters: [
                    {{left: '$$', right: '$$', display: true}},
                    {{left: '$', right: '$', display: false}}
                ]
            }});"></script>
    <style>
        @page {{
            size: A4;
            margin: 20mm 16mm 20mm 16mm;
            @bottom-right {{
                content: counter(page);
            }}
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
            font-size: 13.5px;
            line-height: 1.6;
            color: #24292e;
            background-color: #ffffff;
            margin: 0;
            padding: 20px;
        }}
        h1 {{
            font-size: 26px;
            border-bottom: 2px solid #0366d6;
            padding-bottom: 8px;
            margin-top: 10px;
            margin-bottom: 16px;
            color: #0366d6;
            page-break-before: auto;
        }}
        h2 {{
            font-size: 20px;
            border-bottom: 1px solid #eaecef;
            padding-bottom: 6px;
            margin-top: 28px;
            margin-bottom: 14px;
            color: #1a1f2c;
            page-break-after: avoid;
        }}
        h3 {{
            font-size: 16px;
            margin-top: 20px;
            margin-bottom: 10px;
            color: #2b3a4a;
            page-break-after: avoid;
        }}
        h4, h5, h6 {{
            font-size: 14px;
            margin-top: 16px;
            margin-bottom: 8px;
            page-break-after: avoid;
        }}
        p {{
            margin-top: 0;
            margin-bottom: 12px;
        }}
        blockquote {{
            margin: 12px 0;
            padding: 10px 16px;
            color: #4a5568;
            background-color: #f7fafc;
            border-left: 4px solid #3182ce;
            border-radius: 3px;
        }}
        .code-block {{
            background-color: #f6f8fa;
            border: 1px solid #e1e4e8;
            border-radius: 6px;
            padding: 12px 14px;
            overflow-x: auto;
            font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace;
            font-size: 11.5px;
            line-height: 1.45;
            page-break-inside: avoid;
            margin-bottom: 14px;
        }}
        .inline-code {{
            background-color: #f0f3f6;
            padding: 2px 5px;
            border-radius: 3px;
            font-family: "SFMono-Regular", Consolas, monospace;
            font-size: 12px;
            color: #d73a49;
        }}
        .table-container {{
            margin: 14px 0;
            overflow-x: auto;
            page-break-inside: avoid;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin-bottom: 12px;
        }}
        th, td {{
            border: 1px solid #d1d5db;
            padding: 8px 12px;
            text-align: left;
            font-size: 12.5px;
        }}
        th {{
            background-color: #f3f4f6;
            font-weight: 600;
            color: #111827;
        }}
        tr:nth-child(even) {{
            background-color: #f9fafb;
        }}
        ul, ol {{
            margin-top: 0;
            margin-bottom: 12px;
            padding-left: 24px;
        }}
        li {{
            margin-bottom: 4px;
        }}
        hr {{
            border: 0;
            height: 1px;
            background: #e5e7eb;
            margin: 24px 0;
        }}
    </style>
</head>
<body>
{body_html}
</body>
</html>
"""

def convert():
    print("[*] Reading README.md...")
    with open(MD_PATH, "r", encoding="utf-8") as f:
        md_text = f.read()

    print("[*] Converting Markdown & LaTeX math to HTML...")
    body_html = parse_markdown_to_html(md_text)
    full_html = build_full_html(body_html)

    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"[*] HTML preview written to: {HTML_PATH}")

    print(f"[*] Invoking Microsoft Edge headless renderer...")
    # Convert file path to file:// URL
    file_url = "file:///" + HTML_PATH.replace("\\", "/")

    cmd = [
        EDGE_PATH,
        "--headless",
        "--disable-gpu",
        "--virtual-time-budget=3500",  # Allow KaTeX 3.5s to render equations
        f"--print-to-pdf={PDF_PATH}",
        "--no-pdf-header-footer",
        file_url
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if os.path.exists(PDF_PATH) and os.path.getsize(PDF_PATH) > 1000:
        pdf_size_kb = os.path.getsize(PDF_PATH) / 1024.0
        print(f"[SUCCESS] PDF generated successfully!")
        print(f"Location: {PDF_PATH}")
        print(f"Size:     {pdf_size_kb:.1f} KB")
    else:
        print(f"[!] PDF generation failed: {result.stderr}")

if __name__ == "__main__":
    convert()
