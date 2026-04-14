"""
Generate a styled HTML report from Markdown.
Open the output in any browser and use File > Print > Save as PDF.
"""
import argparse
from pathlib import Path
import markdown

STYLE = """
<style>
  :root { --blue: #2563eb; --dark: #1e293b; --light: #f8fafc; --border: #e2e8f0; }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', Arial, sans-serif; font-size: 11pt;
         line-height: 1.7; color: #1e293b; background: #fff;
         max-width: 900px; margin: 0 auto; padding: 2.5cm 2cm; }
  h1 { font-size: 22pt; color: var(--blue); border-bottom: 3px solid var(--blue);
       padding-bottom: 0.4em; margin: 1.2em 0 0.5em; }
  h2 { font-size: 15pt; color: var(--dark); border-bottom: 1px solid var(--border);
       padding-bottom: 0.25em; margin: 1.5em 0 0.5em; }
  h3 { font-size: 12pt; color: #334155; margin: 1.2em 0 0.4em; }
  h4 { font-size: 11pt; color: #475569; margin: 1em 0 0.3em; }
  p  { margin: 0.5em 0; }
  ul, ol { margin: 0.5em 0 0.5em 1.8em; }
  li { margin: 0.2em 0; }
  code { background: #f1f5f9; padding: 1px 5px; border-radius: 3px;
         font-family: 'Consolas', monospace; font-size: 9.5pt; color: #c0392b; }
  pre  { background: #f1f5f9; padding: 12px 16px; border-radius: 6px;
         border-left: 4px solid var(--blue); overflow-x: auto; margin: 0.8em 0; }
  pre code { color: #1e293b; background: none; padding: 0; }
  table { border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 10pt; }
  th { background: var(--blue); color: #fff; padding: 8px 12px; text-align: left; }
  td { padding: 7px 12px; border: 1px solid var(--border); }
  tr:nth-child(even) td { background: var(--light); }
  blockquote { border-left: 4px solid var(--blue); padding: 8px 16px;
               background: #eff6ff; margin: 0.8em 0; color: #1d4ed8; }
  hr { border: none; border-top: 2px solid var(--border); margin: 2em 0; }
  a  { color: var(--blue); }
  strong { color: #0f172a; }
  .cover { text-align: center; padding: 3em 0 2em; border-bottom: 3px solid var(--blue); margin-bottom: 2em; }
  .cover h1 { border: none; font-size: 26pt; }
  .cover p  { font-size: 12pt; color: #64748b; margin: 0.3em 0; }
  @media print {
    body { max-width: 100%; padding: 1.5cm; }
    pre  { white-space: pre-wrap; }
    .pagebreak { page-break-before: always; }
  }
</style>
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--md',  required=True, help='Input Markdown file')
    parser.add_argument('--out', default=None,  help='Output HTML file')
    args = parser.parse_args()

    md_path  = Path(args.md)
    out_path = Path(args.out) if args.out else md_path.with_suffix('.html')
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(md_path, encoding='utf-8') as f:
        text = f.read()

    # Strip YAML front-matter if present
    if text.startswith('---'):
        end = text.find('---', 3)
        if end != -1:
            text = text[end + 3:].lstrip()

    html_body = markdown.markdown(
        text,
        extensions=['tables', 'fenced_code', 'toc', 'nl2br']
    )

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Steel Defect Detection Report</title>
  {STYLE}
</head>
<body>
{html_body}
</body>
</html>"""

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(full_html)

    print(f"HTML report saved to: {out_path}")
    print("Open in browser and use File > Print > Save as PDF to get a PDF.")

if __name__ == '__main__':
    main()
