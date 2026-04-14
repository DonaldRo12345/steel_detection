"""
PDF Report Generator
Converts Markdown reports to PDF using markdown and weasyprint or pandoc.
"""

import argparse
import logging
from pathlib import Path
import subprocess
import sys

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def convert_with_pandoc(md_file: str, pdf_file: str) -> bool:
    """
    Convert Markdown to PDF using Pandoc.
    
    Args:
        md_file: Input Markdown file
        pdf_file: Output PDF file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Check if pandoc is installed
        result = subprocess.run(['pandoc', '--version'], 
                              capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error("Pandoc not found. Install from: https://pandoc.org/installing.html")
            return False
        
        logger.info(f"Using Pandoc: {result.stdout.split()[1]}")
        
        # Convert to PDF
        cmd = [
            'pandoc',
            md_file,
            '-o', pdf_file,
            '--pdf-engine=xelatex',  # or pdflatex, lualatex
            '-V', 'geometry:margin=1in',
            '-V', 'fontsize=11pt',
            '-V', 'documentclass=article',
            '--toc',  # Table of contents
            '--number-sections',
        ]
        
        logger.info(f"Converting {md_file} to PDF...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"✓ PDF created: {pdf_file}")
            return True
        else:
            logger.error(f"Pandoc error: {result.stderr}")
            return False
            
    except FileNotFoundError:
        logger.error("Pandoc not found. Please install Pandoc.")
        return False
    except Exception as e:
        logger.error(f"Error: {e}")
        return False


def convert_with_markdown_pdf(md_file: str, pdf_file: str) -> bool:
    """
    Convert Markdown to PDF using markdown and weasyprint.
    
    Args:
        md_file: Input Markdown file
        pdf_file: Output PDF file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        import markdown
        from weasyprint import HTML, CSS
        
        # Read Markdown
        with open(md_file, 'r', encoding='utf-8') as f:
            md_text = f.read()
        
        # Convert to HTML
        html = markdown.markdown(md_text, extensions=['tables', 'fenced_code', 'toc'])
        
        # Add basic styling
        styled_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 2cm;
                    font-size: 11pt;
                }}
                h1 {{
                    color: #2c3e50;
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 0.3em;
                }}
                h2 {{
                    color: #34495e;
                    border-bottom: 1px solid #bdc3c7;
                    padding-bottom: 0.2em;
                }}
                code {{
                    background-color: #f4f4f4;
                    padding: 2px 5px;
                    border-radius: 3px;
                }}
                pre {{
                    background-color: #f4f4f4;
                    padding: 10px;
                    border-radius: 5px;
                    overflow-x: auto;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin: 1em 0;
                }}
                table, th, td {{
                    border: 1px solid #ddd;
                }}
                th, td {{
                    padding: 8px;
                    text-align: left;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
            </style>
        </head>
        <body>
            {html}
        </body>
        </html>
        """
        
        # Convert to PDF
        logger.info(f"Converting {md_file} to PDF...")
        HTML(string=styled_html).write_pdf(pdf_file)
        logger.info(f"✓ PDF created: {pdf_file}")
        return True
        
    except ImportError as e:
        logger.error(f"Required library not found: {e}")
        logger.info("Install with: pip install markdown weasyprint")
        return False
    except Exception as e:
        logger.error(f"Error: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Convert Markdown reports to PDF')
    parser.add_argument('--md', type=str, required=True,
                       help='Input Markdown file')
    parser.add_argument('--out', type=str, default=None,
                       help='Output PDF file (default: same name as input)')
    parser.add_argument('--method', type=str, default='pandoc',
                       choices=['pandoc', 'weasyprint'],
                       help='Conversion method')
    
    args = parser.parse_args()
    
    # Resolve paths
    md_file = Path(args.md)
    if not md_file.exists():
        logger.error(f"Markdown file not found: {md_file}")
        sys.exit(1)
    
    if args.out:
        pdf_file = Path(args.out)
    else:
        pdf_file = md_file.with_suffix('.pdf')
    
    # Create output directory if needed
    pdf_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert
    if args.method == 'pandoc':
        success = convert_with_pandoc(str(md_file), str(pdf_file))
    else:
        success = convert_with_markdown_pdf(str(md_file), str(pdf_file))
    
    if not success:
        logger.error("PDF generation failed")
        sys.exit(1)


if __name__ == '__main__':
    main()
