import json

with open("presentation.md", "r", encoding="utf-8") as f:
    md_content = f.read()

html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI for Palestine Smart Library - Final Presentation</title>
    <!-- Marked.js for Markdown parsing -->
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <!-- Mermaid.js for diagrams -->
    <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
        mermaid.initialize({{ startOnLoad: false, theme: 'default' }});
        
        document.addEventListener('DOMContentLoaded', async () => {{
            const markdownText = {json.dumps(md_content)};
            const contentDiv = document.getElementById('content');
            
            // Parse Markdown to HTML
            contentDiv.innerHTML = marked.parse(markdownText);
            
            // Find all mermaid code blocks and convert them to mermaid divs
            const mermaidBlocks = document.querySelectorAll('code.language-mermaid');
            mermaidBlocks.forEach(block => {{
                const pre = block.parentElement;
                const div = document.createElement('div');
                div.className = 'mermaid';
                div.textContent = block.textContent;
                pre.parentNode.replaceChild(div, pre);
            }});
            
            // Render Mermaid diagrams
            await mermaid.run({{
                querySelector: '.mermaid',
            }});
        }});
    </script>
    <style>
        body {{
            font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 900px;
            margin: 0 auto;
            padding: 40px 20px;
            background-color: #ffffff;
        }}
        h1, h2, h3 {{
            color: #1a1a4e;
            margin-top: 1.5em;
            page-break-after: avoid;
        }}
        h1 {{ border-bottom: 3px solid #667eea; padding-bottom: 10px; }}
        h2 {{ border-bottom: 2px solid #667eea; padding-bottom: 5px; }}
        pre {{
            background: #f4f4f4;
            padding: 15px;
            border-radius: 8px;
            overflow-x: auto;
        }}
        code {{
            background: #f4f4f4;
            padding: 2px 5px;
            border-radius: 4px;
            color: #d11141;
        }}
        pre code {{
            background: transparent;
            color: inherit;
        }}
        hr {{
            border: 0;
            height: 2px;
            background: #e8e8e8;
            margin: 60px 0;
            page-break-before: always; /* Creates a new page for each slide when printing */
        }}
        .mermaid {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            display: flex;
            justify-content: center;
        }}
        a {{ color: #667eea; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        
        /* Print Styles for PDF export */
        @media print {{
            body {{
                background-color: white;
                padding: 0;
                margin: 0;
            }}
            hr {{
                display: none; /* Hide HR when printing to PDF, page-break handles it */
            }}
            .slide-break {{
                page-break-before: always;
            }}
        }}
    </style>
</head>
<body>
    <div id="content"></div>
</body>
</html>
"""

with open("presentation.html", "w", encoding="utf-8") as f:
    f.write(html_template)

print("HTML export successful!")
