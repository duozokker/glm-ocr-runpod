SINGLE_OCR_PROMPT = """Extract every visible piece of text from this page as faithfully as possible.
Do not omit titles, subtitles, headers, footers, labels, captions, footnotes, notes, side text, or text near or around tables.
Preserve the document structure and formatting semantics whenever they are visually identifiable:
- titles and section headings
- subtitles
- paragraphs
- ordered and unordered lists
- bold text
- italic text
- code blocks
- table content
Respond in Markdown as much as possible.
Use standard Markdown for headings, paragraphs, lists, emphasis, strong text, and code blocks whenever Markdown can represent them cleanly.
Use HTML only when Markdown cannot preserve the layout well enough.
Render tables with HTML table tags instead of Markdown tables.
If text is visually bold or italic, preserve that with Markdown when possible, otherwise use HTML.
Keep text in reading order and keep nearby labels with the content they belong to.
Return only Markdown and inline HTML with no explanation."""
