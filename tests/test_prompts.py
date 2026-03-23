from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prompts import SINGLE_OCR_PROMPT


def test_single_prompt_is_strong_and_structured():
    assert "Extract every visible piece of text" in SINGLE_OCR_PROMPT
    assert "Render tables with HTML table tags" in SINGLE_OCR_PROMPT
    assert "Return only Markdown and inline HTML" in SINGLE_OCR_PROMPT
