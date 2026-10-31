import json
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
import pytest

from docscan.src.editing import rotation_layout


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(shutil.which('node') is None, reason='Node.js is required for the JS/Python coordinate contract')
def test_preview_and_server_use_identical_rotation_coordinates():
    cases = [[640, 480, angle] for angle in [-180, -91, -45, 0, 1, 31, 90, 180]]
    cases += [[4001, 6001, -17], [1, 10, 90]]
    script = 'const G=require(process.argv[1]); console.log(JSON.stringify(JSON.parse(process.argv[2]).map(([w,h,a])=>G.rotationLayout(w,h,a))));'
    output = subprocess.run([shutil.which('node'), '-e', script, str(ROOT / 'docscan/web/editor-geometry.js'), json.dumps(cases)],
                            capture_output=True, text=True, check=True, timeout=10)
    previews = json.loads(output.stdout)
    for (width, height, angle), preview in zip(cases, previews):
        matrix, w, h = rotation_layout(width, height, angle)
        assert (w, h) == (preview['width'], preview['height'])
        np.testing.assert_allclose(
            [matrix[0, 0], matrix[1, 0], matrix[0, 1], matrix[1, 1], matrix[0, 2], matrix[1, 2]],
            preview['matrix'], atol=1e-9, rtol=0,
        )


def test_editor_markup_contains_every_referenced_control():
    html = (ROOT / 'docscan/web/index.html').read_text(encoding='utf-8')
    code = (ROOT / 'docscan/web/editor.js').read_text(encoding='utf-8')
    ids = re.findall(r'\bid="([^"]+)"', html)
    assert len(ids) == len(set(ids)), 'Duplicate HTML ids'
    references = set(re.findall(r"getElementById\('([^']+)'\)", code))
    assert references.issubset(ids), references - set(ids)
    assert html.index('src="/static/editor-geometry.js"') < html.index('src="/static/editor.js"') < html.index('new DocScanEditor')
