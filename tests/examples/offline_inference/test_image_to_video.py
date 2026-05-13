"""
Offline inference tests: image-to-video.
See examples/offline_inference/image_to_video/README.md
"""

from pathlib import Path

import pytest

from tests.examples.helpers import EXAMPLES, ExampleRunner, ReadmeSnippet
from tests.helpers.assertions import assert_video_valid
from tests.helpers.mark import hardware_marks

pytestmark = [
    pytest.mark.usefixtures("clean_gpu_memory_between_tests"),
    pytest.mark.full_model,
    pytest.mark.example,
    *hardware_marks(res={"cuda": "H100"}),
]

I2V_SCRIPT = EXAMPLES / "offline_inference" / "image_to_video" / "image_to_video.py"
README_PATH = I2V_SCRIPT.with_name("README.md")
EXAMPLE_OUTPUT_SUBFOLDER = "example_offline_i2v"

_SKIP_SECTIONS = {
    "Prerequisites",
    "Advanced Features",
    "FAQ",
}


def _skip_readme_snippet(language: str, code: str, h2_title: str) -> tuple[bool, str]:
    if h2_title in _SKIP_SECTIONS:
        return True, f"README section '{h2_title}' is intentionally excluded for examples tests"
    if language == "python":
        return True, "Python API snippets produce video files that ExampleRunner does not auto-collect"
    if "/path/to/" in code:
        return True, "Snippet references a placeholder local model path"
    return False, ""


README_SNIPPETS = ReadmeSnippet.extract_readme_snippets(README_PATH, skipif=_skip_readme_snippet)


@pytest.mark.parametrize("snippet", README_SNIPPETS, ids=lambda snippet: snippet.test_id)
def test_image_to_video(snippet: ReadmeSnippet, example_runner: ExampleRunner):
    should_skip, reason = snippet.skip
    if should_skip:
        pytest.skip(reason)

    result = example_runner.run(snippet, output_subfolder=Path(EXAMPLE_OUTPUT_SUBFOLDER))
    for asset in result.assets:
        assert_video_valid(asset)
