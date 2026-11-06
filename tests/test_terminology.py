"""Tests to enforce strict terminology rules.

제약 사항:
- 문서 및 코드 전반에서 용어를 Train/Validation/Test로 통일한다.
- Dev, Val 등 비표준 축약어 혼용 금지.
"""

import os
import re
import pytest

FORBIDDEN_TERMS = [
    r"\bDev\b",
    r"\bdev\b",
    r"\bVal_set\b",
    r"\bdev_set\b",
]


def test_no_forbidden_terminology_in_reports():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    violations = []
    found_reports_dir = False
    for root, _, files in os.walk(repo_root):
        if os.path.basename(root) != "reports":
            continue
        found_reports_dir = True
        for f in files:
            if f.endswith(".md"):
                file_path = os.path.join(root, f)
                with open(file_path, "r", encoding="utf-8") as fp:
                    content = fp.read()
                    for pattern in FORBIDDEN_TERMS:
                        matches = re.findall(pattern, content)
                        if matches:
                            violations.append(f"{f}: 발견된 비표준 용어 {matches}")

    if not found_reports_dir:
        pytest.skip("reports 디렉토리가 아직 생성되지 않았습니다.")

    assert len(violations) == 0, f"용어 통일성 위반 발견: {violations}"
