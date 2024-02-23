"""Created on Dec 2, 2013.

@author: jakeret
"""

import py
import subprocess


def test_linkcheck(tmpdir):
    doctrees = tmpdir.join("doctrees")
    htmldir = tmpdir.join("html")
    subprocess.check_call(
        ["sphinx-build", "-blinkcheck", "-d", str(doctrees), ".", str(htmldir)]
    )


def test_build_docs(tmpdir):
    doctrees = tmpdir.join("doctrees")
    htmldir = tmpdir.join("html")
    subprocess.check_call(
        ["sphinx-build", "-bhtml", "-d", str(doctrees), ".", str(htmldir)]
    )
