import ctypes

import pytest


# Fails if there is any output to stdout or stderr.
# Based on https://github.com/TileDB-Inc/TileDB-Py/blob/6f02f34e087a9c1eb408cd7c61aeecb3817f9947/tiledb/tests/conftest.py#L25
@pytest.fixture(scope="function", autouse=True)
def no_output(capfd):
    # Wait for the test to finish.
    yield

    # Flush stdout.
    libc = ctypes.CDLL(None)
    libc.fflush(None)

    # Fail if there is any output.
    out, err = capfd.readouterr()
    if out or err:
        pytest.fail(
            f"Test failed because output was captured. out:\n{out}\nerr:\n{err}"
        )
