from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import requests

from orbitdet.data.kernel import KernelEntry, KernelManager


def _mock_response(
    content: bytes | None = None, text: str = "", status_code: int = 200
) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.content = content or b""
    resp.text = text
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(response=resp)
    return resp


FAKE_KERNEL_BYTES = b"DAF/SPK\x00" + b"\x00" * 120  # plausible-ish binary header


class TestKernelManager:
    def test_kernel_entry_stores_url_and_name(self) -> None:
        entry = KernelEntry("https://example.com/kernel.tpc", "kernel.tpc")

        assert entry.url == "https://example.com/kernel.tpc"
        assert entry.name == "kernel.tpc"

    def test_init_raises_when_kernel_folder_missing(self, tmp_path: Path) -> None:
        missing_path = tmp_path / "missing"
        cfg = SimpleNamespace(kernel_folder=str(missing_path))

        with pytest.raises(FileNotFoundError, match="Kernel path"):
            KernelManager(cfg)

    def test_downloads_kernel_when_missing(self, tmp_path: Path) -> None:
        mock_response = _mock_response(content=FAKE_KERNEL_BYTES)
        kernel_url = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/pck00010.tpc"
        kernel_name = "pck00010.tpc"

        with patch("requests.get", return_value=mock_response) as mock_get:
            km = KernelManager.__new__(KernelManager)  # bypass __init__
            km._fetch(kernel_url, kernel_name, dest=tmp_path)

        mock_get.assert_called_once()
        assert (tmp_path / kernel_name).exists()
        assert (tmp_path / kernel_name).read_bytes() == FAKE_KERNEL_BYTES

    def test_download_all_kernels_calls_fetch_for_each_kernel(self, tmp_path: Path) -> None:
        cfg = SimpleNamespace(
            kernel_folder=str(tmp_path),
            kernels={
                "pck00010.tpc": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/pck00010.tpc",
                "naif0012.tls": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls",
            },
            data_folder=str(tmp_path),
        )

        km = KernelManager(cfg)

        with patch.object(km, "_fetch") as mock_fetch:
            km.download_all_kernels()

        mock_fetch.assert_any_call(
            "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/pck00010.tpc",
            "pck00010.tpc",
            str(tmp_path),
        )
        mock_fetch.assert_any_call(
            "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls",
            "naif0012.tls",
            str(tmp_path),
        )
        assert mock_fetch.call_count == 2

    def test_init_allows_missing_optional_data_configuration(self, tmp_path: Path) -> None:
        cfg = SimpleNamespace(kernel_folder=str(tmp_path), kernels={})

        km = KernelManager(cfg)

        assert km._cfg is cfg

    def test_download_all_data_files_noops_without_optional_data_configuration(self) -> None:
        km = KernelManager.__new__(KernelManager)
        km._cfg = SimpleNamespace()

        with patch.object(km, "_fetch") as mock_fetch:
            km.download_all_data_files()

        mock_fetch.assert_not_called()

    def test_downloads_kernel_by_streaming_chunks_when_content_is_empty(
        self, tmp_path: Path
    ) -> None:
        kernel_name = "pck00010.tpc"
        kernel_url = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/pck00010.tpc"
        chunk_one = FAKE_KERNEL_BYTES[:12]
        chunk_two = FAKE_KERNEL_BYTES[12:]

        mock_response = _mock_response(content=b"")
        mock_response.headers = {"content-length": str(len(FAKE_KERNEL_BYTES))}
        mock_response.iter_content.return_value = [chunk_one, chunk_two]

        with patch("requests.get", return_value=mock_response) as mock_get:
            km = KernelManager.__new__(KernelManager)
            km._fetch(kernel_url, kernel_name, dest=tmp_path)

        mock_get.assert_called_once()
        assert (tmp_path / kernel_name).exists()
        assert (tmp_path / kernel_name).read_bytes() == FAKE_KERNEL_BYTES

    def test_skips_download_when_file_exists(self, tmp_path: Path) -> None:
        # Pre-create the file
        test_file = tmp_path / "test.bsp"
        test_file.write_bytes(FAKE_KERNEL_BYTES)

        mock_response = _mock_response(content=FAKE_KERNEL_BYTES)

        with patch("requests.get", return_value=mock_response) as mock_get:
            km = KernelManager.__new__(KernelManager)
            km._fetch("https://example.com/test.bsp", "test.bsp", dest=tmp_path)

        mock_get.assert_not_called()
        assert (tmp_path / "test.bsp").read_bytes() == FAKE_KERNEL_BYTES

    def test_raises_on_http_error(self, tmp_path: Path) -> None:
        mock_response = _mock_response(status_code=404)

        with patch("requests.get", return_value=mock_response):
            km = KernelManager.__new__(KernelManager)
            with pytest.raises(requests.HTTPError):
                km._fetch("https://example.com/missing.bsp", "missing.bsp", dest=tmp_path)

    def test_raises_on_server_error(self, tmp_path: Path) -> None:
        mock_response = _mock_response(status_code=500)

        with patch("requests.get", return_value=mock_response):
            km = KernelManager.__new__(KernelManager)
            with pytest.raises(requests.HTTPError):
                km._fetch("https://example.com/error.bsp", "error.bsp", dest=tmp_path)

    def test_does_not_write_on_http_error(self, tmp_path: Path) -> None:
        mock_response = _mock_response(status_code=403)

        with patch("requests.get", return_value=mock_response):
            km = KernelManager.__new__(KernelManager)
            with pytest.raises(requests.HTTPError):
                km._fetch("https://example.com/forbidden.bsp", "forbidden.bsp", dest=tmp_path)

        assert not (tmp_path / "forbidden.bsp").exists()

    def test_furnish_loads_configured_kernels(self, tmp_path: Path) -> None:
        kernel_name = "pck00010.tpc"
        kernel_path = tmp_path / kernel_name
        kernel_path.write_bytes(FAKE_KERNEL_BYTES)

        km = KernelManager.__new__(KernelManager)
        km._cfg = SimpleNamespace(
            kernel_folder=str(tmp_path),
            kernels={
                kernel_name: "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/pck00010.tpc"
            },
        )

        with (
            patch("orbitdet.data.kernel.spice.load_standard_kernels") as mock_load_standard,
            patch("orbitdet.data.kernel.spice.load_kernel") as mock_load_kernel,
            patch("orbitdet.data.kernel.spiceypy.furnsh") as mock_furnsh,
            patch("orbitdet.data.kernel.spiceypy.ktotal", return_value=0),
            patch("orbitdet.data.kernel.spiceypy.kdata"),
        ):
            km.furnish()

        mock_load_standard.assert_called_once()
        expected_path = str(kernel_path.resolve())
        mock_load_kernel.assert_called_once_with(expected_path)
        mock_furnsh.assert_called_once_with(expected_path)
