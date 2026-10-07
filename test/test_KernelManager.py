import tarfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, PropertyMock, patch

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_kernel_set_config(
    archive_name: str = "inpop19a.tar.gz",
    url: str = "https://example.com/inpop19a.tar.gz",
    archive_type: str = "tar",
) -> SimpleNamespace:
    """Build a ``SimpleNamespace`` that mimics a single ``kernel_sets.*`` entry."""
    return SimpleNamespace(
        archive=archive_name,
        url=url,
        archive_type=archive_type,
    )


def _make_tar_archive(
    dest: Path,
    archive_name: str,
    members: dict[str, bytes],
) -> Path:
    """Create a real tar archive at *dest* / *archive_name* containing *members*.

    *members* maps member names to their byte content.
    """
    path = dest / archive_name
    mode = "w:gz" if archive_name.endswith(".gz") else "w"
    with tarfile.open(path, mode) as tar:
        for member_name, content in members.items():
            bio = __import__("io").BytesIO(content)
            info = tarfile.TarInfo(name=member_name)
            info.size = len(content)
            tar.addfile(info, bio)
    return path


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
            use_default_kernels=False,
        )

        with (
            patch("orbitdet.data.kernel.spice.load_standard_kernels") as mock_load_standard,
            patch("orbitdet.data.kernel.spice.load_kernel") as mock_load_kernel,
            patch("orbitdet.data.kernel.spiceypy.ktotal", return_value=0),
            patch("orbitdet.data.kernel.spiceypy.kdata"),
        ):
            km.furnish()

        mock_load_standard.assert_not_called()
        expected_path = str(kernel_path.resolve())
        mock_load_kernel.assert_called_once_with(expected_path)


# ======================================================================
# Kernel-set (archive) tests
# ======================================================================


class TestKernelSet:
    """Tests for the ``kernel_sets`` archive download / extract / load feature."""

    # -- _download_and_extract_all_kernel_sets ------------------------------

    def test_download_and_extract_all_kernel_sets_noops_when_no_sets(
        self,
    ) -> None:
        km = KernelManager.__new__(KernelManager)
        km._cfg = SimpleNamespace()

        with patch.object(km, "_download_and_extract_kernel_set") as mock_extract:
            km._download_and_extract_all_kernel_sets()

        mock_extract.assert_not_called()

    def test_download_and_extract_all_kernel_sets_calls_for_each_set(
        self,
    ) -> None:
        km = KernelManager.__new__(KernelManager)
        km._cfg = SimpleNamespace(
            kernel_sets={
                "set_a": _make_kernel_set_config(archive_name="a.tar.gz"),
                "set_b": _make_kernel_set_config(archive_name="b.tar.gz"),
            },
        )

        with patch.object(km, "_download_and_extract_kernel_set") as mock_extract:
            km._download_and_extract_all_kernel_sets()

        assert mock_extract.call_count == 2

    # -- _download_and_extract_kernel_set -----------------------------------

    def test_download_and_extract_kernel_set_downloads_archive(
        self,
        tmp_path: Path,
    ) -> None:
        """Archive is downloaded when it does not exist yet."""
        archive_name = "inpop19a.tar.gz"
        archive_url = "https://example.com/inpop19a.tar.gz"
        tar_bytes = b"fake-tar-content"

        config = _make_kernel_set_config(
            archive_name=archive_name,
            url=archive_url,
        )

        km = KernelManager.__new__(KernelManager)
        km._cfg = SimpleNamespace(kernel_folder=str(tmp_path))

        with patch("requests.get", return_value=_mock_response(content=tar_bytes)):
            with patch.object(km, "_extract_from_tar") as mock_extract:
                km._download_and_extract_kernel_set("inpop19a", config)

        # Archive file was written
        archive_path = tmp_path / archive_name
        assert archive_path.exists()
        assert archive_path.read_bytes() == tar_bytes
        # Extraction was called
        mock_extract.assert_called_once_with(archive_path, tmp_path)

    def test_download_and_extract_kernel_set_skips_existing_archive(
        self,
        tmp_path: Path,
    ) -> None:
        """Archive download is skipped when the file already exists."""
        archive_name = "inpop19a.tar.gz"
        (tmp_path / archive_name).write_text("existing")

        config = _make_kernel_set_config(
            archive_name=archive_name,
            url="https://example.com/inpop19a.tar.gz",
        )

        km = KernelManager.__new__(KernelManager)
        km._cfg = SimpleNamespace(kernel_folder=str(tmp_path))

        with patch("requests.get") as mock_get:
            with patch.object(km, "_extract_from_tar"):
                km._download_and_extract_kernel_set("inpop19a", config)

        mock_get.assert_not_called()

    def test_download_and_extract_kernel_set_raises_on_unsupported_type(
        self,
        tmp_path: Path,
    ) -> None:
        config = _make_kernel_set_config(
            archive_name="data.zip",
            archive_type="zip",
        )

        km = KernelManager.__new__(KernelManager)
        km._cfg = SimpleNamespace(kernel_folder=str(tmp_path))

        with patch("requests.get", return_value=_mock_response(content=b"data")):
            with pytest.raises(ValueError, match="Unsupported archive type"):
                km._download_and_extract_kernel_set("bad", config)

    # -- _extract_from_tar --------------------------------------------------

    def test_extract_from_tar_extracts_all_files(
        self,
        tmp_path: Path,
    ) -> None:
        archive_path = _make_tar_archive(
            tmp_path,
            "kernels.tar.gz",
            {
                "kern.bpc": b"bpc-data",
                "kern.bsp": b"bsp-data",
                "kern.tf": b"tf-data",
                "extra.txt": b"should-also-appear",
            },
        )

        KernelManager._extract_from_tar(archive_path, tmp_path)

        expected = {
            "kern.bpc": b"bpc-data",
            "kern.bsp": b"bsp-data",
            "kern.tf": b"tf-data",
            "extra.txt": b"should-also-appear",
        }
        for name, content in expected.items():
            assert (tmp_path / name).exists(), f"{name} should have been extracted"
            assert (tmp_path / name).read_bytes() == content

    def test_extract_from_tar_skips_already_extracted_files(
        self,
        tmp_path: Path,
    ) -> None:
        """Files that already exist on disk are not re-extracted."""
        (tmp_path / "kern.bsp").write_text("existing")
        archive_path = _make_tar_archive(
            tmp_path,
            "kernels.tar.gz",
            {"kern.bsp": b"new-data", "kern.bpc": b"bpc-data"},
        )

        KernelManager._extract_from_tar(archive_path, tmp_path)

        # Existing file was NOT overwritten
        assert (tmp_path / "kern.bsp").read_text() == "existing"
        # New file was extracted
        assert (tmp_path / "kern.bpc").read_bytes() == b"bpc-data"

    # -- download_all_kernels with kernel sets ------------------------------

    def test_download_all_kernels_extracts_sets_then_fetches_individual(
        self,
        tmp_path: Path,
    ) -> None:
        """download_all_kernels extracts archives first, then fetches every
        kernel individually (archive-extracted files are already on disk)."""
        cfg = SimpleNamespace(
            kernel_folder=str(tmp_path),
            kernels={
                "pck00010.tpc": "https://example.com/pck00010.tpc",
                "inpop19a.bsp": "https://example.com/inpop19a.tar.gz",
            },
            kernel_sets={
                "inpop19a": _make_kernel_set_config(
                    archive_name="inpop19a.tar.gz",
                    url="https://example.com/inpop19a.tar.gz",
                ),
            },
        )

        km = KernelManager(cfg)

        with (
            patch.object(km, "_download_and_extract_all_kernel_sets") as mock_extract_sets,
            patch.object(km, "_fetch") as mock_fetch,
        ):
            km.download_all_kernels()

        mock_extract_sets.assert_called_once()
        # Both kernels are fetched individually (archive URL is NOT skipped)
        assert mock_fetch.call_count == 2
        mock_fetch.assert_any_call(
            "https://example.com/pck00010.tpc",
            "pck00010.tpc",
            str(tmp_path),
        )
        mock_fetch.assert_any_call(
            "https://example.com/inpop19a.tar.gz",
            "inpop19a.bsp",
            str(tmp_path),
        )

    # -- _get_all_kernel_names ----------------------------------------------

    def test_get_all_kernel_names_returns_kernels_in_order(self) -> None:
        km = KernelManager.__new__(KernelManager)
        km._cfg = SimpleNamespace(
            kernels={
                "first.tpc": "https://example.com/first.tpc",
                "second.bsp": "https://example.com/second.bsp",
                "third.tls": "https://example.com/third.tls",
            },
        )

        names = km._get_all_kernel_names()

        assert names == ["first.tpc", "second.bsp", "third.tls"]

    def test_get_all_kernel_names_includes_archive_urls(self) -> None:
        """Archive URLs are NOT filtered out — they are valid kernel names
        whose files end up on disk after extraction."""
        km = KernelManager.__new__(KernelManager)
        km._cfg = SimpleNamespace(
            kernels={
                "pck00010.tpc": "https://example.com/pck.tpc",
                "inpop19a.bsp": "https://example.com/inpop19a.tar.gz",
            },
            kernel_sets={
                "inpop19a": _make_kernel_set_config(
                    url="https://example.com/inpop19a.tar.gz",
                ),
            },
        )

        names = km._get_all_kernel_names()

        assert names == ["pck00010.tpc", "inpop19a.bsp"]

    # -- furnish with kernel sets -------------------------------------------

    def test_furnish_loads_all_kernels_in_order(self, tmp_path: Path) -> None:
        """furnish() loads kernels in the order they appear in the config."""
        # Create the kernel files on disk
        (tmp_path / "first.tpc").write_bytes(FAKE_KERNEL_BYTES)
        (tmp_path / "second.bsp").write_bytes(FAKE_KERNEL_BYTES)
        (tmp_path / "third.tls").write_bytes(FAKE_KERNEL_BYTES)

        km = KernelManager.__new__(KernelManager)
        km._cfg = SimpleNamespace(
            kernel_folder=str(tmp_path),
            kernels={
                "first.tpc": "https://example.com/first.tpc",
                "second.bsp": "https://example.com/second.bsp",
                "third.tls": "https://example.com/third.tls",
            },
            use_default_kernels=False,
        )

        with (
            patch("orbitdet.data.kernel.spice.load_standard_kernels") as mock_std,
            patch("orbitdet.data.kernel.spice.load_kernel") as mock_load,
            patch("orbitdet.data.kernel.spiceypy.ktotal", return_value=0),
            patch("orbitdet.data.kernel.spiceypy.kdata"),
        ):
            km.furnish()

        mock_std.assert_not_called()
        assert mock_load.call_count == 3
        # Verify order is preserved
        calls = [args[0][0] for args in mock_load.call_args_list]
        assert calls == [
            str((tmp_path / "first.tpc").resolve()),
            str((tmp_path / "second.bsp").resolve()),
            str((tmp_path / "third.tls").resolve()),
        ]

    # -- _load_kernel -------------------------------------------------------

    def test_load_kernel_raises_when_file_missing(self, tmp_path: Path) -> None:
        km = KernelManager.__new__(KernelManager)
        km._cfg = SimpleNamespace(kernel_folder=str(tmp_path))

        with (
            patch("orbitdet.data.kernel.spiceypy.ktotal", return_value=0),
            patch("orbitdet.data.kernel.spiceypy.kdata"),
        ):
            with pytest.raises(FileNotFoundError, match="missing.bsp"):
                km._load_kernel("missing.bsp")

    def test_load_kernel_skips_already_loaded_by_name(self, tmp_path: Path) -> None:
        (tmp_path / "dup.bsp").write_bytes(FAKE_KERNEL_BYTES)

        km = KernelManager.__new__(KernelManager)
        km._cfg = SimpleNamespace(kernel_folder=str(tmp_path))

        with (
            patch("orbitdet.data.kernel.spice.load_kernel") as mock_load,
            patch("orbitdet.data.kernel.spiceypy.ktotal", return_value=1),
            patch(
                "orbitdet.data.kernel.spiceypy.kdata",
                return_value=("/some/path/dup.bsp", "SPK", "", ""),
            ),
        ):
            km._load_kernel("dup.bsp")

        mock_load.assert_not_called()
