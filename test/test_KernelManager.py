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
    use_from_archive: list[str] | None = None,
) -> SimpleNamespace:
    """Build a ``SimpleNamespace`` that mimics a single ``kernel_sets.*`` entry."""
    if use_from_archive is None:
        use_from_archive = [
            "inpop19a.bpc",
            "inpop19a.bsp",
            "inpop19a.tf",
            "inpop19a.tpc",
            "inpop19a_time.bsp",
        ]
    return SimpleNamespace(
        archive=archive_name,
        url=url,
        archive_type=archive_type,
        use_from_archive=use_from_archive,
    )


def _make_tar_archive(
    dest: Path,
    archive_name: str,
    members: dict[str, bytes],
    archive_type: str = "tar",
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

    # -- _get_kernel_set_urls -----------------------------------------------

    def test_get_kernel_set_urls_returns_empty_when_no_sets(self) -> None:
        km = KernelManager.__new__(KernelManager)
        km._cfg = SimpleNamespace()

        assert km._get_kernel_set_urls() == set()

    def test_get_kernel_set_urls_returns_urls(self) -> None:
        km = KernelManager.__new__(KernelManager)
        km._cfg = SimpleNamespace(
            kernel_sets={
                "a": SimpleNamespace(url="https://example.com/a.tar.gz"),
                "b": SimpleNamespace(url="https://example.com/b.tar.gz"),
            },
        )

        result = km._get_kernel_set_urls()

        assert result == {
            "https://example.com/a.tar.gz",
            "https://example.com/b.tar.gz",
        }

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
            use_from_archive=["kern.bsp"],
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
        mock_extract.assert_called_once_with(
            archive_path,
            config.use_from_archive,
            tmp_path,
        )

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

    def test_extract_from_tar_extracts_requested_files(
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
                "extra.txt": b"should-not-appear",
            },
        )
        requested = ["kern.bpc", "kern.bsp", "kern.tf"]

        KernelManager._extract_from_tar(archive_path, requested, tmp_path)

        expected = {
            "kern.bpc": b"bpc-data",
            "kern.bsp": b"bsp-data",
            "kern.tf": b"tf-data",
        }
        for name in requested:
            assert (tmp_path / name).exists(), f"{name} should have been extracted"
            assert (tmp_path / name).read_bytes() == expected[name]
        # Extra file not in the requested list was NOT extracted
        assert not (tmp_path / "extra.txt").exists()

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

        KernelManager._extract_from_tar(archive_path, ["kern.bsp", "kern.bpc"], tmp_path)

        # Existing file was NOT overwritten
        assert (tmp_path / "kern.bsp").read_text() == "existing"
        # New file was extracted
        assert (tmp_path / "kern.bpc").read_bytes() == b"bpc-data"

    # -- download_all_kernels with kernel sets ------------------------------

    def test_download_all_kernels_skips_archive_urls(
        self,
        tmp_path: Path,
    ) -> None:
        """Top-level kernel entries whose URL matches a kernel-set archive URL
        are skipped during individual download."""
        archive_url = "https://example.com/inpop19a.tar.gz"
        cfg = SimpleNamespace(
            kernel_folder=str(tmp_path),
            kernels={
                "pck00010.tpc": "https://example.com/pck00010.tpc",
                "inpop19a.bsp": archive_url,
            },
            kernel_sets={
                "inpop19a": _make_kernel_set_config(
                    archive_name="inpop19a.tar.gz",
                    url=archive_url,
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
        # Only the non-archive kernel is fetched individually
        mock_fetch.assert_called_once_with(
            "https://example.com/pck00010.tpc",
            "pck00010.tpc",
            str(tmp_path),
        )

    # -- _get_all_kernel_names ----------------------------------------------

    def test_get_all_kernel_names_excludes_archive_urls_and_mk_files(
        self,
    ) -> None:
        km = KernelManager.__new__(KernelManager)
        km._cfg = SimpleNamespace(
            kernels={
                "pck00010.tpc": "https://example.com/pck.tpc",
                "inpop19a.bsp": "https://example.com/inpop19a.tar.gz",
            },
            kernel_sets={
                "inpop19a": _make_kernel_set_config(
                    url="https://example.com/inpop19a.tar.gz",
                    use_from_archive=[
                        "inpop19a_TDB.bpc",
                        "inpop19a_TDB.bsp",
                        "inpop19a.mk",
                        "inpop19a_TDB.tf",
                    ],
                ),
            },
        )

        names = km._get_all_kernel_names()

        # Regular kernel (non-archive URL) is included
        assert "pck00010.tpc" in names
        # Archive URL kernel is excluded
        assert "inpop19a.bsp" not in names
        # Archive members are included
        assert "inpop19a_TDB.bpc" in names
        assert "inpop19a_TDB.bsp" in names
        assert "inpop19a_TDB.tf" in names
        # .mk file is excluded
        assert "inpop19a.mk" not in names

    def test_get_all_kernel_names_no_kernel_sets(self) -> None:
        km = KernelManager.__new__(KernelManager)
        km._cfg = SimpleNamespace(
            kernels={
                "a.tpc": "https://example.com/a.tpc",
                "b.bsp": "https://example.com/b.bsp",
            },
        )

        names = km._get_all_kernel_names()

        assert names == ["a.tpc", "b.bsp"]

    # -- furnish with kernel sets -------------------------------------------

    def test_furnish_loads_kernel_set_files(self, tmp_path: Path) -> None:
        """furnish() loads both regular kernels and kernel-set members."""
        # Create the kernel files on disk
        (tmp_path / "pck00010.tpc").write_bytes(FAKE_KERNEL_BYTES)
        (tmp_path / "inpop19a.bpc").write_bytes(FAKE_KERNEL_BYTES)
        (tmp_path / "inpop19a.bsp").write_bytes(FAKE_KERNEL_BYTES)

        km = KernelManager.__new__(KernelManager)
        km._cfg = SimpleNamespace(
            kernel_folder=str(tmp_path),
            kernels={
                "pck00010.tpc": "https://example.com/pck.tpc",
            },
            kernel_sets={
                "inpop19a": _make_kernel_set_config(
                    use_from_archive=[
                        "inpop19a.bpc",
                        "inpop19a.bsp",
                    ],
                ),
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
        # All three kernel files are loaded
        assert mock_load.call_count == 3
        mock_load.assert_any_call(str((tmp_path / "pck00010.tpc").resolve()))
        mock_load.assert_any_call(str((tmp_path / "inpop19a.bpc").resolve()))
        mock_load.assert_any_call(str((tmp_path / "inpop19a.bsp").resolve()))

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
