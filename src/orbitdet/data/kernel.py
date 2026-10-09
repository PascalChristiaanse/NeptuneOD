import logging
import tarfile
from pathlib import Path
from typing import NoReturn

import requests
import spiceypy
from omegaconf import DictConfig
from tudatpy.interface import spice

logger = logging.getLogger(__name__)


class KernelEntry:
    def __init__(self, url, name):
        self.url = url
        self.name = name


class KernelManager:
    def __init__(self, cfg: DictConfig):
        self._cfg = cfg

        if not Path(self._cfg.kernel_folder).exists():
            raise FileNotFoundError(f"Kernel path {self._cfg.kernel_folder} does not exist")
        data_folder = getattr(self._cfg, "data_folder", None)
        if data_folder is not None and not Path(data_folder).exists():
            raise FileNotFoundError(f"Data path {data_folder} does not exist")

    def download_all_data_files(self):
        data_files = getattr(self._cfg, "data_files", None)
        data_folder = getattr(self._cfg, "data_folder", None)
        if not data_files or not data_folder:
            return

        for file, url in self._cfg.data_files.items():
            self._fetch(url, file, self._cfg.data_folder)

    def download_all_kernels(self):
        # First, download and extract any kernel sets (archives)
        self._download_and_extract_all_kernel_sets()

        # Then download individual kernels
        for kernel, url in self._cfg.kernels.items():
            self._fetch(url, kernel, self._cfg.kernel_folder)

    # ------------------------------------------------------------------
    # Kernel set (archive) support
    # ------------------------------------------------------------------

    def _download_and_extract_all_kernel_sets(self):
        """Download and extract every kernel set archive."""
        kernel_sets = getattr(self._cfg, "kernel_sets", None)
        if not kernel_sets:
            return

        for ks_name, ks_config in kernel_sets.items():
            self._download_and_extract_kernel_set(ks_name, ks_config)

    def _download_and_extract_kernel_set(self, name: str, config: DictConfig):
        """Download a single kernel-set archive and extract all files from it."""
        archive_name = config.archive
        archive_url = config.url
        archive_type = getattr(config, "archive_type", "tar")

        dest_path = Path(self._cfg.kernel_folder)
        dest_path.mkdir(parents=True, exist_ok=True)

        archive_path = dest_path / archive_name

        # Download the archive if not already present
        if not archive_path.exists():
            logger.info(
                "Downloading kernel set archive %s from %s",
                archive_name,
                archive_url,
            )
            response = requests.get(archive_url, stream=True)
            response.raise_for_status()

            with archive_path.open("wb") as fh:
                content = getattr(response, "content", b"") or b""
                if content:
                    fh.write(content)
                else:
                    total = int(response.headers.get("content-length", 0) or 0)
                    downloaded = 0
                    chunk_size = 8192
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if not chunk:
                            continue
                        fh.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            percent = downloaded * 100 / total
                            logger.debug(
                                "%s: %d/%d bytes (%.1f%%)",
                                archive_name,
                                downloaded,
                                total,
                                percent,
                            )
                        else:
                            logger.debug("%s: %d bytes", archive_name, downloaded)
            logger.info("Download of archive %s complete.", archive_name)
        else:
            logger.info("Archive %s already exists, skipping download", archive_name)

        # Extract all files from the archive
        if archive_type == "tar":
            self._extract_from_tar(archive_path, dest_path)
        else:
            raise ValueError(f"Unsupported archive type: {archive_type}")

    @staticmethod
    def _extract_from_tar(archive_path: Path, dest: Path):
        """Extract all files from the tar archive at *archive_path* into *dest*.

        Existing files are skipped to avoid unnecessary I/O.
        """
        with tarfile.open(archive_path, "r:*") as tar:
            for member in tar.getmembers():
                member_path = dest / member.name
                if member_path.exists():
                    logger.info("File %s already extracted, skipping", member.name)
                    continue
                logger.info("Extracting %s from %s", member.name, archive_path.name)
                tar.extract(member, path=dest, filter="data")

    # ------------------------------------------------------------------
    # Single-file download
    # ------------------------------------------------------------------

    def _fetch(self, url: str, name: str, dest: Path) -> NoReturn:
        """Download all required kernels

        Args:
            url (str): NAIF sub URL for the kernel, e.g. "/lsk/naif0012.tls"
            name (str): Name of the kernel file, e.g. "naif0012.tls"
            dest (Path): Destination path for the kernel file (default: KERNEL_PATH)
        """
        dest_path = Path(dest)
        dest_path.mkdir(parents=True, exist_ok=True)
        file = dest_path / name
        if not file.exists():
            logger.info(f"Downloading {url} -> {file.name}")
            response = requests.get(url, stream=True)
            response.raise_for_status()

            with file.open("wb") as fh:
                content = getattr(response, "content", b"") or b""
                if content:
                    fh.write(content)
                else:
                    total = int(response.headers.get("content-length", 0) or 0)
                    downloaded = 0
                    chunk_size = 8192
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if not chunk:
                            continue
                        fh.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            percent = downloaded * 100 / total
                            # Per-chunk progress at DEBUG so large downloads
                            # do not flood the INFO log (the old code printed
                            # a single \r-updated line to the terminal).
                            logger.debug(
                                "%s: %d/%d bytes (%.1f%%)",
                                file.name,
                                downloaded,
                                total,
                                percent,
                            )
                        else:
                            logger.debug("%s: %d bytes", file.name, downloaded)
            logger.info("Download of %s complete.", file.name)
        else:
            logger.info(f"Kernel {name} already exists, skipping download")

    # ------------------------------------------------------------------
    # Kernel loading
    # ------------------------------------------------------------------

    def furnish(self):
        """Load all required kernels"""

        if getattr(self._cfg, "use_default_kernels", False):
            spice.load_standard_kernels()
            logger.warning(
                """Standard SPICE kernels loaded. This may lead to conflicts if """
                """custom kernels have overlapping coverage."""
            )

        for kernel in self._get_all_kernel_names():
            self._load_kernel(kernel)

    def _get_all_kernel_names(self) -> list[str]:
        """Return every kernel file that should be loaded, in order.

        Simply returns the keys of ``kernels`` in the order they are defined.
        Archive-extracted files are already on disk and will be found by name.
        """
        return list(self._cfg.kernels.keys())

    def _load_kernel(self, kernel_name: str):
        """Load a single kernel by *kernel_name*, guarding against duplicates."""
        path = Path(self._cfg.kernel_folder, kernel_name)
        loaded_kernels = self.get_current_kernels()[0]
        if kernel_name in loaded_kernels:
            logger.warning("Kernel %s already loaded, skipping", kernel_name)
            return
        loaded_files = self.get_current_kernels()[1]
        if str(path.resolve()) in loaded_files:
            logger.error("Kernel %s already loaded from %s!", kernel_name, path)
            raise RuntimeError(f"Kernel {kernel_name} already loaded from {path}!")

        if not path.exists():
            raise FileNotFoundError(f"Required kernel {kernel_name} not found at {path}")
        logger.info("Loading kernel %s from %s", kernel_name, path)
        spice.load_kernel(str(path.resolve()))
        # spiceypy.furnsh(str(path.resolve())) spice and spiceypy share the same kernel pool,
        # so loading with one library makes the kernels available to the other
        # self.log_current_kernel_pool()

    def log_current_kernel_pool(self):
        logger.info("Current loaded kernels:")

        # Log ALL loaded kernels sorted by type
        kernel_count = spiceypy.ktotal("ALL")
        kernel_types = set()
        for i in range(kernel_count):
            kernel_type = spiceypy.kdata(i, "ALL")[0]
            # Collect unique kernel types based on extension
            ktype = kernel_type.split(".")[-1].upper()  # Get extension and convert to uppercase
            kernel_types.add(ktype)

        for kernel_type in sorted(kernel_types):
            logger.info(f"  {kernel_type} kernels:")
            for i in range(kernel_count):
                kernel_path, _, _, _ = spiceypy.kdata(i, "ALL")
                ktype_i = kernel_path.split(".")[-1].upper()
                kernel_name = kernel_path.split("/")[-1]  # Get just the filename
                if ktype_i == kernel_type:
                    logger.info(f"    {kernel_name} ({kernel_path})")

    def get_current_kernels(self) -> tuple[set[str], set[str]]:
        """Get the set of currently loaded kernels and files in the kernel folder"""
        kernel_count = spiceypy.ktotal("ALL")
        kernels = set()
        kernel_files = set()
        for i in range(kernel_count):
            kernel_path, _, _, _ = spiceypy.kdata(i, "ALL")
            kernel_name = kernel_path.split("/")[-1]  # Get just the filename
            kernels.add(kernel_name)
            kernel_files.add(kernel_path)
        return kernels, kernel_files
