import logging
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

    def download_all_kernels(self):
        for kernel, url in self._cfg.kernels.items():
            self._fetch(url, kernel, self._cfg.kernel_folder)

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
                            print(
                                f"\r{file.name}: {downloaded}/{total} bytes ({percent:.1f}%)",
                                end="",
                                flush=True,
                            )
                        else:
                            print(f"\r{file.name}: {downloaded} bytes", end="", flush=True)
            print()
        else:
            logger.info(f"Kernel {name} already exists, skipping download")

    def furnish(self):
        """Load all required kernels"""
        spice.load_standard_kernels()
        logger.warning(
            """Standard SPICE kernels loaded. This may lead to conflicts if """
            """custom kernels have overlapping coverage."""
        )
        for kernel in self._cfg.kernels.keys():
            path = Path(self._cfg.kernel_folder + "/" + kernel)
            loaded_kernels = self.get_current_kernels()[0]
            if kernel in loaded_kernels:
                logger.warning(f"Kernel {kernel} already loaded, skipping")
                continue
            loaded_files = self.get_current_kernels()[1]
            if str(path.resolve()) in loaded_files:
                logger.error(f"Kernel {kernel} already loaded from {path}!")
                raise RuntimeError(f"Kernel {kernel} already loaded from {path}!")

            if not path.exists():
                raise FileNotFoundError(f"Required kernel {kernel} not found at {path}")
            logger.info(f"Loading kernel {kernel} from {path}")
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
