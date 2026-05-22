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
            if not path.exists():
                raise FileNotFoundError(f"Required kernel {kernel} not found at {path}")
            logger.info(f"Loading kernel {kernel} from {path}")
            spice.load_kernel(str(path.resolve()))
            spiceypy.furnsh(str(path.resolve()))
