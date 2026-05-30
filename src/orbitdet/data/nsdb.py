"""nsdb.py

Manager for NSDB page scraping, metadata enrichment, and output generation.
This module provides the NSDBManager class which can be used to download NSDB datasets, parse their
metadata, and generate Hydra configuration files for use in the dataset generation process.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import requests
import yaml
from bs4 import BeautifulSoup


class NSDBManager:
    """Manager for NSDB page scraping, metadata enrichment, and output generation."""

    NSDB_BASE_URL = "https://nsdb.imcce.fr/obspos/OBS_COLL/"
    DEFAULT_TIMEOUT_SECONDS = 30

    def __init__(
        self,
        base_url: str = NSDB_BASE_URL,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    ):
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds

    def download_dataset(self, name: str, destination: str | Path) -> tuple[Path, Path]:
        """Download an NSDB observation data file to a destination path.

        Args:
            name (str): name of the dataset to download, e.g. "nm0001"
            destination (str | Path): directory to download the dataset files into. The method will
            create a subdirectory named after the dataset, e.g. "nm0001", within this destination.

        Raises:
            FileExistsError: if the destination path already exists to avoid overwriting existing
            data

        Returns:
            tuple[Path, Path]: paths to the downloaded data file and content metadata file
            respectively
        """
        # Create destination directory, fails if path already exists to avoid overwriting
        destination_path = Path(destination)
        if destination_path.exists():
            raise FileExistsError(f"Destination path {destination_path} already exists")
        destination_path.mkdir()
        primary_body = name[0]

        data_file = destination_path / f"{name}.txt"
        content_file = destination_path / f"{name}.html"

        # Download
        main_url = self.base_url + primary_body.upper() + "/" + name
        response_data = requests.get(main_url + ".txt", timeout=self.timeout_seconds)
        response_data.raise_for_status()
        data_file.write_bytes(response_data.content)
        response_content = requests.get(main_url + ".html", timeout=self.timeout_seconds)
        response_content.raise_for_status()
        content_file.write_bytes(response_content.content)
        return data_file, content_file

    def generate_hydra_configs(
        self,
        content: dict,
        output_dir: str | Path,
        data_file: str | Path | None = None,
    ) -> None:
        """Generate dataset YAML files using enriched contents metadata.

        Args:
            content (dict): metadata dictionary
            output_dir (str | Path): directory to write the generated dataset config into
            data_file (str | Path | None): resolved path to the dataset data file
        """

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        config = dict(content)
        if data_file is not None:
            config["file"] = str(Path(data_file).resolve())
        filename = f"{content['identifier']}.yaml"
        with (output_path / filename).open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)

    def _parse_contents_metadata(
        self, content_html: str, identifier: str | None = None
    ) -> dict[str, Any]:
        soup = BeautifulSoup(content_html, "html.parser")
        title = soup.title.get_text(strip=True) if soup.title else ""
        pre = soup.find("pre")
        text = pre.get_text("\n") if pre else soup.get_text("\n")

        # Split into lines and collect section blocks by heading
        lines = [ln.rstrip() for ln in text.replace("\r", "").split("\n")]
        sections: dict[str, list[str]] = {
            "contents": [],
            "informations": [],
            "reference": [],
            "comments": [],
            "format": [],
        }
        current: str | None = None

        heading_re = re.compile(
            r"^(contents|informations|reference|comments|format)\.?$", re.IGNORECASE
        )
        for raw in lines:
            ln = raw.strip()
            if not ln:
                continue
            m = heading_re.match(ln)
            if m:
                current = m.group(1).lower()
                continue
            if current:
                sections[current].append(ln)

        # Build result dict
        result: dict[str, Any] = {"identifier": identifier or title, "format_columns": {}}

        # Parse format section into mapping index->label
        fmt_map: dict[str, str] = {}
        for line in sections["format"]:
            match = re.match(r"^(\d+)\.\s*(.+)$", line)
            if match:
                fmt_map[match.group(1)] = re.sub(r"\s+", " ", match.group(2)).strip()
        result["format_columns"] = fmt_map

        # Parse contents and informations and merge into root.
        try:
            contents_text = "\n".join(sections["contents"]) if sections["contents"] else ""
            if contents_text:
                parsed_contents = self._parse_contents(contents_text)
                # Merge, do not overwrite existing top-level reserved keys
                for k, v in parsed_contents.items():
                    if k in ("identifier", "format_columns"):
                        continue
                    result[k] = v
        except Exception:
            pass

        try:
            informations_text = (
                "\n".join(sections["informations"]) if sections["informations"] else ""
            )
            if informations_text:
                parsed_infos = self._parse_informations(informations_text)
                for k, v in parsed_infos.items():
                    if k in ("identifier", "format_columns"):
                        continue
                    # informations take precedence over contents on collision
                    result[k] = v
        except Exception:
            pass

        # Reference and comments are kept as trimmed raw strings if present
        if sections["reference"]:
            result["reference"] = "\n".join(sections["reference"]).strip()
        if sections["comments"]:
            result["comments"] = "\n".join(sections["comments"]).strip()

        original_type = result.get("type")
        receptor = result.get("receptor")
        coordinates = result.get("coordinates").replace(" ", "_")
        if original_type and receptor:
            result["type"] = f"{original_type}_{coordinates}_{receptor}_nsdb"

        return result

    def _parse_contents(self, contents: str) -> dict[str, Any]:
        """Parses contents section in nsdb contents file

            Args:
                contents (str): set of lines containing contents section

            Returns:
                dict[str, str]: formatted and fully split contents section dictionary

            Example contents line:
            Contents.
                planet: 8 - Neptune
                satellites: N2-Nereid   :174
                total number: 174
                type: relative
                dates: 1993-1998
                observatory: 874 - Itajuba (Laboratorio Nacional de Astrofisica)

            returns:
            {
            "planet": {
                "number": 8,
                "name": "Neptune"
            },
            "satellites": {
                Nereid: {
                    "designation": "N2-Nereid",
                    "number_of_observations": 174
                    }
                },
            "total_number": 174,
            "type": "relative",
            "dates": {
                start_year: 1993,
                end_year: 1998
                },
            }
            observatory: {
                "code": 874,
                "name": "Itajuba (Laboratorio Nacional de Astrofisica)"
            }
        }
        """
        # Normalize input into lines
        lines = [ln.strip() for ln in contents.replace("\r", "").split("\n") if ln.strip()]

        contents_block: dict[str, Any] = {}

        for line in lines:
            if ":" not in line:
                continue
            key, val = [p.strip() for p in line.split(":", 1)]
            lkey = key.lower()

            if lkey == "planet":
                # e.g. '8 - Neptune'
                parts = [p.strip() for p in re.split(r"-", val, maxsplit=1)]
                try:
                    number = int(parts[0])
                except Exception:
                    number = None
                name = parts[1] if len(parts) > 1 else (parts[0] if number is None else "")
                contents_block["planet"] = {"number": number, "name": name}
                continue

            if lkey == "satellites":
                # e.g. 'N2-Nereid   :174' or 'N2-Nereid :174'
                # split designation and optional count
                if ":" in val:
                    designation, count_s = [p.strip() for p in val.split(":", 1)]
                    try:
                        count = int(count_s)
                    except Exception:
                        count = None
                else:
                    designation = val.strip()
                    count = None
                # try to extract short name after last '-'
                if "-" in designation:
                    short_name = designation.split("-")[-1].strip()
                else:
                    short_name = designation
                satellites = {
                    short_name: {"designation": designation, "number_of_observations": count}
                }
                contents_block["satellites"] = satellites
                continue

            if lkey in {"total number", "total_number"}:
                try:
                    contents_block["total_number"] = int(re.sub(r"[^0-9]", "", val))
                except Exception:
                    contents_block["total_number"] = None
                continue

            if lkey == "type":
                contents_block["type"] = val
                continue

            if lkey == "dates":
                # e.g. '1993-1998' or '1993 - 1998'
                parts = re.split(r"-", val)
                try:
                    start = int(re.sub(r"[^0-9]", "", parts[0]))
                except Exception:
                    start = None
                try:
                    end = int(re.sub(r"[^0-9]", "", parts[1])) if len(parts) > 1 else None
                except Exception:
                    end = None
                contents_block["dates"] = {"start_year": start, "end_year": end}
                continue

            if lkey == "observatory":
                # e.g. '874 - Itajuba (Lab...)'
                parts = [p.strip() for p in re.split(r"-", val, maxsplit=1)]
                try:
                    code = int(re.sub(r"[^0-9]", "", parts[0]))
                except Exception:
                    code = None
                name = parts[1] if len(parts) > 1 else parts[0]
                contents_block["observatory"] = {"code": code, "name": name}
                continue

            # Fallback: store raw value under its normalized key
            contents_block[lkey.replace(" ", "_")] = val

        return contents_block

    def _parse_informations(self, informations: str) -> dict[str, Any]:
        """Parses informations section on NSDB contents file

        Args:
            informations (str): Information section string

        Returns:
            dict[str, str]: formatted and fully split informations section as dict

        Example:
        Informations.
            relative to: N1-Triton  :174
            reference frame: astrometric
            centre of frame: topocentre
            epoch of equinox: J2000
            time scale: UTC
            reduction: no information
            coordinates: X,Y
            diff. refraction: no information
            receptor: CCD
            telescope: Reflector, D = 1.6 m
            observers: Veiga C.H., Vieira Martins R.
        data included in standard data file: no

        returns:
        {
            relative_to: {
                "designation": "N1-Triton",
                "number_of_observations": 174
            },
            reference_frame: astrometric,
            centre_of_frame: topocentre,
            epoch_of_equinox: J2000,
            time_scale: UTC,
            reduction: no information,
            coordinates: X,Y,
            diff_refraction: no information,
            receptor: CCD,
            telescope: Reflector, D = 1.6 m,
            observers: Veiga C.H., Vieira Martins R.
            data_included_in_standard_data_file: no

        """
        lines = [ln.strip() for ln in informations.replace("\r", "").split("\n") if ln.strip()]

        field_map = {
            "reference frame": "reference_frame",
            "centre of frame": "centre_of_frame",
            "epoch of equinox": "epoch_of_equinox",
            "time scale": "time_scale",
            "reduction": "reduction",
            "coordinates": "coordinates",
            "diff. refraction": "diff_refraction",
            "receptor": "receptor",
            "telescope": "telescope",
            "observers": "observers",
            "data included in standard data file": "data_included_in_standard_data_file",
        }

        parsed: dict[str, Any] = {}
        current_alias: str | None = None

        for line in lines:
            if ":" in line:
                key, value = [p.strip() for p in line.split(":", 1)]
                lowered_key = key.lower()

                if lowered_key == "relative to":
                    designation = value
                    count: int | None = None
                    if ":" in value:
                        designation, count_s = [p.strip() for p in value.split(":", 1)]
                        try:
                            count = int(re.sub(r"[^0-9]", "", count_s))
                        except Exception:
                            count = None
                    parsed["relative_to"] = {
                        "designation": designation,
                        "number_of_observations": count,
                    }
                    current_alias = None
                    continue

                alias = field_map.get(lowered_key)
                if alias is not None:
                    parsed[alias] = value
                    current_alias = alias
                    continue

                current_alias = None
                continue

            # Support line wraps in long fields like telescope/observers
            if current_alias and "-" * 5 not in line:
                joined = f"{parsed[current_alias]} {line}".strip()
                parsed[current_alias] = re.sub(r"\s+", " ", joined)

        return parsed
