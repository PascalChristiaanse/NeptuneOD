"""Tests for :mod:`orbitdet.data.nsdb_downloader` and :mod:`orbitdet.data.nsdb_manager`.

The downloader is exercised with a fake session rather than the network, so the
tests are hermetic. The only file that matters is :class:`FakeSession`, which
records what was requested and serves canned bytes.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml

from orbitdet.data.nsdb_downloader import NSDBDatasetFiles, NSDBDownloader
from orbitdet.data.nsdb_manager import NSDBManager
from orbitdet.data.nsdb_parser import NSDBParser

DATA_RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
REAL_DESCRIPTORS = sorted(DATA_RAW.glob("nm*/nm*.html"))

DESCRIPTOR = """<HTML><HEAD><TITLE>nm0001</TITLE></HEAD><PRE>
Contents.
      planet: 8 - Neptune
  satellites: N1-Triton    :3
total number: 3
        type: absolute
       dates: 2001-2001
 observatory: 673 -Table Mountain Observatory, Wrightwood

Reference.
   (2001) Communicated to NSDC by W. M. Owen Jr.,

Informations.
         relative to: absolute
     reference frame: astrometric
     centre of frame: topocentre
    epoch of equinox: J2000
          time scale: UTC
           reduction: no information
         coordinates: absolute
    diff. refraction: no information
            receptor: CCD
           telescope: Reflector, D = 0.61 m, f/16
           observers: Owen Jr.W.M.
 data included in standard data file: no

Format.
  1. Number of satellite (N sat)
  2. Year   of the moment of observation
  3. Day    of the moment of observation with decimals

----------------------------------------------------
 N  Year      UT
sat.  month day
----------------------------------------------------
</PRE></BODY></HTML>
"""

DATA = " 1  2001 05 30.45754620\n"


class FakeResponse:
    """Stand-in for a requests response."""

    def __init__(self, content: bytes, status_code: int = 200):
        self.content = content
        self.status_code = status_code
        self.headers: dict[str, str] = {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size: int = 8192):
        for start in range(0, len(self.content), chunk_size):
            yield self.content[start : start + chunk_size]


class FakeSession:
    """Records requests and serves canned bodies keyed by file suffix."""

    def __init__(self, *, missing: set[str] | None = None):
        self.missing = missing or set()
        self.requested: list[str] = []

    def get(self, url: str, timeout=None, stream: bool = False):
        self.requested.append(url)
        for suffix in (".html", ".txt"):
            if url.endswith(suffix):
                if suffix in self.missing:
                    return FakeResponse(b"", status_code=404)
                body = DESCRIPTOR if suffix == ".html" else DATA
                return FakeResponse(body.encode())
        return FakeResponse(b"<table></table>")

    # The downloader may use itself as a context manager in future; keep parity
    # with requests.Session's surface for the calls actually made.
    def close(self):
        pass


@pytest.fixture
def session() -> FakeSession:
    return FakeSession()


@pytest.fixture
def downloader(tmp_path, session) -> NSDBDownloader:
    return NSDBDownloader(tmp_path / "raw", session=session)


# ---------------------------------------------------------------------------
# URL construction
# ---------------------------------------------------------------------------


class TestCatalogueFor:
    @pytest.mark.parametrize(
        ("identifier", "expected_fragment"),
        [
            ("nm0001", "/N/"),
            ("jm0001", "/J/"),
            ("sm0001", "/S/"),
            ("um0001", "/U/"),
            ("mm0001", "/M/"),
            ("pm0001", "/P/"),
            ("abin0001", "/abin/"),
        ],
    )
    def test_known_prefixes(self, identifier, expected_fragment):
        assert expected_fragment in NSDBDownloader.catalogue_for(identifier)

    def test_case_insensitive(self):
        assert NSDBDownloader.catalogue_for("NM0001") == NSDBDownloader.catalogue_for("nm0001")

    def test_unknown_prefix(self):
        with pytest.raises(ValueError, match="Unknown catalogue prefix"):
            NSDBDownloader.catalogue_for("xm0001")

    def test_empty_name(self):
        with pytest.raises(ValueError, match="must not be empty"):
            NSDBDownloader.catalogue_for("")

    def test_url_for(self):
        url = NSDBDownloader.url_for("nm0001", "html")

        assert url == "https://nsdb.imcce.fr/obspos/OBS_COLL/N/nm0001.html"

    def test_url_for_txt(self):
        assert NSDBDownloader.url_for("nm0001", "txt").endswith("nm0001.txt")

    def test_no_double_slash(self):
        """The catalogue URLs end in a slash, so joining needs care."""
        for name in ("nm0001", "abin0001"):
            for suffix in ("html", "txt"):
                assert "//" not in NSDBDownloader.url_for(name, suffix).removeprefix("https://")


# ---------------------------------------------------------------------------
# Downloading
# ---------------------------------------------------------------------------


class TestFetch:
    def test_downloads_both_files(self, downloader, session):
        files = downloader.fetch("nm0001")

        assert files.descriptor.name == "nm0001.html"
        assert files.data.name == "nm0001.txt"
        assert files.descriptor.exists()
        assert files.data.exists()
        assert files.downloaded is True
        assert len(session.requested) == 2

    def test_returns_paths_in_dataset_folder(self, downloader, tmp_path):
        files = downloader.fetch("nm0001")

        assert files.descriptor.parent == tmp_path / "raw" / "nm0001"

    def test_content_is_written(self, downloader):
        files = downloader.fetch("nm0001")

        assert "planet: 8 - Neptune" in files.descriptor.read_text(encoding="utf-8")
        assert "2001" in files.data.read_text(encoding="utf-8")

    def test_second_fetch_does_not_hit_the_network(self, downloader, session):
        downloader.fetch("nm0001")
        session.requested.clear()
        files = downloader.fetch("nm0001")

        assert session.requested == []
        assert files.downloaded is False

    def test_force_redownloads(self, downloader, session):
        downloader.fetch("nm0001")
        session.requested.clear()
        files = downloader.fetch("nm0001", force=True)

        assert len(session.requested) == 2
        assert files.downloaded is True

    def test_missing_file_raises_file_not_found(self, tmp_path):
        downloader = NSDBDownloader(tmp_path, session=FakeSession(missing={".txt"}))

        with pytest.raises(FileNotFoundError, match="no file at"):
            downloader.fetch("nm0001")

    def test_no_partial_file_is_left_behind(self, tmp_path):
        """An interrupted download must not leave a file a later run trusts."""
        downloader = NSDBDownloader(tmp_path, session=FakeSession(missing={".txt"}))

        with pytest.raises(FileNotFoundError):
            downloader.fetch("nm0001")

        folder = tmp_path / "nm0001"
        assert not (folder / "nm0001.txt").exists()
        assert not list(folder.glob("*.part"))

    def test_is_available_locally(self, downloader, tmp_path):
        assert downloader.is_available_locally("nm0001") is False
        downloader.fetch("nm0001")
        assert downloader.is_available_locally("nm0001") is True

    def test_fetch_many(self, downloader):
        files = downloader.fetch_many(["nm0001", "nm0002"])

        assert set(files) == {"nm0001", "nm0002"}
        assert all(isinstance(value, NSDBDatasetFiles) for value in files.values())

    def test_default_session_is_created_lazily(self, tmp_path):
        downloader = NSDBDownloader(tmp_path)

        assert downloader._session is None
        assert downloader.session is not None


class TestFetchAvailableDatasets:
    def test_parses_the_table(self, tmp_path):
        html = (
            "<table>"
            "<tr><td></td></tr>"
            "<tr><td>nm0001.html</td><td>2020</td><td>1k</td></tr>"
            "<tr><td>nm0001.txt</td><td>2020</td><td>2k</td></tr>"
            "<tr><td>nm0002.html</td><td>2021</td><td>1k</td></tr>"
            "</table>"
        )

        class TableSession(FakeSession):
            def get(self, url, timeout=None, stream=False):
                return FakeResponse(html.encode())

        downloader = NSDBDownloader(tmp_path, session=TableSession())

        assert downloader.fetch_available_datasets("Neptune") == ["nm0001", "nm0002"]

    def test_unknown_planet(self, tmp_path):
        downloader = NSDBDownloader(tmp_path, session=FakeSession())

        with pytest.raises(ValueError, match="Unknown planet"):
            downloader.fetch_available_datasets("Vulcan")

    def test_no_table(self, tmp_path):
        class EmptySession(FakeSession):
            def get(self, url, timeout=None, stream=False):
                return FakeResponse(b"<html></html>")

        downloader = NSDBDownloader(tmp_path, session=EmptySession())

        with pytest.raises(ValueError, match="No dataset table"):
            downloader.fetch_available_datasets("Neptune")


# ---------------------------------------------------------------------------
# Manager: loading
# ---------------------------------------------------------------------------


class TestManagerLoad:
    def test_load_returns_a_record(self, tmp_path, session):
        manager = NSDBManager(tmp_path, downloader=NSDBDownloader(tmp_path, session=session))

        record = manager.load("nm0001")

        assert record.identifier == "nm0001"
        assert record.planet == {"number": 8, "name": "Neptune"}

    def test_load_downloads_when_absent(self, tmp_path, session):
        manager = NSDBManager(tmp_path, downloader=NSDBDownloader(tmp_path, session=session))
        manager.load("nm0001")

        assert (tmp_path / "nm0001" / "nm0001.html").exists()

    def test_load_many(self, tmp_path, session):
        manager = NSDBManager(tmp_path, downloader=NSDBDownloader(tmp_path, session=session))

        records = manager.load_many(["nm0001", "nm0002"])

        assert set(records) == {"nm0001", "nm0002"}

    def test_default_construction(self, tmp_path):
        manager = NSDBManager(tmp_path)

        assert isinstance(manager.downloader, NSDBDownloader)
        assert isinstance(manager.parser, NSDBParser)

    def test_known_planets(self):
        assert "Neptune" in NSDBManager.known_planets()
        assert NSDBManager.known_planets() == sorted(NSDBManager.known_planets())


# ---------------------------------------------------------------------------
# Manager: YAML output
# ---------------------------------------------------------------------------


class TestWriteConfig:
    def test_writes_expected_filename(self, tmp_path, session):
        manager = NSDBManager(tmp_path, downloader=NSDBDownloader(tmp_path, session=session))
        destination = manager.write_config(manager.load("nm0001"), tmp_path / "configs")

        assert destination == tmp_path / "configs" / "nm0001.yaml"
        assert destination.exists()

    def test_config_contains_the_contract_keys(self, tmp_path, session):
        manager = NSDBManager(tmp_path, downloader=NSDBDownloader(tmp_path, session=session))
        destination = manager.write_config(manager.load("nm0001"), tmp_path / "configs")

        config = yaml.safe_load(destination.read_text(encoding="utf-8"))

        assert config["identifier"] == "nm0001"
        assert config["type"] == "absolute_absolute_CCD_nsdb"
        assert config["reference_frame"] == "astrometric"
        assert config["epoch_of_equinox"] == "J2000"
        assert config["observatory"][0]["code"] == 673
        assert config["format_columns"]["1"] == "Number of satellite (N sat)"

    def test_file_key_is_absolute(self, tmp_path, session):
        manager = NSDBManager(tmp_path, downloader=NSDBDownloader(tmp_path, session=session))
        destination = manager.write_config(manager.load("nm0001"), tmp_path / "configs")

        config = yaml.safe_load(destination.read_text(encoding="utf-8"))

        assert Path(config["file"]).is_absolute()
        assert config["file"].endswith("nm0001.txt")

    def test_accepts_an_identifier(self, tmp_path, session):
        """Passing a name instead of a record loads it, like the old API."""
        manager = NSDBManager(tmp_path, downloader=NSDBDownloader(tmp_path, session=session))
        destination = manager.write_config("nm0001", tmp_path / "configs")

        assert yaml.safe_load(destination.read_text(encoding="utf-8"))["identifier"] == "nm0001"

    def test_creates_the_output_directory(self, tmp_path, session):
        manager = NSDBManager(tmp_path, downloader=NSDBDownloader(tmp_path, session=session))
        manager.write_config(manager.load("nm0001"), tmp_path / "a" / "b")

        assert (tmp_path / "a" / "b" / "nm0001.yaml").exists()

    def test_explicit_data_file_wins(self, tmp_path, session):
        manager = NSDBManager(tmp_path, downloader=NSDBDownloader(tmp_path, session=session))
        custom = tmp_path / "elsewhere.txt"
        destination = manager.write_config(manager.load("nm0001"), tmp_path / "configs", custom)

        assert yaml.safe_load(destination.read_text(encoding="utf-8"))["file"] == str(
            custom.resolve()
        )


class TestWriteCollectionConfig:
    def test_references_each_dataset(self, tmp_path):
        manager = NSDBManager(tmp_path)
        destination = manager.write_collection_config(
            ["nm0001", "nm0002"], tmp_path / "collection", "conf/datasets/generated"
        )

        text = destination.read_text(encoding="utf-8")
        assert "defaults:" in text
        assert "/datasets/generated/nm0001.yaml@datasets.nm0001" in text
        assert "/datasets/generated/nm0002.yaml@datasets.nm0002" in text

    def test_strips_the_conf_prefix(self, tmp_path):
        """Hydra addresses configs relative to its search path, not the repo."""
        manager = NSDBManager(tmp_path)
        destination = manager.write_collection_config(["nm0001"], tmp_path, "conf/datasets/x")

        assert "conf/conf/" not in destination.read_text(encoding="utf-8")


class TestWriteAll:
    def test_writes_configs_and_collection(self, tmp_path, session):
        manager = NSDBManager(tmp_path, downloader=NSDBDownloader(tmp_path, session=session))

        written = manager.write_all(
            ["nm0001", "nm0002"], tmp_path / "configs", tmp_path / "collection"
        )

        assert set(written) == {"nm0001", "nm0002"}
        assert (tmp_path / "collection" / "collection.yaml").exists()
        for path in written.values():
            assert path.exists()


# ---------------------------------------------------------------------------
# Manager: Excel output
# ---------------------------------------------------------------------------


class TestExcelOutput:
    def test_records_to_dataframe_has_one_row_per_record(self, tmp_path, session):
        manager = NSDBManager(tmp_path, downloader=NSDBDownloader(tmp_path, session=session))
        records = manager.load_many(["nm0001", "nm0002"])

        frame = manager.records_to_dataframe(records.values())

        assert len(frame) == 2
        assert set(frame["identifier"]) == {"nm0001", "nm0002"}

    def test_summary_columns_are_useful(self, tmp_path, session):
        manager = NSDBManager(tmp_path, downloader=NSDBDownloader(tmp_path, session=session))
        frame = manager.records_to_dataframe([manager.load("nm0001")])

        for column in (
            "identifier",
            "type",
            "planet",
            "observatory_code",
            "reference_frame",
            "epoch_of_equinox",
            "time_scale",
            "n_format_columns",
        ):
            assert column in frame.columns

    def test_writes_both_sheets(self, tmp_path, session):
        manager = NSDBManager(tmp_path, downloader=NSDBDownloader(tmp_path, session=session))
        destination = manager.write_excel([manager.load("nm0001")], tmp_path / "out.xlsx")

        assert destination.exists()

        import openpyxl

        workbook = openpyxl.load_workbook(destination)
        assert set(workbook.sheetnames) == {"datasets", "attributes"}

    def test_attribute_sheet_can_be_omitted(self, tmp_path, session):
        manager = NSDBManager(tmp_path, downloader=NSDBDownloader(tmp_path, session=session))
        destination = manager.write_excel(
            [manager.load("nm0001")], tmp_path / "out.xlsx", attribute_sheet=False
        )

        import openpyxl

        assert openpyxl.load_workbook(destination).sheetnames == ["datasets"]

    def test_nested_values_are_serialised(self, tmp_path, session):
        """Excel cannot hold dicts or lists, so they must be rendered."""
        manager = NSDBManager(tmp_path, downloader=NSDBDownloader(tmp_path, session=session))
        destination = manager.write_excel_for(["nm0001"], tmp_path / "out.xlsx")

        import openpyxl

        workbook = openpyxl.load_workbook(destination)
        sheet = workbook["datasets"]
        headers = [cell.value for cell in sheet[1]]
        observatory_column = headers.index("observatory_name") + 1

        assert sheet.cell(row=2, column=observatory_column).value is not None


# ---------------------------------------------------------------------------
# Characteristics of the output across the real corpus
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not REAL_DESCRIPTORS, reason="data/raw has no NSDB descriptors")
class TestManagerAgainstRealData:
    def test_excel_for_the_whole_corpus(self, tmp_path):
        """The workbook must survive every real record's field shapes."""
        import openpyxl

        manager = NSDBManager(DATA_RAW.parent)
        names = [path.stem for path in REAL_DESCRIPTORS]
        records = [NSDBParser().parse_file(path) for path in REAL_DESCRIPTORS]
        destination = manager.write_excel(records, tmp_path / "corpus.xlsx")

        workbook = openpyxl.load_workbook(destination)
        assert workbook["datasets"].max_row == len(names) + 1
        assert workbook["attributes"].max_row > len(names)

    def test_dataframe_row_count(self):
        manager = NSDBManager(DATA_RAW.parent)
        records = [NSDBParser().parse_file(path) for path in REAL_DESCRIPTORS]

        assert len(manager.records_to_dataframe(records)) == len(REAL_DESCRIPTORS)

    def test_no_unparsed_observatory_lines_in_the_corpus(self):
        """A non-zero count here means prose reached the observatory field."""
        records = [NSDBParser().parse_file(path) for path in REAL_DESCRIPTORS]

        total = sum(len(record.unparsed_observatory_lines) for record in records)

        assert total == 0


# ---------------------------------------------------------------------------
# Logging behaviour
# ---------------------------------------------------------------------------


def test_skipping_a_download_is_logged(tmp_path, session, caplog):
    downloader = NSDBDownloader(tmp_path, session=session)
    downloader.fetch("nm0001")

    with caplog.at_level(logging.INFO):
        downloader.fetch("nm0001")

    assert any("already present" in message for message in caplog.messages)


def test_missing_informations_section_is_warned_about(caplog):
    """A descriptor with no Informations. heading must say so, not fail quietly."""
    without_heading = DESCRIPTOR.replace("Informations.\n", "")

    with caplog.at_level(logging.WARNING):
        record = NSDBParser().parse(without_heading, "nm0000")

    assert record.parameters["reference_frame"] == "astrometric"
    assert any("Informations." in message for message in caplog.messages)
