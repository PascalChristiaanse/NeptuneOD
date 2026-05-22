from pathlib import Path

import yaml

from orbitdet.data.nsdb import NSDBManager

SAMPLE_HTML = """
<html>
  <head><title>incorrect-title</title></head>
  <body>
<pre>
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

Comments.
                      no information

Format.
  1. Number of satellite (N sat)
  2. Year   of the moment of observation
  3. Month  of the moment of observation
  4. Day    of the moment of observation with decimals
  5. Hour   of right ascension (alpha, h)
  6. Minute of right ascension (alpha, m)
  7. Second of right ascension (alpha, s)
  8. Degree of declination (delta, deg)
  9. Minute of declination (delta, '  )
 10. Second of declination (delta, '' )
</pre>
  </body>
</html>
"""


def test_parse_nsdb_sample():
    mgr = NSDBManager()
    parsed = mgr._parse_contents_metadata(SAMPLE_HTML, "nm0001")

    assert parsed["identifier"] == "nm0001"
    assert "format_columns" in parsed
    assert len(parsed["format_columns"]) == 10

    # Contents parsed fields
    assert "planet" in parsed
    assert parsed["planet"]["number"] == 8
    assert parsed["planet"]["name"] == "Neptune"

    assert parsed.get("total_number") == 3
    assert parsed.get("type") == "absolute_CCD_nsdb"
    assert parsed.get("dates", {}).get("start_year") == 2001
    assert parsed.get("dates", {}).get("end_year") == 2001

    assert parsed.get("observatory", {}).get("code") == 673

    # Informations fields
    assert parsed.get("reference_frame") == "astrometric"
    assert parsed.get("telescope") and "Reflector" in parsed.get("telescope")

    # reference and comments present as strings
    assert "reference" in parsed and "Communicated" in parsed["reference"]
    assert "comments" in parsed and "no information" in parsed["comments"]


def test_generate_hydra_configs_adds_resolved_file_path(tmp_path: Path):
    mgr = NSDBManager()
    parsed = mgr._parse_contents_metadata(SAMPLE_HTML, "nm0001")
    data_file = tmp_path / "data" / "nm0001.txt"
    data_file.parent.mkdir(parents=True)
    data_file.write_text("sample data", encoding="utf-8")

    mgr.generate_hydra_configs(parsed, tmp_path, data_file)

    generated = yaml.safe_load((tmp_path / "nm0001.yaml").read_text(encoding="utf-8"))

    assert generated["identifier"] == "nm0001"
    assert generated["file"] == str(data_file.resolve())
