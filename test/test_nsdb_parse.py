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
    assert parsed.get("type") == "absolute_absolute_CCD_nsdb"
    assert parsed.get("dates", {}).get("start_year") == 2001
    assert parsed.get("dates", {}).get("end_year") == 2001

    assert parsed.get("observatory", [])[0].get("code") == 673

    # Informations fields
    assert parsed.get("reference_frame") == "astrometric"
    assert parsed.get("telescope") and "Reflector" in parsed.get("telescope")

    # reference and comments present as strings
    assert "reference" in parsed and "Communicated" in parsed["reference"]
    assert "comments" in parsed and "no information" in parsed["comments"]


MULTI_OBSERVATORY_HTML = """
<html>
  <head><title>nm0083</title></head>
  <body>
<pre>
Contents.
      planet: 8 - Neptune
  satellites: N1 - Triton : 10
total number: 10
        type: absolute
       dates: 1963-1990
 observatory: 083 - Golosseevo-Kiev
              188 - Majdanak

Reference.
   Yizhakevych, O. M.; Andruk, V. M.; Pakuliak, L. K. (2016)

Informations.
         relative to: absolute
     reference frame: astrometric
     centre of frame: topocentre
    epoch of equinox: J2000
          time scale: UTC
           reduction: Catalogue Tycho-2
         coordinates: absolute
    diff. refraction: no information
            receptor: photographic
           telescope: 083 - DL: The Tepfer Double Long-Focus Astrograph (D=40cm, F=550cm)
                      188 - Z6: The 60-cm Zeiss Reflector (D=60cm, Cassegrain F=750cm)
            observers: Kulyk I., Izakevich E.M., Shatokhina S
 data included in standard data file: no

Comments.
  1. This portion includes most observations published earlier in the portion nm0016.

Format.
  1. Year   of the date of observation
  2. Month  of the date of observation
  3. Day    of the date of observation with decimals
  4. Hour   of right ascension (alpha, h)
  5. Minute of right ascension (alpha, m)
  6. Second of right ascension with decimals (alpha, s)
  7. Degree of declination (delta, deg)
  8. Minute of declination (delta, '  )
  9. Second of declination with decimals (delta, '' )
 10. (O-C) in right ascension, (O-C)a (arcsec)
 11. (O-C) in declination, (O-C)d (arcsec)
 12. Exposure time (ExT, sec)
 13. Index of emulsion (I): 1 - ORWO ZU21, 2 - Agfa Astro, 3 - ORWO NP27
 14. Photographic magnitude (Bmag)
 15. Photographic magnitude R.M.S. error (e_Bmag)
 16. Right ascension R.M.S. error (e_RAs, arcsec)
 17. Declination R.M.S. error (e_DEs, arcsec)
 18. Number of reference stars (Nz)
 19. Telescope (T): 1 - DL, 2 - Z6
 20. Unique identifier of the plate (NPl)
</pre>
  </body>
</html>
"""


def test_parse_nsdb_multiple_observatories():
    mgr = NSDBManager()
    parsed = mgr._parse_contents_metadata(MULTI_OBSERVATORY_HTML, "nm0083")

    observatories = parsed["observatory"]
    assert isinstance(observatories, list)
    assert len(observatories) == 2

    # First observatory with its telescope attached
    assert observatories[0]["code"] == 83
    assert observatories[0]["name"] == "Golosseevo-Kiev"
    assert len(observatories[0]["telescopes"]) == 1
    assert observatories[0]["telescopes"][0]["name"] == "DL"

    # Second observatory with its telescope attached
    assert observatories[1]["code"] == 188
    assert observatories[1]["name"] == "Majdanak"
    assert len(observatories[1]["telescopes"]) == 1
    assert observatories[1]["telescopes"][0]["name"] == "Z6"

    # Telescope field parsed as a list of entries
    telescopes = parsed["telescope"]
    assert isinstance(telescopes, list)
    assert len(telescopes) == 2
    assert telescopes[0]["code"] == 83
    assert telescopes[1]["code"] == 188

    # Telescope index column maps data-column index -> observatory code
    assert parsed["telescope_index"] == {"1": 83, "2": 188}


OBSERVATORY_CODE_COLUMN_HTML = """
<html>
  <head><title>nm0019</title></head>
  <body>
<pre>
Contents.
      planet: 8 - Neptune
  satellites: N1-Triton : 1095
total number: 1095
        type: absolute
       dates: 2007-2009
 observatory: 327 - Peking Observatory, Xinglong Station
              337 - Sheshan, formerly Zo-Se

Reference.
    Qiao R. C., Zhang H. Y., Dourneau G., Yu Y., Yan D., Shen K. X. (2014)

Informations.
         relative to: absolute
     reference frame: astrometric
     centre of frame: topocentre
    epoch of equinox: J2000
          time scale: UTC
           reduction: UCAC2
         coordinates: absolute
    diff. refraction: no information
            receptor: CCD
           telescope: 1 - Reflector, D=156 cm (at the Sheshan Station, code 337),
                      2 - Reflector, D=100 cm (at the Xinglong Station, code 327),
                      3 - Reflector, D=216 cm (at the Xinglong Station, code 327),
           observers: Qiao R. C., Zhang H. Y. et al.
data included in standard data file: no

Comments.
   The 1.56 m telescope was used.

Format.
  1. Year of observation
  2. Month of observation
  3. Day of observation with decimals
  4. Hour   of right ascension (alpha, h)
  5. Minute of right ascension (alpha, m)
  6. Second of right ascension (alpha, s)
  7. Degree of declination (delta, deg)
  8. Minute of declination (delta, '  )
  9. Second of declination (delta, '' )
 10. Code of observatory (327 or 337).
</pre>
  </body>
</html>
"""


def test_parse_nsdb_observatory_code_column():
    mgr = NSDBManager()
    parsed = mgr._parse_contents_metadata(OBSERVATORY_CODE_COLUMN_HTML, "nm0019")

    observatories = parsed["observatory"]
    assert isinstance(observatories, list)
    assert len(observatories) == 2
    assert observatories[0]["code"] == 327
    assert observatories[1]["code"] == 337

    # Telescopes attached to their observatory by the code embedded in the description
    assert len(observatories[0]["telescopes"]) == 2  # Xinglong (327)
    assert len(observatories[1]["telescopes"]) == 1  # Sheshan (337)
    assert all(t["code"] == 327 for t in observatories[0]["telescopes"])
    assert all(t["code"] == 337 for t in observatories[1]["telescopes"])

    # Telescope entries carry the observatory code extracted from the description
    telescopes = parsed["telescope"]
    assert isinstance(telescopes, list)
    assert len(telescopes) == 3
    assert telescopes[0]["code"] == 337
    assert telescopes[1]["code"] == 327
    assert telescopes[2]["code"] == 327

    # The data column directly holds observatory codes -> identity mapping
    assert parsed["telescope_index"] == {"327": 327, "337": 337}


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
