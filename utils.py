
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
from collections import defaultdict


# Cole's note:
# Since these functions are all largely non-scientific, these are mainly AI code that I have modified.
# They only deal with reading and managing data.

def build_lightcurves(
    tables,
    mjds,
    ra_col='ra',
    dec_col='dec',
    flux_col='flux',
    flux_err_col = 'flux_err',
    match_radius_arcsec=0.5,
):
    """
    Cross-match single-epoch source catalogs and assemble per-star light curves.

    The first table is used as the *reference catalog*. Stars in all subsequent
    tables are matched back to it by sky position. Stars that are detected in
    later epochs but not in the first table will NOT appear in the output —
    see the note below if that's a concern.

    Parameters
    ----------
    tables : list of astropy.table.Table
        One table per epoch. Each must contain RA, DEC, and flux columns.
        All tables should use the same column names.
    mjds : list of float
        MJD timestamp corresponding to each table, in the same order.
    ra_col : str
        Name of the RA column (assumed to be in degrees).
    dec_col : str
        Name of the Dec column (assumed to be in degrees).
    flux_col : str
        Name of the flux column.
    match_radius_arcsec : float
        Maximum on-sky angular separation (in arcseconds) to consider two
        detections the same star. Tune this to your pixel scale / astrometric
        precision. Default is 0.5 arcsec.

    Returns
    -------
    lightcurves : dict
        Keys are integer star IDs (= row index in the reference/first table).
        Values are dicts with four entries:
            'ra'   : float      — reference RA  (degrees)
            'dec'  : float      — reference Dec (degrees)
            'mjd'  : np.ndarray — observation times (sorted ascending)
            'flux' : np.ndarray — measured flux at each time

        Stars without detections in a given epoch simply have no entry for
        that epoch — arrays may be shorter than len(tables).

    Notes
    -----
    Reference catalog choice:
        Using the first table as the reference is convenient but not always
        optimal. If one epoch is significantly deeper (more detected stars),
        consider passing it first, or sorting `tables` and `mjds` together by
        depth beforehand.

    Stars not in the reference:
        Any star that appears in a later epoch but was NOT detected in epoch 0
        will be silently dropped. If you need to capture those too, you could
        run this function a second time with a combined/stacked reference
        catalog as tables[0].

    Duplicate matches:
        If two catalog sources in the same epoch both fall within
        `match_radius_arcsec` of the same reference star, only the closer one
        is kept.

    Examples
    --------
    >>> from astropy.table import Table
    >>> import numpy as np
    >>> t1 = Table({'ra': [10.0, 20.0], 'dec': [5.0, 6.0], 'flux': [100.0, 200.0]})
    >>> t2 = Table({'ra': [10.0001, 20.0], 'dec': [5.0, 6.0001], 'flux': [110.0, 195.0]})
    >>> lcs = build_lightcurves([t1, t2], mjds=[59000.0, 59001.0])
    >>> lcs[0]
    {'ra': 10.0, 'dec': 5.0, 'mjd': array([59000., 59001.]), 'flux': array([100., 110.])}
    """
    if len(tables) != len(mjds):
        raise ValueError(
            f"len(tables)={len(tables)} does not match len(mjds)={len(mjds)}."
        )
    if len(tables) == 0:
        return {}

    # ------------------------------------------------------------------ #
    # Step 1: Build the reference SkyCoord from the first table           #
    # ------------------------------------------------------------------ #
    ref_table = tables[0]

    ref_coords = SkyCoord(
        ra=ref_table[ra_col] * u.deg,
        dec=ref_table[dec_col] * u.deg,
    )

    # Storage: star_id (= row index in ref_table) -> lists of times and fluxes
    # defaultdict is like a regular dict except accessing a missing key
    # automatically creates an empty entry rather than raising a KeyError.
    raw = defaultdict(lambda: {'mjd': [], 'flux': [], 'flux_err': []})

    # Seed with the reference epoch's own detections
    for i in range(len(ref_table)):
        raw[i]['mjd'].append(float(mjds[0]))
        raw[i]['flux'].append(float(ref_table[flux_col][i]))
        raw[i]['flux_err'].append(float(ref_table[flux_err_col][i]))

    match_radius = match_radius_arcsec * u.arcsec

    # ------------------------------------------------------------------ #
    # Step 2: Match every subsequent epoch to the reference catalog       #
    # ------------------------------------------------------------------ #
    for table, mjd in zip(tables[1:], mjds[1:]):
        if len(table) == 0:
            continue

        cat_coords = SkyCoord(
            ra=table[ra_col] * u.deg,
            dec=table[dec_col] * u.deg,
        )

        # match_to_catalog_sky: for each source in `cat_coords`, find its
        # nearest neighbor in `ref_coords`.
        #   idx[i]   = index into ref_coords of the closest match to cat_coords[i]
        #   sep2d[i] = angular separation to that match
        idx, sep2d, _ = cat_coords.match_to_catalog_sky(ref_coords)

        # Resolve duplicates: if two catalog sources both claim the same
        # reference star, keep only the one with the smaller separation.
        # `best` maps ref_star_id -> (catalog_row_index, separation)
        best = {}
        for cat_row, (ref_id, sep) in enumerate(zip(idx, sep2d)):
            if sep > match_radius:
                continue  # too far away — not the same star
            if ref_id not in best or sep < best[ref_id][1]:
                best[ref_id] = (cat_row, sep)

        # Record the matched detections
        for ref_id, (cat_row, _) in best.items():
            raw[ref_id]['mjd'].append(float(mjd))
            raw[ref_id]['flux'].append(float(table[flux_col][cat_row]))
            raw[ref_id]['flux_err'].append(float(table[flux_err_col][cat_row]))

    # ------------------------------------------------------------------ #
    # Step 3: Package into final output dict, sorted by time              #
    # ------------------------------------------------------------------ #
    lightcurves = {}
    for star_id, data in raw.items():
        order = np.argsort(data['mjd'])
        lightcurves[star_id] = {
            'ra':   float(ref_table[ra_col][star_id]),
            'dec':  float(ref_table[dec_col][star_id]),
            'mjd':  np.array(data['mjd'])[order],
            'flux': np.array(data['flux'])[order],
            'flux_err': np.array(data['flux_err'])[order],
        }

    return lightcurves

############


def extract_image_data(af):
    """
    Return the primary 2-D image array from an ASDF file.
    Searches common keys; falls back to the first large ndarray found.
    """
    tree = af.tree

    for path in [
        ("data",),
        ("image",),
        ("sci",),
        ("roman", "data"),
        ("meta", "data"),
    ]:
        obj = tree
        try:
            for key in path:
                obj = obj[key]
            arr = np.asarray(obj)
            if arr.ndim >= 2:
                #print(f"[info] Found image data at tree path: {' → '.join(path)}")
                return arr.squeeze()
        except (KeyError, TypeError):
            pass

    # Walk the top-level tree for the first large ndarray
    for key, val in tree.items():
        try:
            arr = np.asarray(val)
            if arr.ndim >= 2 and arr.size > 100:
                #print(f"[info] Using first large array found at key '{key}'.")
                return arr.squeeze()
        except Exception:
            pass

    raise RuntimeError(
        "Could not locate image data in the ASDF file.\n"
        "Common keys tried: data, image, sci, roman.data.\n"
        "Please inspect af.tree and pass the data key explicitly."
    )

def extract_noise_data(af, prefer: str = "err"):
    """
    Return the noise/uncertainty image array from an ASDF file.

    Searches common tree paths in priority order. The `prefer` argument
    lets you bias toward a specific noise type when multiple are present:
      - "err"      → error (std dev, same units as science data)   [default]
      - "var"      → variance (err², pick the first var_* found)
      - "weight"   → inverse-variance weight map

    Parameters
    ----------
    af     : open asdf.AsdfFile object
    prefer : "err" | "var" | "weight"

    Returns
    -------
    noise : np.ndarray (2-D, squeezed)
    kind  : str describing what was found (e.g. "err", "var_rnoise", "wht")
    """
    tree = af.tree

    # Candidate paths grouped by noise type.
    # Checked in order; first match wins within each group.
    candidates = {
        "err": [
            ("err",),
            ("error",),
            ("noise",),
            ("rms",),
            ("uncertainty",),
            ("meta", "err"),
            ("roman", "err"),
        ],
        "var": [
            ("var_rnoise",),          # JWST read-noise variance
            ("var_flat",),            # JWST flat-field variance
            ("var_poisson",),         # JWST Poisson variance
            ("variance",),
            ("var",),
            ("roman", "var_rnoise"),
            ("roman", "var_flat"),
            ("roman", "var_poisson"),
        ],
        "weight": [
            ("wht",),
            ("weight",),
            ("ivar",),
            ("inv_variance",),
        ],
    }

    # Build search order: preferred group first, then the rest
    order = [prefer] + [k for k in candidates if k != prefer]

    for group in order:
        for path in candidates[group]:
            obj = tree
            try:
                for key in path:
                    obj = obj[key]
                arr = np.asarray(obj).squeeze()
                if arr.ndim >= 2 and arr.size > 1:
                    kind = path[-1]   # e.g. "err", "var_rnoise", "wht"
                    print(f"[info] Found noise data ('{kind}') at tree path: "
                          f"{' → '.join(path)}")
                    return arr, kind
            except (KeyError, TypeError):
                pass

    # Last resort: walk top-level keys for anything noise-like by name
    noise_hints = {"err", "noise", "rms", "var", "sigma", "wht", "weight", "uncertainty"}
    for key, val in tree.items():
        if any(hint in key.lower() for hint in noise_hints):
            try:
                arr = np.asarray(val).squeeze()
                if arr.ndim >= 2 and arr.size > 1:
                    print(f"[info] Found noise-like array at top-level key '{key}'.")
                    return arr, key
            except Exception:
                pass

    # if sqrt_on_fail:
    #     print("[warning] No noise array found; attempting to use sqrt of image data as a last resort.")
    #     try:
    #         image_data = extract_image_data(af)
    #         if np.all(image_data >= 0):
    #             return np.sqrt(image_data), "sqrt(image)"
    #         else:
    #             print("[warning] Image data contains negative values; cannot take sqrt.")
    #     except Exception:
    #         print("[warning] Failed to extract image data for sqrt fallback.")


    raise RuntimeError(
        "Could not locate a noise/error/variance image in the ASDF file.\n"
        "Common keys tried: err, var_rnoise, var_flat, var_poisson, wht, weight.\n"
        "Inspect af.tree and pass the key explicitly if needed."
        "Here is the tree for reference:\n" + str(tree)
    )

def extract_wcs(af):
    """
    Try several common locations where WCS information lives in ASDF files.

    Priority order:
      1. af['meta']['wcs']          – JWST / Roman pipeline products
      2. af['wcs']                  – simple convention
      3. af['meta']['wcsinfo']      – header-keyword style dict
      4. Reconstruct from FITS WCS keywords stored under meta
    """
    tree = af.tree

    # --- 1. gWCS object -------------------------------------------------------
    for path in [("meta", "wcs"), ("wcs",), ("roman", "meta", "wcs")]:
        obj = tree
        try:
            for key in path:
                obj = obj[key]
            if obj is not None:
                #print(f"[info] Found WCS at tree path: {' → '.join(path)}")
                return obj, "gwcs"
        except (KeyError, TypeError):
            pass

    # --- 2. FITS-header-style keywords ----------------------------------------
    header_sources = []
    for path in [("meta", "wcsinfo"), ("meta", "fits_header"), ("header",)]:
        obj = tree
        try:
            for key in path:
                obj = obj[key]
            if obj is not None:
                header_sources.append(obj)
        except (KeyError, TypeError):
            pass

    for src in header_sources:
        try:
            from astropy.io.fits import Header
            if isinstance(src, dict):
                hdr = Header(src.items())
            else:
                hdr = Header(dict(src).items())
            wcs = WCS(hdr)
            if wcs.has_celestial:
                print("[info] Reconstructed FITS WCS from header keywords.")
                return wcs, "fits"
        except Exception:
            pass

    raise RuntimeError(
        "Could not locate WCS information in the ASDF file.\n"
        "Common locations tried: meta.wcs, wcs, meta.wcsinfo, meta.fits_header.\n"
        "Please inspect af.tree and pass the WCS path explicitly."
    )
