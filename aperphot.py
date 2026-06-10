import asdf
import numpy as np
from photutils.aperture import aperture_photometry
from photutils.aperture import CircularAperture
from photutils.detection import DAOStarFinder
import glob
import pandas as pd

import warnings
from asdf import exceptions
warnings.filterwarnings("ignore", category=exceptions.AsdfWarning)

# AperOL2
from utils import build_lightcurves, extract_wcs, extract_image_data, extract_noise_data

def run_all_images(asdf_file_dir, asdf_glob, catalogs, method = "catalog"):

    asdf_images = glob.glob(f"{asdf_file_dir}/{asdf_glob}")

    catalog = pd.DataFrame()
    for cat in catalogs:
        new_cat = pd.read_csv(cat, comment="#", sep=",")
        catalog = pd.concat([catalog, new_cat])



    tables = []

    asdf_images = asdf_images[:3]

    ## From Thushara ##
    threshold = 0
    band = "F129"
    Fnum = 8
    pixsize = 10 # 10 micron
    midlambda = float(str(band)[1:]) / 100.
    sig_um = .42*midlambda*Fnum # illumination Gaussian sigma in microns
    sig = sig_um / pixsize # illumination sigma in pixel units
    psfwhm = 2.35*sig
    ####################

    for i, asdf_image in enumerate(asdf_images):

        print(f"Processing image {i+1}/{len(asdf_images)}")
        image_data = extract_image_data(asdf.open(asdf_image))
        image_noise, kind = extract_noise_data(asdf.open(asdf_image))
        wcs, wcs_type = extract_wcs(asdf.open(asdf_image))
        stars_x, stars_y = wcs.world_to_pixel_values(catalog["ra"], catalog["dec"])


        mask = np.where((stars_x >= 0) & (stars_x < image_data.shape[1]) & (stars_y >= 0) & (stars_y < image_data.shape[0]))
        stars_x = stars_x[mask]
        stars_y = stars_y[mask]

        catalog = catalog.iloc[mask]

        stars_x_catalog = stars_x
        stars_y_catalog = stars_y

        if method == "DAOStarFinder":

            print(f"Looking for {len(stars_x)} stars with DAOStarFinder...")

            finder = DAOStarFinder(threshold=threshold, fwhm=1.3*psfwhm, n_brightest = len(stars_x))
            finder_results = finder(image_data)
            stars_x = finder_results['x_centroid']
            stars_y = finder_results['y_centroid']

        print("Calculating BG using catalog star positions...")
        positions = np.transpose((stars_x_catalog, stars_y_catalog))

        aperture = CircularAperture(positions, r=5)
        master_mask = np.zeros(image_data.shape, dtype=bool)
        for ap in aperture:
            master_mask += ap.to_mask(method="exact").to_image(image_data.shape) > 0

        positions = np.transpose((stars_x, stars_y))

        from photutils.background import Background2D

        bg_calc = Background2D(image_data, (50, 50), filter_size=(3, 3), mask = master_mask).background

        image_data -= bg_calc

        phot_table = aperture_photometry(image_data, aperture, error = image_noise)
        phot_table['aperture_sum'].info.format = '%.8g'  # for consistent table output

        phot_table['ra'] = catalog['ra']
        phot_table['dec'] = catalog['dec']
        phot_table['flux'] = phot_table['aperture_sum']
        phot_table['flux_err'] = phot_table['aperture_sum_err']

        if method == "DAOStarFinder":

            fit_x = phot_table["x_center"]
            fit_y = phot_table["y_center"]
            aperture_fluxes = []

            for star in catalog.itertuples():
                star_x, star_y = wcs.world_to_pixel_values(star.ra, star.dec)
                x_dist = np.abs(fit_x - star_x)
                y_dist = np.abs(fit_y - star_y)
                total_dist = np.sqrt(x_dist**2 + y_dist**2)
                closest_index = np.argmin(total_dist)
                aperture_fluxes.append(phot_table['aperture_sum'][closest_index])
                if total_dist[closest_index] > 1:  # If the closest star is more than 1 pixels away, print a warning
                    print(f"Warning: Closest star to GAIA source {star.Index} is {total_dist[closest_index]:.2f} pixels away.")

        else:
            aperture_fluxes = phot_table['aperture_sum']

        catalog_fluxes = catalog[band]
        aperture_fluxes = np.array(aperture_fluxes)

        catalog_mags = -2.5 * np.log10(catalog_fluxes)
        aperture_mags = -2.5 * np.log10(aperture_fluxes)
        aperture_mag_errs = 2.5 * phot_table['aperture_sum_err'] / (aperture_fluxes * np.log(10))

        tables.append(phot_table)

    mjd_time_objects = [i.tree['roman']['meta']['exposure']['mid_time'] for i in map(asdf.open, asdf_images)]

    mjds = [t.to_value('mjd') for t in mjd_time_objects]
    print(tables[0])

    lightcurve_dict = build_lightcurves(tables, mjds)

    np.save("lightcurve_dict.npy", lightcurve_dict)

if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--asdf_file_dir", help="Path to the directory containing ASDF files")
    argparser.add_argument("--asdf_glob", help="Glob pattern for ASDF files", default = "*.asdf")
    argparser.add_argument("--catalogs",nargs="+", help="List of catalog files to use" )
    argparser.add_argument("--method", help="Method to use for star finding."
                           " Catalog or DAOStarFinder", default = "catalog")
    args = argparser.parse_args()
    run_all_images(args.asdf_file_dir, args.asdf_glob, args.catalogs)



# all_pulls = []
# for i in lightcurve_dict.keys():
#     mjds = lightcurve_dict[i]['mjd']
#     fluxes = lightcurve_dict[i]['flux']
#     flux_errs = lightcurve_dict[i]['flux_err']
#     pulls = (fluxes - np.mean(fluxes)) / flux_errs
#     all_pulls.extend(pulls)


# plt.hist(all_pulls, bins=30, density=True)

# # Overplot 1 sigma gaussian
# x = np.linspace(-5, 5, 100)
# plt.plot(x, 1/np.sqrt(2*np.pi)*np.exp(-0.5*x**2), label="1-sigma Gaussian")
# plt.xlabel("Pull (normalized residual)")
# plt.ylabel("Density")
# plt.title("Pull Distribution for All Stars")

# # Add a textbox with mean and sigma of pull
# mean_pull = np.nanmean(all_pulls)
# std_pull = np.nanstd(all_pulls)
# textstr = f"Mean: {mean_pull:.2f}\nSigma: {std_pull:.2f}"
# plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

# plt.legend()

# plt.savefig("lcs.png")
# print("saved")
