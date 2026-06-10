

singularity run --overlay pit_overlay /data/snpit/roman-snpit-env-cpu-0.1.36.sif \
    bash -c "
        python -m virtualenv new_venv &&
        ls new_venv/bin/python &&
        source new_venv/bin/activate &&
        python aperol/aperphot.py --asdf_file_dir /home/cfmeldorf/data/rick_romanisim_copy/output_images_galid_force/ --asdf_glob "*L2*.asdf" --catalogs "/home/cfmeldorf/data/rick_romanisim_copy/STARS_SYN.csv" "/home/cfmeldorf/data/rick_romanisim_copy/STARS_GAIA.csv"
    "