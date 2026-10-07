# CenSat — Central and Satellite Galaxy Identifier

**Author:** M. Reza Ayromlou (ayromlou@uni-bonn.de; ayromlou@gmail.com)

CenSat identifies **central** and **satellite** galaxies in simulation mock catalogues and observational data. It classifies galaxies in order of decreasing stellar mass:

- The most massive galaxy is assigned as the first **central**.
- Unclassified galaxies within `MaxScale · R200` in projected separation and `MaxVel · V200` in line-of-sight velocity difference are assigned as **satellites**. These are independent cuts, with defaults `MaxScale=1.5` and `MaxVel=2.0`.
- The next most massive *unassigned* galaxy is taken as a new central, and the process repeats until every galaxy carries a label.

Assigned labels are retained throughout the search. Halo quantities (`M200`, `R200`, `V200`) are estimated internally from stellar mass through a fixed stellar-to-halo-mass relation, so the main entry points require no halo catalogue.

If you use CenSat in your work, please cite **[Ayromlou et al. 2023a](https://ui.adsabs.harvard.edu/abs/2023MNRAS.519.1913A/abstract)**.

---

## Contents


| File                                         | Description                                                                                            |
| ---------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| [`CenSat.py`](CenSat.py)                     | Core algorithm and helper functions.                                                                   |
| [`run_CenSat.py`](run_CenSat.py)             | Synthetic demo and templates for using your own arrays.                                                |
| [`test_censat_unit.py`](test_censat_unit.py) | Regression tests with in-memory inputs and outputs.                                                    |
| [`test_censat.py`](test_censat.py)           | Example driver for TNG100 and SDSS, including a comparison with simulation labels and a fraction plot. |
| [`__init__.py`](__init__.py)                 | Package initialiser.                                                                                   |
| [`requirements.txt`](requirements.txt)       | Python dependencies.                                                                                   |

---

## Algorithm overview

CenSat uses projected separation and a line-of-sight velocity cut:

- `CenSat_Simulations` projects positions onto the `x-y` plane and uses the `z` velocity component, including Hubble flow and periodic boundaries.
- `CenSat_Observations` embeds RA/Dec directions on a unit sphere, then measures their chord separation at the candidate central's angular-diameter distance. At small angles this is `theta · D_A`. The grid has no periodic boundaries.
- `CenSat_Observations_xyz` is the lower-level backend for the same projected sky calculation. Its Cartesian coordinates are shifted, scaled sky-direction vectors; their three components do not represent physical radial positions in a survey volume.

Galaxies are binned on a regular grid to restrict each search to nearby cells. `numbins` controls grid resolution and computational cost. The final output uses:

- `type = 0` for **central**
- `type = 1` for **satellite**

---

## Requirements


| Package      | Notes                                                                   |
| -------------- | ------------------------------------------------------------------------- |
| Python       | Use a version supported by the dependencies below.                      |
| `numpy`      | Arrays and numerical calculations.                                      |
| `h5py`       | Catalogue input and HDF5 output.                                        |
| `astropy`    | Planck15 cosmological distances for observations.                       |
| `scipy`      | Numerical integration used by Astropy's Planck15 distance calculations. |
| `matplotlib` | Fraction plots in`test_censat.py`.                                      |

Install with:

```bash
pip install -r requirements.txt
```

---

## Quick Start

Run the synthetic demo to check the installation:

```bash
python -B run_CenSat.py
```

This runs a random catalogue through both the simulation and observation paths. It writes:

- `censat_simulation_example.hdf5`
- `censat_observations_radec_example.hdf5`

Each file contains `/Galaxy/type` as `int64` labels. Only rows satisfying `LimPar > LimPar_min` are returned and written, in their original input order. Reconstruct this selection mask when matching labels to your catalogue. An existing output file with the same name is overwritten.

### Using your own arrays

`run_CenSat.py` also contains three template functions. Replace the placeholder arrays in the appropriate function, then invoke that function explicitly:

```bash
python -B -c "from run_CenSat import example_simulation_main; example_simulation_main()"
python -B -c "from run_CenSat import example_observations_ra_dec; example_observations_ra_dec()"
python -B -c "from run_CenSat import example_observations_xyz; example_observations_xyz()"
```

Running `python run_CenSat.py` executes the synthetic demo. You can also import classifiers directly from `CenSat`.

### Input units and shapes

All arrays supplied to one classifier must describe the same rows. Use finite coordinates, masses, and velocities for the selected galaxies.


| Input        | Simulation                                                                                                     | RA/Dec observations                                                                   |
| -------------- | ---------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| Coordinates  | `(N, 3)` positions in Mpc/h, within `[0, boxsize]`.                                                            | `(N,)` RA/Dec in degrees; set `ra_dec_units='radian'` for radians.                    |
| Velocity     | `(N, 3)` peculiar velocities in km/s.                                                                          | `(N,)` observed LOS velocity in km/s, approximately `3e5 * redshift` at low redshift. |
| Stellar mass | `(N,)` linear mass in `1e10 Msun/h`.                                                                           | `(N,)` `log10(Mstar/Msun)`.                                                           |
| Redshift     | Use a snapshot at approximately z=0;`snapnum` is retained for compatibility and does not change the cosmology. | `(N,)` redshifts giving positive angular-diameter distances for selected galaxies.    |
| Selection    | `(N,)` `LimPar`; select strictly `LimPar > LimPar_min`.                                                        | Same.                                                                                 |

The halo helper returns `M200` in `1e10 Msun/h`, `R200` in **proper Mpc/h**, and `V200` in km/s. Simulation positions, box length, and radii must use the same h convention. At z=0, proper and comoving simulation coordinates coincide.

For observations, Astropy's `Planck15.angular_diameter_distance` returns proper Mpc. The wrapper multiplies this by `_H0=0.673` to express distances in Mpc/h, using the same h as the observational stellar-mass conversion. Planck15 supplies the distance-redshift cosmology; `_H0` supplies the mass and distance unit convention.

For direct calls to `CenSat_Observations_xyz`, provide shifted, scaled unit-sphere coordinates and their positive scale factor `r_dummy_scale`. Supply `galdata1_dis` as proper angular-diameter distances in Mpc/h and `galdata1_Rvir` in the same units and h convention. The separation is `norm(delta_position) * central_distance / r_dummy_scale`. `boxsize` is the dummy grid extent, typically 500, rather than a physical survey size. See the function docstring for the remaining arrays.

With `nRvirOrMpc='nRvir'`, `MaxScale` multiplies `R200`; with `'Mpc'`, it is a fixed radius in Mpc/h. With `nVvirORkm='nVvir'`, `MaxVel` multiplies `V200`; with `'km'`, it is a fixed cut in km/s. The simulation search radius, including the largest `MaxScale * R200`, must not exceed half the periodic box length.

---

## Testing

Run the regression tests without external data or filesystem outputs:

```bash
python -B -m unittest -v test_censat_unit
```

The tests cover unit conversions, proper projected distances, grid boundaries and search coverage, agreement with a brute-force search, retained classifications, periodic simulations, output row order, and native TNG catalogue indices.

`test_censat.py` is a separate example driver using real catalogues. **Input catalogues are not shipped with the repository.** Give their paths on the command line:

```bash
python -B test_censat.py --tng100 /path/to/TNG100-1/output --sdss /path/to/SDSS_DR7.hdf5
```

Or use environment variables:

```bash
export CENSAT_TNG100_FILE=/path/to/TNG100-1/output
export CENSAT_SDSS_FILE=/path/to/SDSS_DR7.hdf5
python -B test_censat.py
```

A missing path is skipped with a message. If neither catalogue is available, the driver produces no outputs. Otherwise it writes `test_censat_fractions.png` and, for each available catalogue, `test_censat_tng100.hdf5` or `test_censat_sdss.hdf5`.

### Native TNG100 input

Point `--tng100` at the native output directory or its `groups_099` directory. The reader loads all snapshot 099 chunks in numeric order. It reads box length and h from the header, converts positions from ckpc/h to Mpc/h at z=0, and selects `log10(Mstar/Msun) > 9.5`. It excludes `SubhaloFlag=False` objects by default; use `--include-subhalo-clumps` to include them.

Simulation comparison labels come from `Group/GroupFirstSub`: these are global subhalo indices, and negative entries indicate groups without a subhalo. The returned `run_tng100(...)` dictionary includes `subhalo_id` aligned with the output labels, preserving the selected global input order. The plot converts stellar mass to Msun using the same header h as the mass selection.

### Prepared simulation and observational files

For a single prepared L-Galaxies/TNG/EAGLE-like HDF5 file, convert your fields to the following schema before using this driver:


| Dataset            | Required content                                                |
| -------------------- | ----------------------------------------------------------------- |
| `/SubhaloPos`      | `(N, 3)` z=0 positions in kpc/h; the driver divides by 1000.    |
| `/SubhaloVel`      | `(N, 3)` peculiar velocities in km/s.                           |
| `/SubhaloMassType` | Per-type masses in`1e10 Msun/h`, with stellar mass in column 4. |
| `/SubhaloType`     | `(N,)` comparison labels: 0=central, 1=satellite.               |

Optional `/Header` attributes `BoxSize` (kpc/h) and `HubbleParam` supply the box length and h. Without them the driver defaults to 75 Mpc/h and h=0.673; use `--boxsize` (Mpc/h) and `--hubble` for different catalogues. An optional `/SubhaloFlag` is filtered as in native TNG. This prepared schema is a common input format, and individual simulations may require field and unit conversions.

An SDSS-like file must contain `/Galaxy/ra`, `/Galaxy/dec`, `/Galaxy/redshift`, and `/Galaxy/StellarMass_median_log`, as `(N,)` arrays in degrees, degrees, dimensionless redshift, and `log10(Mstar/Msun)`, respectively. The driver selects `0 < z <= 0.04` and `log10(Mstar/Msun) > 9.5`, and uses observed LOS velocities approximated by `cz`.

---

## Scientific assumptions

The stellar-to-halo relation uses the fixed broken-power-law L-Galaxies fit in `CenSat.py`. Halo radii use a fixed z=0 critical density and overdensity 200; the halo calibration does not evolve with redshift. The simulation velocity conversion uses `100 * delta_z` for separations in Mpc/h, appropriate at approximately z=0. The observation path uses low-redshift observed velocity differences.

These assumptions match the low-redshift use illustrated here. Applying CenSat to another redshift range or a different stellar-mass calibration requires assessing the halo relation, cosmology, and velocity conversion for that sample.

---

## Contact

M. Reza Ayromlou — ayromlou@uni-bonn.de; ayromlou@gmail.com

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for the full license text.
