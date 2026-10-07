"""
Example usage for CenSat on simulations and observations.

The three functions below are templates: replace the placeholder input arrays with your own catalogue fields. Running this file directly executes a small synthetic demo so you can verify the installation end to end.
"""

import numpy as np

try:
    from .CenSat import (
        CenSat_Simulations,
        CenSat_Observations,
        CenSat_Observations_xyz,
    )
except ImportError:
    from CenSat import (
        CenSat_Simulations,
        CenSat_Observations,
        CenSat_Observations_xyz,
    )


def example_simulation_main():
    """
    Example: mock catalogues -> CenSat_Simulations.
    Uses nRvir and nVvir.
    """
    nRvirOrMpc = "nRvir"
    MaxScale = 1.5
    nVvirORkm = "nVvir"
    MaxVel = 2
    h_const = 0.673
    LimPar_min = 10 ** (-0.5) * h_const  # Unit: 1e10 Msun/h
    numbins = 100
    boxsize = 100.0  # Mpc/h
    snapnum = 99

    # Replace these placeholders with your catalog arrays.
    # Shapes: pos (N,3), vel (N,3), Mstar (N,) in 1e10 Msun/h, LimPar (N,)
    galdata1_pos = np.zeros((0, 3))
    galdata1_vel = np.zeros((0, 3))
    galdata1_Mstar = np.zeros(0)
    LimPar = np.zeros(0)

    filename = "censat_simulation_example"

    return CenSat_Simulations(galdata1_pos, galdata1_vel, galdata1_Mstar, snapnum, LimPar, numbins, boxsize, MaxScale, nRvirOrMpc, MaxVel, nVvirORkm, LimPar_min, filename)


def example_observations_ra_dec():
    """
    Example: observations with RA/Dec (SDSS-like configuration).
    """
    numbins = 25
    MaxScale = 1.5
    nRvirOrMpc = "nRvir"
    MaxVel = 2
    nVvirORkm = "nVvir"
    LimPar_min = 9.5  # Unit: log10(Msun)

    # Replace these placeholders with your catalog arrays.
    galdata1_ra = np.zeros(0)
    galdata1_dec = np.zeros(0)
    galdata1_vel = np.zeros(0)
    galdata1_redshift = np.zeros(0)
    galdata1_Mstar = np.zeros(0)          # log10(Msun)
    LimPar = np.zeros(0)

    filename = "censat_observations_radec_example"

    return CenSat_Observations(galdata1_ra, galdata1_dec, galdata1_vel, galdata1_redshift, galdata1_Mstar, LimPar, numbins, MaxScale, nRvirOrMpc, MaxVel, nVvirORkm, LimPar_min, filename)


def example_observations_xyz():
    """
    Example: observations with scaled sky directions and pre-computed halo properties.
    """
    numbins = 25
    boxsize = 500
    MaxScale = 1.5
    nRvirOrMpc = "nRvir"
    MaxVel = 2
    nVvirORkm = "nVvir"
    LimPar_min = 9.5  # Unit: log10(Msun)

    # Replace these placeholders with your catalog arrays.
    galdata1_pos = np.zeros((0, 3))
    galdata1_vel = np.zeros(0)
    galdata1_dis = np.zeros(0)
    galdata1_Rvir = np.zeros(0)    # must be pre-computed when calling _xyz directly
    galdata1_Vvir = np.zeros(0)
    galdata1_Mvir = np.zeros(0)
    LimPar = np.zeros(0)

    r_dummy_scale = 1.0
    filename = "censat_observations_xyz_example"

    return CenSat_Observations_xyz(galdata1_pos, galdata1_vel, galdata1_dis, r_dummy_scale, galdata1_Rvir, galdata1_Vvir, galdata1_Mvir, LimPar, numbins, boxsize, MaxScale, nRvirOrMpc, MaxVel, nVvirORkm, LimPar_min, filename)


def _synthetic_demo():
    """Run both entry points on random data to verify the installation."""
    rng = np.random.default_rng(0)
    ngals = 1000

    # Simulation catalogue.
    boxsize = 100.0
    pos = rng.uniform(0.0, boxsize, size=(ngals, 3))
    vel = rng.normal(0.0, 200.0, size=(ngals, 3))
    mstar = 10 ** rng.uniform(-0.5, 1.5, size=ngals)   # 1e10 Msun/h
    LimPar = mstar

    sim_type = CenSat_Simulations(
        pos, vel, mstar,
        snapnum=99, LimPar=LimPar, numbins=100, boxsize=boxsize,
        MaxScale=1.5, nRvirOrMpc="nRvir",
        MaxVel=2.0, nVvirORkm="nVvir",
        LimPar_min=10 ** (-0.5) * 0.673,
        filename="censat_simulation_example",
    )
    print("  simulation : %d centrals, %d satellites"
          % (np.sum(sim_type == 0), np.sum(sim_type == 1)))

    # Observational catalogue.
    ra = rng.uniform(0.0, 60.0, size=ngals)
    dec = rng.uniform(-1.0, 1.0, size=ngals)
    vel_los = rng.normal(0.0, 200.0, size=ngals)
    redshift = rng.uniform(0.01, 0.04, size=ngals)
    mstar_log = rng.uniform(9.5, 12.0, size=ngals)     # log10(Msun)
    LimPar_obs = mstar_log

    obs_type = CenSat_Observations(
        ra, dec, vel_los, redshift, mstar_log,
        LimPar=LimPar_obs, numbins=25,
        MaxScale=1.5, nRvirOrMpc="nRvir",
        MaxVel=2.0, nVvirORkm="nVvir",
        LimPar_min=9.5,
        filename="censat_observations_radec_example",
    )
    print("  observations: %d centrals, %d satellites"
          % (np.sum(obs_type == 0), np.sum(obs_type == 1)))


if __name__ == "__main__":
    print("Running CenSat synthetic demo ...")
    _synthetic_demo()
    print("Done. Outputs written to censat_simulation_example.hdf5 and "
          "censat_observations_radec_example.hdf5.")
    print("To use your own data, edit the placeholder arrays in the "
          "example_* functions above.")