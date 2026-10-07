"""
Run CenSat on TNG100 and SDSS catalogues and plot central/satellite fractions.

Provide input paths with --tng100 and --sdss, or set CENSAT_TNG100_FILE and
CENSAT_SDSS_FILE. Input catalogues are not included in the repository.

TNG input can be a prepared HDF5 file, a native output directory, or groups_099.
Native catalogues use header units and exclude SubhaloFlag=False by default.
Missing input paths are skipped with a message.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from CenSat import CenSat_Simulations, CenSat_Observations

# Defaults for prepared files without native header metadata.
DEFAULT_HUBBLE = 0.673
DEFAULT_BOXSIZE = 75.0  # Mpc/h


# TNG100 catalogues.

def _native_tng100_catalogue(tng_path, boxsize=None, hubble=None,
                             include_subhalo_clumps=False):
    """Read z=0 native chunks, preserving original global Subhalo index order."""
    root = Path(tng_path)
    if (root / 'groups_099').is_dir():
        root = root / 'groups_099'
    chunks = sorted(root.glob('fof_subhalo_tab_099.*.hdf5'),
                    key=lambda path: int(path.name.split('.')[-2]))
    if not chunks:
        raise ValueError('No snapshot 099 group catalogue chunks found in %s' % root)

    with h5py.File(chunks[0], 'r') as f:
        header = f['Header'].attrs
        boxsize = float(header['BoxSize']) / 1000.0 if boxsize is None else boxsize
        hubble = float(header['HubbleParam']) if hubble is None else hubble
        expected_subhalos = int(header['Nsubgroups_Total'])
        expected_files = int(header['NumFiles'])
        redshift = float(header['Redshift'])
    if len(chunks) != expected_files:
        raise ValueError('Expected %d native catalogue chunks, found %d' % (expected_files, len(chunks)))
    if not np.isclose(redshift, 0.0, atol=1e-5):
        raise ValueError('This example reader requires the z=0 snapshot 099 catalogue')
    if not np.isfinite(hubble) or hubble <= 0 or not np.isfinite(boxsize) or boxsize <= 0:
        raise ValueError('Catalogue h and box size must be finite and positive')

    mass_limit = 10 ** (-0.5) * hubble  # log10(Mstar/Msun) > 9.5
    masses, positions, velocities, indices, first_subhalos = [], [], [], [], []
    offset = 0
    excluded_clumps = 0
    for chunk_index, path in enumerate(chunks):
        with h5py.File(path, 'r') as f:
            if 'Group/GroupFirstSub' in f:
                first = f['Group/GroupFirstSub'][:]
                first_subhalos.append(first[first >= 0])
            if 'Subhalo/SubhaloMassType' in f:
                mstar = f['Subhalo/SubhaloMassType'][:, 4]
                selected = mstar > mass_limit
                if 'Subhalo/SubhaloFlag' in f and not include_subhalo_clumps:
                    flag = f['Subhalo/SubhaloFlag'][:].astype(bool)
                    excluded_clumps += int(np.sum(selected & ~flag))
                    selected &= flag
                masses.append(mstar[selected])
                positions.append(f['Subhalo/SubhaloPos'][:][selected] / 1000.0)
                velocities.append(f['Subhalo/SubhaloVel'][:][selected])
                indices.append(np.flatnonzero(selected) + offset)
                offset += len(mstar)
        if (chunk_index + 1) % 112 == 0:
            print('  Read %d / %d native catalogue chunks' % (chunk_index + 1, len(chunks)))

    if offset != expected_subhalos:
        raise ValueError('Expected %d subhalos, read %d' % (expected_subhalos, offset))
    if not masses or not first_subhalos:
        raise ValueError('Native catalogue must contain Subhalo and GroupFirstSub datasets')
    subhalo_id = np.concatenate(indices)
    central_id = np.concatenate(first_subhalos)
    true_type = np.where(np.isin(subhalo_id, central_id), 0, 1)
    print('  Excluded %d above-limit non-cosmological subhalos (SubhaloFlag=False)' % excluded_clumps)
    return dict(pos=np.concatenate(positions), vel=np.concatenate(velocities),
                mstar=np.concatenate(masses), true_type=true_type,
                subhalo_id=subhalo_id, boxsize=boxsize, h=hubble)


def _prepared_tng100_catalogue(tng_path, boxsize=None, hubble=None,
                               include_subhalo_clumps=False):
    """Read the documented prepared schema; native-style Header units are optional."""
    with h5py.File(tng_path, 'r') as f:
        header = f['Header'].attrs if 'Header' in f else {}
        if boxsize is None:
            boxsize = float(header['BoxSize']) / 1000.0 if 'BoxSize' in header else DEFAULT_BOXSIZE
        if hubble is None:
            hubble = float(header.get('HubbleParam', DEFAULT_HUBBLE))
        pos = f['SubhaloPos'][:] / 1000.0
        vel = f['SubhaloVel'][:]
        mstar = f['SubhaloMassType'][:, 4]
        true_type = f['SubhaloType'][:]
        subhalo_id = np.arange(len(mstar))
        if 'SubhaloFlag' in f and not include_subhalo_clumps:
            flag = f['SubhaloFlag'][:].astype(bool)
            pos, vel, mstar, true_type, subhalo_id = (
                array[flag] for array in (pos, vel, mstar, true_type, subhalo_id))
    if not np.isfinite(hubble) or hubble <= 0 or not np.isfinite(boxsize) or boxsize <= 0:
        raise ValueError('Catalogue h and box size must be finite and positive')
    return dict(pos=pos, vel=vel, mstar=mstar, true_type=true_type,
                subhalo_id=subhalo_id, boxsize=boxsize, h=hubble)


def run_tng100(tng_path, boxsize=None, hubble=None, include_subhalo_clumps=False):
    """Run CenSat on native TNG100 chunks or a prepared catalogue."""
    if not tng_path:
        print("TNG100: no input path given (set CENSAT_TNG100_FILE); skipping.")
        return None
    if not os.path.exists(tng_path):
        print(f"TNG100: path not found: {tng_path}; skipping.")
        return None

    print(f"Loading TNG100: {tng_path}")
    loader = _native_tng100_catalogue if os.path.isdir(tng_path) else _prepared_tng100_catalogue
    catalogue = loader(tng_path, boxsize, hubble, include_subhalo_clumps)
    pos, vel, mstar, true_type = (catalogue[key] for key in ('pos', 'vel', 'mstar', 'true_type'))
    print('  Box = %.3f Mpc/h, h = %.4f' % (catalogue['boxsize'], catalogue['h']))

    LimPar_min = 10 ** (-0.5) * catalogue['h']
    LimPar = mstar

    print(f"  N galaxies = {len(mstar)},  N > mass limit = {np.sum(mstar > LimPar_min)}")
    if not np.any(mstar > LimPar_min):
        print('TNG100: no galaxies pass the mass limit; skipping.')
        return None

    gtype = CenSat_Simulations(
        pos, vel, mstar,
        snapnum=99, LimPar=LimPar, numbins=70, boxsize=catalogue['boxsize'],
        MaxScale=1.5, nRvirOrMpc='nRvir',
        MaxVel=2.0, nVvirORkm='nVvir',
        LimPar_min=LimPar_min,
        filename='test_censat_tng100'
    )
    print(f"  Centrals: {np.sum(gtype == 0)},  Satellites: {np.sum(gtype == 1)}")

    # Compare classifications with the catalogue labels.
    mask = mstar > LimPar_min
    true_masked = true_type[mask]
    cen_cen = np.sum((gtype == 0) & (true_masked == 0))
    cen_sat = np.sum((gtype == 0) & (true_masked == 1))
    sat_cen = np.sum((gtype == 1) & (true_masked == 0))
    sat_sat = np.sum((gtype == 1) & (true_masked == 1))

    print("\nConfusion Matrix (CenSat rows, TNG cols):")
    print("                 TNG central  TNG satellite")
    print(f"  CenSat central    {cen_cen:>6d}       {cen_sat:>6d}")
    print(f"  CenSat satellite  {sat_cen:>6d}       {sat_sat:>6d}")

    purity_cen = cen_cen / (cen_cen + cen_sat) * 100 if (cen_cen + cen_sat) > 0 else 0
    complete_cen = cen_cen / (cen_cen + sat_cen) * 100 if (cen_cen + sat_cen) > 0 else 0
    purity_sat = sat_sat / (sat_sat + sat_cen) * 100 if (sat_sat + sat_cen) > 0 else 0
    complete_sat = sat_sat / (sat_sat + cen_sat) * 100 if (sat_sat + cen_sat) > 0 else 0

    print(f"\n  Central  purity:     {purity_cen:.1f}%  (CenSat centrals that are true centrals)")
    print(f"  Central  completeness: {complete_cen:.1f}%  (true centrals that CenSat finds)")
    print(f"  Satellite purity:     {purity_sat:.1f}%  (CenSat satellites that are true satellites)")
    print(f"  Satellite completeness: {complete_sat:.1f}%  (true satellites that CenSat finds)")
    print(f"  Overall matched:     {np.sum(gtype == true_masked)} / {len(gtype)}  ({np.sum(gtype == true_masked)/len(gtype)*100:.1f}%)")

    return dict(mstar=mstar, gtype=gtype, true_type=true_type,
                mask=mask, true_masked=true_masked, h=catalogue['h'],
                subhalo_id=catalogue['subhalo_id'][mask])


# SDSS DR7 observations (z <= 0.04).

def run_sdss(sdss_path):
    """Run CenSat on an SDSS DR7-like observational catalogue; return results."""
    if not sdss_path:
        print("\nSDSS: no input path given (set CENSAT_SDSS_FILE); skipping.")
        return None
    if not os.path.isfile(sdss_path):
        print(f"\nSDSS: file not found: {sdss_path}; skipping.")
        return None

    print(f"\nLoading SDSS: {sdss_path}")
    with h5py.File(sdss_path, "r") as f:
        ra_sdss       = f["Galaxy/ra"][:]
        dec_sdss      = f["Galaxy/dec"][:]
        redshift_sdss = f["Galaxy/redshift"][:]
        mstar_sdss    = f["Galaxy/StellarMass_median_log"][:]   # log10(Msun)

    # Redshift cut
    zcut = (redshift_sdss > 0) & (redshift_sdss <= 0.04)
    ra_sdss       = ra_sdss[zcut]
    dec_sdss      = dec_sdss[zcut]
    redshift_sdss = redshift_sdss[zcut]
    mstar_sdss    = mstar_sdss[zcut]

    print(f"  N after 0<z<=0.04: {len(ra_sdss)}")

    # Line-of-sight velocity ≈ cz
    c_light = 3e5
    vel_los = c_light * redshift_sdss

    LimPar_min_obs = 9.5       # log10(Msun)
    LimPar_obs = mstar_sdss

    print(f"  N > mass limit ({LimPar_min_obs}): {np.sum(mstar_sdss > LimPar_min_obs)}")
    if not np.any(mstar_sdss > LimPar_min_obs):
        print('SDSS: no galaxies pass the redshift and mass limits; skipping.')
        return None

    gtype_sdss = CenSat_Observations(
        ra_sdss, dec_sdss, vel_los, redshift_sdss, mstar_sdss,
        LimPar=LimPar_obs, numbins=25,
        MaxScale=1.5, nRvirOrMpc='nRvir',
        MaxVel=2.0, nVvirORkm='nVvir',
        LimPar_min=LimPar_min_obs,
        filename='test_censat_sdss'
    )
    print(f"  Centrals: {np.sum(gtype_sdss == 0)},  Satellites: {np.sum(gtype_sdss == 1)}")

    return dict(mstar=mstar_sdss, gtype=gtype_sdss, LimPar_min=LimPar_min_obs)


# Fraction plots.

def make_plot(tng, sdss, outfile='test_censat_fractions.png'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # TNG100.
    ax = axes[0]
    if tng is not None:
        mstar_masked = tng['mstar'][tng['mask']]
        logM = np.log10(mstar_masked * 1e10 / tng['h'])
        bins_tng = np.linspace(9.0, 12.5, 20)

        n_total, _ = np.histogram(logM, bins=bins_tng)
        n_cen, _   = np.histogram(logM[tng['gtype'] == 0], bins=bins_tng)
        n_sat, _   = np.histogram(logM[tng['gtype'] == 1], bins=bins_tng)
        n_cen_t, _ = np.histogram(logM[tng['true_masked'] == 0], bins=bins_tng)
        n_sat_t, _ = np.histogram(logM[tng['true_masked'] == 1], bins=bins_tng)
        n_match, _ = np.histogram(logM[tng['gtype'] == tng['true_masked']], bins=bins_tng)

        with np.errstate(divide='ignore', invalid='ignore'):
            f_cen, f_sat = n_cen / n_total, n_sat / n_total
            f_cen_t, f_sat_t = n_cen_t / n_total, n_sat_t / n_total
            f_match = n_match / n_total
        bx_tng = 0.5 * (bins_tng[:-1] + bins_tng[1:])

        ax.plot(bx_tng, f_cen,   'o-', color='blue',  label='CenSat central')
        ax.plot(bx_tng, f_sat,   's-', color='red',   label='CenSat satellite')
        ax.plot(bx_tng, f_cen_t, '--', color='blue',  alpha=0.6, label='TNG central')
        ax.plot(bx_tng, f_sat_t, '--', color='red',   alpha=0.6, label='TNG satellite')
        ax.plot(bx_tng, f_match, 'D-', color='green', label='Matched')
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, 'TNG100 catalogue not available',
                ha='center', va='center', transform=ax.transAxes)
    ax.set_xlabel(r'$\log_{10}(M_\star \;/\; M_\odot)$')
    ax.set_ylabel('Fraction')
    ax.set_title('TNG100 (z=0)')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # SDSS.
    ax = axes[1]
    if sdss is not None:
        LimPar_min_obs = sdss['LimPar_min']
        mask_sdss = sdss['mstar'] > LimPar_min_obs
        logM_sdss = sdss['mstar'][mask_sdss]
        gtype_sdss_m = sdss['gtype']
        bins_sdss = np.linspace(9.5, 12.0, 15)

        n_tot_s, _ = np.histogram(logM_sdss, bins=bins_sdss)
        n_cen_s, _ = np.histogram(logM_sdss[gtype_sdss_m == 0], bins=bins_sdss)
        n_sat_s, _ = np.histogram(logM_sdss[gtype_sdss_m == 1], bins=bins_sdss)

        with np.errstate(divide='ignore', invalid='ignore'):
            f_cen_s = n_cen_s / n_tot_s
            f_sat_s = n_sat_s / n_tot_s
        bx_sdss = 0.5 * (bins_sdss[:-1] + bins_sdss[1:])

        ax.plot(bx_sdss, f_cen_s, 'o-', color='blue',  label='CenSat central')
        ax.plot(bx_sdss, f_sat_s, 's-', color='red',   label='CenSat satellite')
        ax.set_title(f'SDSS DR7 (z≤0.04, N={np.sum(mask_sdss)})')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'SDSS catalogue not available',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('SDSS DR7 (z≤0.04)')
    ax.set_xlabel(r'$\log_{10}(M_\star \;/\; M_\odot)$')
    ax.set_ylabel('Fraction')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {outfile}")


def main():
    parser = argparse.ArgumentParser(description="Run CenSat on example catalogues.")
    parser.add_argument("--tng100",
                        default=os.environ.get("CENSAT_TNG100_FILE", ""),
                        help="Prepared HDF5 file, native TNG100 output directory, or groups_099 directory.")
    parser.add_argument('--boxsize', type=float,
                        help='Override simulation box length in Mpc/h (otherwise native header or 75).')
    parser.add_argument('--hubble', type=float,
                        help='Override catalogue h (otherwise native header or 0.673 for prepared files).')
    parser.add_argument('--include-subhalo-clumps', action='store_true',
                        help='Include SubhaloFlag=False objects when that field is available.')
    parser.add_argument("--sdss",
                        default=os.environ.get("CENSAT_SDSS_FILE", ""),
                        help="Path to an SDSS DR7-like observational catalogue (HDF5).")
    parser.add_argument("--outfile", default="test_censat_fractions.png",
                        help="Output plot filename.")
    args = parser.parse_args()

    tng = run_tng100(args.tng100, args.boxsize, args.hubble, args.include_subhalo_clumps)
    sdss = run_sdss(args.sdss)

    if tng is None and sdss is None:
        print("\nNo input catalogues available. Provide --tng100 and/or --sdss "
              "(or the CENSAT_TNG100_FILE / CENSAT_SDSS_FILE environment variables).")
        return

    make_plot(tng, sdss, outfile=args.outfile)


if __name__ == "__main__":
    main()
