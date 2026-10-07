"""
CenSat: classify galaxies as centrals or satellites using projected separation and line-of-sight velocity.

Written by M. Reza Ayromlou (ayromlou@gmail.com).
If you use the code, please cite Ayromlou et al. 2023a: https://ui.adsabs.harvard.edu/abs/2023MNRAS.519.1913A/abstract
"""

import numpy as np
import h5py
import time
from astropy.cosmology import Planck15

# Physical constants.
_G = 43.00708008          # Mpc km^2 s^-2 / (10^10 Msun)
_H0 = 0.673               # Dimensionless h used for observational mass/distance units
_RHO_CRIT = 27.7536627     # z=0 critical density [1e10 Msun h^2 / Mpc^3]

# Stellar-to-halo mass relation (Millennium fit).
# Original calibration:  log10(M200 [Msun/h]) = c * log10(Mstar [Msun/h]) + d
# with break at Mstar = 10^10.5 Msun/h.
# Transformed to 1e10 Msun/h input/output:
#   c_eff = c,  d_eff = 10*c + d - 10,  break_eff = 10^(10.5 - 10) = 10^0.5
_C_EFF_HI = 1.6514906803470302
_D_EFF_HI = -5.158306273312964 + 10.0 * 1.6514906803470302 - 10.0
_C_EFF_LO = 0.7981732835292858
_D_EFF_LO = 3.703213645427758  + 10.0 * 0.7981732835292858 - 10.0
_MSTAR_BREAK_E10 = 10.0 ** 0.5


def _stellar_to_halo_mock(mstar, mstar_units='Msun_per_h_e10', h=_H0):
    """
    Convert stellar mass to mock M200, R200, V200.

    Parameters
    ----------
    mstar : ndarray
        Stellar masses.
    mstar_units : str
        'Msun_per_h_e10'  — mstar in units of 1e10 Msun/h (simulations).
        'Msun_log10'      — mstar in log10(Msun) (observations).
    h : float
        Dimensionless Hubble parameter h (used for 'Msun_log10' conversion).

    Returns
    -------
    M200 : ndarray  [1e10 Msun/h]
    R200 : ndarray  [proper Mpc/h; z=0 halo-radius calibration]
    V200 : ndarray  [km/s]
    """
    if mstar_units == 'Msun_log10':
        # Mstar in log10(Msun).  Convert to log10(1e10 Msun/h).
        # log10(Mstar [1e10 Msun/h]) = Mstar_log10_Msun + log10(h) - 10
        mstar_log = mstar + np.log10(h) - 10.0
    elif mstar_units == 'Msun_per_h_e10':
        mstar_log = np.log10(np.maximum(mstar, 1e-30))
    else:
        raise ValueError("mstar_units must be 'Msun_per_h_e10' or 'Msun_log10'")

    M200_log = np.where(
        mstar_log >= np.log10(_MSTAR_BREAK_E10),
        _C_EFF_HI * mstar_log + _D_EFF_HI,
        _C_EFF_LO * mstar_log + _D_EFF_LO,
    )
    M200 = 10.0 ** M200_log
    R200 = (3.0 * M200 / (4.0 * np.pi * 200.0 * _RHO_CRIT)) ** (1.0 / 3.0)
    V200 = np.sqrt(_G * M200 / np.maximum(R200, 1e-6))
    return M200, R200, V200


def _grid_cell_width(numbins, boxsize):
    """Validate a grid and return its cell width in coordinate units."""
    if not isinstance(numbins, (int, np.integer)) or numbins <= 0:
        raise ValueError("numbins must be a positive integer")
    if not np.isfinite(boxsize) or boxsize <= 0:
        raise ValueError("boxsize must be finite and positive")
    return boxsize / numbins


def CenSat_Simulations(galdata1_pos,galdata1_vel,galdata1_Mstar,snapnum,LimPar,numbins,boxsize,MaxScale=1.5,nRvirOrMpc='nRvir',MaxVel=2.0,nVvirORkm='nVvir',LimPar_min=-np.inf,filename='cen_sat_simulations'):
    """
    Classify a low-redshift simulation catalogue in projected redshift space.

    galdata1_pos : (N, 3) positions in Mpc/h within the periodic box.
    galdata1_vel : (N, 3) peculiar velocities in km/s.
    galdata1_Mstar : (N,) stellar masses in 1e10 Msun/h.
    LimPar : (N,) selection variable; only LimPar > LimPar_min is classified.
    boxsize : periodic box length in Mpc/h; numbins is the grid resolution.
    snapnum : retained for compatibility; it does not change the cosmology.

    The x-y separation and z velocity are used. The velocity cut includes
    100 * delta_z [Mpc/h] km/s of Hubble flow, appropriate at z approximately 0.
    Halo properties are derived from stellar mass. With 'nRvir'/'nVvir',
    MaxScale/MaxVel multiply R200/V200; with 'Mpc'/'km', they are fixed cuts
    in Mpc/h and km/s. The search radius must not exceed half the box length.

    Return int labels (0=central, 1=satellite) in selected input-row order.
    Write the same labels to <filename>.hdf5 under /Galaxy/type.
    """

    Mvir, Rvir, Vvir = _stellar_to_halo_mock(galdata1_Mstar, mstar_units='Msun_per_h_e10')

    h = _grid_cell_width(numbins, boxsize)

    realgal_idx = np.where(LimPar>LimPar_min)[0]

    ngal = np.size(realgal_idx)
    print('ngal = %d' %ngal)
    gals_type = np.ones(ngal,dtype=np.int64)*(-1) # -1: unclassified; 0: central; 1: satellite.

    if(nRvirOrMpc=='nRvir'):
        Rmax = Rvir*MaxScale
        Rmax1 = Rmax[realgal_idx]
    elif(nRvirOrMpc=='Mpc'):
        Rmax1 = np.ones(ngal)*MaxScale
    else:
        raise ValueError('nRvirOrMpc should be either "nRvir" or "Mpc"')

    if np.any(~np.isfinite(Rmax1)) or np.any(Rmax1 < 0) or np.any(Rmax1 > boxsize / 2):
        raise ValueError("Simulation search radii must be finite and between zero and boxsize/2")

    if(nVvirORkm=='nVvir'):
        Velmax = Vvir*MaxVel
        Velmax1 = Velmax[realgal_idx]
    elif(nVvirORkm=='km'):
        Velmax1 = np.ones(ngal)*MaxVel
    else:
        raise ValueError("nVvirORkm must be 'nVvir' or 'km' for simulations.")


    Pos1 = galdata1_pos[realgal_idx][:,0:2]
    Pos1_zaxis = galdata1_pos[realgal_idx][:,2]

    Vel1 = galdata1_vel[realgal_idx][:,2]
    Mvir1 = Mvir[realgal_idx]

    nbins2d = numbins**2


    Mvir_sorted_idx = np.argsort(Mvir1)[::-1]

    Pos_sorted = Pos1[Mvir_sorted_idx]
    Pos_sorted_zaxis = Pos1_zaxis[Mvir_sorted_idx]
    pos_bins_sorted = bin_the_pos(Pos_sorted,numbins,boxsize,2)
    Vel_sorted = Vel1[Mvir_sorted_idx]
    Rmax_sorted = Rmax1[Mvir_sorted_idx]
    Velmax_sorted = Velmax1[Mvir_sorted_idx]

    print('start binning the galaxies')

    galbin_idx = [[] for i in range(0,nbins2d)]

    for igal in range(0,ngal):
        galbin_idx[np.int64(pos_bins_sorted[igal])]+=[igal]

    t = time.time()

    gal_type_minus1_idx = np.where(gals_type==-1)
    gal_type_minus1_idx_size = np.size(gal_type_minus1_idx)

    while (gal_type_minus1_idx_size>0):
        gal_type_minus1_idx = np.where(gals_type==-1)
        gal_type_minus1_idx_size = np.size(gal_type_minus1_idx)
        if gal_type_minus1_idx_size == 0:
            break

        this_type0_idx = gal_type_minus1_idx[0][0]
        if this_type0_idx % 10000 == 0:
            print('this_type0_idx = %d' %this_type0_idx)
        this_Rmax = Rmax_sorted[this_type0_idx]
        this_Velmax = Velmax_sorted[this_type0_idx]
        gals_type[this_type0_idx] = 0

        jgalpos = Pos_sorted[this_type0_idx,:]
        jgalpos1_zaxis = Pos_sorted_zaxis[this_type0_idx]
        jgalvel = Vel_sorted[this_type0_idx]
        jpos_bin = np.minimum(np.floor(jgalpos / h), numbins - 1)
        if (np.floor((jgalpos[0]-this_Rmax)/h)==jpos_bin[0] and np.floor((jgalpos[0]+this_Rmax)/h)==jpos_bin[0] and np.floor((jgalpos[1]-this_Rmax)/h)==jpos_bin[1] and np.floor((jgalpos[1]+this_Rmax)/h)==jpos_bin[1]):
            my_bins_num = 0
        else:
            my_bins_num = np.int64(np.ceil(this_Rmax*numbins/boxsize))

        for k1 in range(-my_bins_num,my_bins_num+1):
            for k2 in range(-my_bins_num,my_bins_num+1):

                jgalpos1 = np.zeros(2)
                # Periodic boundary condition
                x_bin = np.int64(jpos_bin[0]+k1)
                y_bin = np.int64(jpos_bin[1]+k2)
                if x_bin<0:
                    x_bin+=numbins
                    jgalpos1[0] = jgalpos[0] + boxsize
                elif x_bin>numbins-1:
                    x_bin-=numbins
                    jgalpos1[0] = jgalpos[0] - boxsize
                else:
                    jgalpos1[0] = jgalpos[0]

                if y_bin<0:
                    y_bin+=numbins
                    jgalpos1[1] = jgalpos[1] + boxsize
                elif y_bin>numbins-1:
                    y_bin-=numbins
                    jgalpos1[1] = jgalpos[1] - boxsize
                else:
                    jgalpos1[1] = jgalpos[1]

                jbin = x_bin + numbins*y_bin

                dis_vec = Pos_sorted[galbin_idx[jbin],:] - jgalpos1
                dis = calculate_magnitude(dis_vec,2)


                dis_vec_zaxis = Pos_sorted_zaxis[galbin_idx[jbin]] - jgalpos1_zaxis
                dummy = np.where(dis_vec_zaxis>boxsize/2)
                dis_vec_zaxis[dummy] = dis_vec_zaxis[dummy] - boxsize
                dummy = np.where(dis_vec_zaxis<-boxsize/2)
                dis_vec_zaxis[dummy] = dis_vec_zaxis[dummy] + boxsize

                velrel = Vel_sorted[galbin_idx[jbin]] - jgalvel
                vLOS_projected_vec = velrel + dis_vec_zaxis*100 # when dis is in [Mpc/h] and vel is in [km/s] and z~0
                vLOS_projected = np.abs(vLOS_projected_vec)
                candidate_idx = np.asarray(galbin_idx[jbin], dtype=np.intp)
                satellite_mask = ((dis < this_Rmax) & (vLOS_projected < this_Velmax)
                                  & (gals_type[candidate_idx] == -1))
                gals_type[candidate_idx[satellite_mask]] = 1

    t2 = time.time() - t
    print('CenSat: done in %.2f s' % t2)

    gals_type_save = np.zeros(ngal,dtype=np.int64)
    gals_type_save[Mvir_sorted_idx] = gals_type
    with h5py.File('%s.hdf5' % filename, 'w') as f:
        f['/Galaxy/type'] = gals_type_save

    return(gals_type_save)


def CenSat_Observations(galdata1_ra,galdata1_dec,galdata1_vel,galdata1_redshift,galdata1_Mstar,LimPar,numbins,MaxScale=1.5,nRvirOrMpc='nRvir',MaxVel=2.0,nVvirORkm='nVvir',LimPar_min=-np.inf,filename='cen_sat_observations',ra_dec_units='degree'):
    """
    Classify a low-redshift observational catalogue from sky coordinates.

    All input arrays have shape (N,). RA/Dec are degrees by default, or radians when ra_dec_units='radian'. galdata1_vel is observed LOS velocity in km/s (approximately c*z at low redshift), not an independent peculiar velocity.
    Stellar mass is log10(Mstar/Msun). Select rows with LimPar > LimPar_min.

    Sky directions are embedded in a non-periodic dummy Cartesian grid.
    Planck15 angular-diameter distances are converted from Mpc to proper Mpc/h using _H0, the same h used for the stellar-mass conversion. Thus projected distances, R200, and fixed 'Mpc' cuts use proper Mpc/h.
    The stellar-to-halo relation and critical density are calibrated at z=0; this function does not evolve that halo calibration with redshift.

    Return int labels (0=central, 1=satellite) in selected input-row order.
    Write the same labels to <filename>.hdf5 under /Galaxy/type.
    """

    boxsize = 500.0  # Arbitrary extent of the dummy coordinate grid.

    galdata1_pos = convert_ra_dec_to_xyz(galdata1_ra, galdata1_dec, np.ones(np.size(galdata1_ra)), ra_dec_units)

    extent = np.max(galdata1_pos, initial=0.0)
    r_dummy_scale = boxsize / extent if extent > 0 else 1.0
    galdata1_pos = np.minimum(galdata1_pos * r_dummy_scale, boxsize)

    # Astropy returns Mpc. Multiplication by the mass-conversion h gives Mpc/h.
    galdata1_dis = (Planck15.angular_diameter_distance(galdata1_redshift).to_value('Mpc') * _H0
                   if np.size(galdata1_redshift) else np.empty(0))

    Mvir, Rvir, Vvir = _stellar_to_halo_mock(galdata1_Mstar, mstar_units='Msun_log10')

    return CenSat_Observations_xyz(galdata1_pos, galdata1_vel, galdata1_dis,
                                    r_dummy_scale, Rvir, Vvir, Mvir,
                                    LimPar, numbins, boxsize,
                                    MaxScale, nRvirOrMpc, MaxVel, nVvirORkm,
                                    LimPar_min, filename)


def CenSat_Observations_xyz(galdata1_pos,galdata1_vel,galdata1_dis,r_dummy_scale,galdata1_Rvir,galdata1_Vvir,galdata1_Mvir,LimPar,numbins,boxsize,MaxScale=1.5,nRvirOrMpc='nRvir',MaxVel=2.0,nVvirORkm='nVvir',LimPar_min=-np.inf,filename='cen_sat_observations_xyz'):
    """
    * Backend for the projected observational classifier; accepts scaled sky vectors and pre-computed halo properties instead of RA/Dec + Mstar.
    * galdata1_pos    : (N,3) sky-direction vectors shifted and scaled to [0, boxsize].
    * galdata1_dis    : (N,) angular-diameter distances in proper Mpc/h, positive
                        for selected rows. These must use the same h as R200.
    * r_dummy_scale   : positive scalar that multiplies the shifted unit-sphere
                        positions. The projected chord is dis_dummy * D_A / r_dummy_scale.
    * galdata1_Rvir   : (N,) proper R200 in Mpc/h, as returned by _stellar_to_halo_mock.
    * galdata1_Vvir   : (N,) V200 in km/s.
    * galdata1_Mvir   : (N,) M200 in 1e10 Msun/h (used only for mass-rank sorting).
    * boxsize         : extent of the dummy coordinate grid, typically 500.
                        It is not a physical survey size.
    * LimPar          : (N,) selection variable; select LimPar > LimPar_min.
    * Output: type = 0 for central, type = 1 for satellite, in selected input order.
    * The output file is written to "<filename>.hdf5" under "/Galaxy/type".
    """

    cell_width = _grid_cell_width(numbins, boxsize)
    if not np.isfinite(r_dummy_scale) or r_dummy_scale <= 0:
        raise ValueError("r_dummy_scale must be finite and positive")

    realgal_idx = np.where(LimPar>LimPar_min)[0]

    ngal = np.size(realgal_idx)
    print('ngal = %d' %ngal)
    gals_type = np.ones(ngal,dtype=np.int64)*(-1)

    if(nRvirOrMpc=='nRvir'):
        Rmax = galdata1_Rvir*MaxScale
        Rmax1 = Rmax[realgal_idx]
    elif(nRvirOrMpc=='Mpc'):
        Rmax1 = np.ones(ngal)*MaxScale
    else:
        raise ValueError('nRvirOrMpc should be either "nRvir" or "Mpc"')

    if(nVvirORkm=='nVvir'):
        Velmax = galdata1_Vvir*MaxVel
        Velmax1 = Velmax[realgal_idx]
    elif(nVvirORkm=='km'):
        Velmax1 = np.ones(ngal)*MaxVel
    else:
        raise ValueError("nVvirORkm must be 'nVvir' or 'km' for observations.")

    Pos1 = galdata1_pos[realgal_idx]
    Vel1 = galdata1_vel[realgal_idx]
    Mvir1 = galdata1_Mvir[realgal_idx]
    galdis1 = galdata1_dis[realgal_idx]
    if np.any(~np.isfinite(galdis1)) or np.any(galdis1 <= 0):
        raise ValueError("Selected observational distances must be finite and positive")
    if np.any(~np.isfinite(Rmax1)) or np.any(Rmax1 < 0):
        raise ValueError("Observational search radii must be finite and non-negative")

    nbins3d = numbins**3

    Mvir_sorted_idx = np.argsort(Mvir1)[::-1]
    galdis_sorted = galdis1[Mvir_sorted_idx]

    Pos_sorted = Pos1[Mvir_sorted_idx]
    pos_bins_sorted = bin_the_pos(Pos_sorted,numbins,boxsize,3)
    Vel_sorted = Vel1[Mvir_sorted_idx]
    Rmax_sorted = Rmax1[Mvir_sorted_idx]
    Velmax_sorted = Velmax1[Mvir_sorted_idx]

    print('start binning the galaxies')

    galbin_idx = [[] for i in range(0,nbins3d)]

    for igal in range(0,ngal):
        galbin_idx[np.int64(pos_bins_sorted[igal])]+=[igal]

    t = time.time()

    gal_type_minus1_idx = np.where(gals_type==-1)
    gal_type_minus1_idx_size = np.size(gal_type_minus1_idx)

    j = 0
    while (gal_type_minus1_idx_size>0):
        gal_type_minus1_idx = np.where(gals_type==-1)
        gal_type_minus1_idx_size = np.size(gal_type_minus1_idx)
        if gal_type_minus1_idx_size == 0:
            break

        this_type0_idx = gal_type_minus1_idx[0][0]
        if j % 10000 == 0:
            print('this_type0_idx = %d' %this_type0_idx)
        j = j+1
        this_Rmax = Rmax_sorted[this_type0_idx]
        this_Velmax = Velmax_sorted[this_type0_idx]
        this_gal_dis = galdis_sorted[this_type0_idx]
        gals_type[this_type0_idx] = 0

        jgalpos = Pos_sorted[this_type0_idx,:]
        jgalvel = Vel_sorted[this_type0_idx]
        # dis_projected = dis_dummy * D_A / scale, so invert that conversion
        # before finding every grid cell intersecting the candidate search sphere.
        radius_dummy = this_Rmax * r_dummy_scale / this_gal_dis
        bin_min = np.maximum(np.floor((jgalpos - radius_dummy) / cell_width), 0).astype(np.int64)
        bin_max = np.minimum(np.floor((jgalpos + radius_dummy) / cell_width), numbins - 1).astype(np.int64)

        for x_bin in range(int(bin_min[0]), int(bin_max[0]) + 1):
            for y_bin in range(int(bin_min[1]), int(bin_max[1]) + 1):
                for z_bin in range(int(bin_min[2]), int(bin_max[2]) + 1):
                    jbin = x_bin + numbins*(y_bin+numbins*z_bin)
                    candidate_idx = np.asarray(galbin_idx[jbin], dtype=np.intp)
                    if candidate_idx.size == 0:
                        continue

                    dis_vec = Pos_sorted[candidate_idx,:] - jgalpos
                    dis = calculate_magnitude(dis_vec,3)

                    # Both this projected chord and Rmax are in proper Mpc/h.
                    dis = dis * this_gal_dis / r_dummy_scale

                    velrel = Vel_sorted[candidate_idx] - jgalvel
                    vLOS_projected = np.abs(velrel)
                    satellite_mask = ((dis < this_Rmax) & (vLOS_projected < this_Velmax)
                                      & (gals_type[candidate_idx] == -1))
                    gals_type[candidate_idx[satellite_mask]] = 1

    t2 = time.time() - t
    print('CenSat: done in %.2f s' % t2)

    gals_type_save = np.zeros(ngal,dtype=np.int64)
    gals_type_save[Mvir_sorted_idx] = gals_type
    with h5py.File('%s.hdf5' % filename, 'w') as f:
        f['/Galaxy/type'] = gals_type_save

    return(gals_type_save)


def calculate_magnitude(data,dim=2):
    if dim==2:
        mag = np.sqrt(data[:,0]**2 + data[:,1]**2)
    elif dim==3:
        mag = np.sqrt(data[:,0]**2 + data[:,1]**2 + data[:,2]**2)
    return(mag)


def bin_the_pos(Pos,numbins,boxsize,dim=2):
    """
    * This function bins the positions of galaxies into a 1D array of bin numbers.
    * Input: the position of galaxies, the number of bins, the box size, and the dimension (2 or 3).
    * Output: 1D array of bin numbers for each galaxy.
    """
    h = _grid_cell_width(numbins, boxsize)
    if dim not in (2, 3):
        raise ValueError("dim must be 2 or 3")
    Pos = np.asarray(Pos)
    if Pos.ndim != 2 or Pos.shape[1] < dim:
        raise ValueError("Positions must be an (N, dim) array")
    if np.any(~np.isfinite(Pos[:, :dim])) or np.any(Pos[:, :dim] < 0) or np.any(Pos[:, :dim] > boxsize):
        raise ValueError("Positions must be finite and within [0, boxsize]")
    if dim==3:
        # A point exactly on the upper boundary belongs to the final cell.
        pos_bins_3d = np.minimum(np.floor(Pos[:, :3]/h), numbins - 1)
        pos_bins_3d = np.array(pos_bins_3d,dtype=np.int64)
        pos_bins_1d = pos_bins_3d[:,0] + numbins*(pos_bins_3d[:,1]+numbins*pos_bins_3d[:,2])
    elif dim==2:
        pos_bins_2d = np.minimum(np.floor(Pos[:, :2]/h), numbins - 1)
        pos_bins_2d = np.array(pos_bins_2d,dtype=np.int64)
        pos_bins_1d = pos_bins_2d[:,0] + numbins*pos_bins_2d[:,1]

    return(pos_bins_1d)


def convert_ra_dec_to_xyz(ra,dec,dis,units_input='degree'):
    """Embed sky directions at distance dis, shifting each axis to start at zero."""
    if np.size(ra) == 0:
        return np.empty((0, 3), dtype=np.float64)
    if units_input == 'degree':
        ra = ra*np.pi/180
        dec = dec*np.pi/180

    x = dis*np.cos(dec)*np.cos(ra)
    x = x - np.min(x)
    y = dis*np.cos(dec)*np.sin(ra)
    y = y - np.min(y)
    z = dis*np.sin(dec)
    z = z - np.min(z)

    d = np.transpose(np.array([x,y,z],dtype=np.float64))

    return(d)
