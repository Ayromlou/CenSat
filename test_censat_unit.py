"""Regression tests requiring no external catalogues or output files.

Run with: python -B -m unittest -v test_censat_unit
"""

import contextlib
import io
from pathlib import Path
import unittest
from unittest import mock

import h5py
import numpy as np
from astropy.cosmology import Planck15

import CenSat as censat
import test_censat as examples


HDF5_FILE = h5py.File


def brute_force_projected(pos, velocity, distance, scale, radius, velcut, mass):
    """Reference classification that checks every galaxy, without a grid."""
    order = np.argsort(mass)[::-1]
    labels = np.full(len(mass), -1, dtype=np.int64)
    for central in order:
        if labels[central] != -1:
            continue
        labels[central] = 0
        separation = np.linalg.norm(pos - pos[central], axis=1) * distance[central] / scale
        satellites = ((labels == -1) & (separation < radius[central])
                      & (np.abs(velocity - velocity[central]) < velcut[central]))
        labels[satellites] = 1
    return labels


def ra_offset(projected_separation, distance, dec=30.0):
    """RA offset giving the requested unit-sphere chord at fixed declination."""
    return np.rad2deg(2 * np.arcsin(projected_separation /
                                  (2 * distance * np.cos(np.deg2rad(dec)))))


def redshift_at_distance(distance):
    """Invert the low-redshift angular-diameter distance by bisection."""
    lower, upper = 1e-8, 0.1
    for _ in range(60):
        midpoint = (lower + upper) / 2
        if Planck15.angular_diameter_distance(midpoint).to_value('Mpc') * censat._H0 < distance:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2


class CenSatRegressionTests(unittest.TestCase):
    def setUp(self):
        self.outputs = {}
        # Exercise real HDF5 serialization without creating filesystem artifacts.
        self.file_patch = mock.patch.object(censat.h5py, 'File', self.memory_output)
        self.file_patch.start()
        self.addCleanup(self.file_patch.stop)
        self.stdout = contextlib.redirect_stdout(io.StringIO())
        self.stdout.__enter__()
        self.addCleanup(self.stdout.__exit__, None, None, None)

    def memory_output(self, filename, mode, *args, **kwargs):
        self.assertEqual(mode, 'w')
        buffer = io.BytesIO()
        self.outputs[str(filename)] = buffer
        return HDF5_FILE(buffer, mode, *args, **kwargs)

    def assert_saved_labels(self, filename, expected):
        with HDF5_FILE(self.outputs[filename], 'r') as output:
            dataset = output['Galaxy/type']
            self.assertEqual(dataset.dtype, np.dtype('int64'))
            np.testing.assert_array_equal(dataset[:], expected)

    def sky_labels(self, ra, dec, redshift, mass, numbins=25, **kwargs):
        ra, dec, redshift, mass = (np.asarray(values, dtype=float)
                                  for values in (ra, dec, redshift, mass))
        return censat.CenSat_Observations(
            ra, dec, 3e5 * redshift, redshift, mass,
            LimPar=mass, numbins=numbins, LimPar_min=9.5,
            filename='sky_regression', **kwargs)

    def test_stellar_mass_units_are_equivalent(self):
        physical_log_mass = np.array([9.5, 10.0, 10.5, 11.0, 12.0])
        mass_per_h_e10 = 10 ** (physical_log_mass - 10) * censat._H0
        simulation = censat._stellar_to_halo_mock(mass_per_h_e10)
        observations = censat._stellar_to_halo_mock(physical_log_mass, 'Msun_log10')
        for from_simulation, from_observations in zip(simulation, observations):
            np.testing.assert_allclose(from_simulation, from_observations, rtol=2e-14)

    def test_upper_grid_boundary_belongs_to_last_cell(self):
        pos = np.array([[0., 0., 0.], [500., 500., 500.]])
        for dimension in (2, 3):
            with self.subTest(dimension=dimension):
                np.testing.assert_array_equal(
                    censat.bin_the_pos(pos, 25, 500., dimension),
                    [0, 25 ** dimension - 1])

    def test_sky_upper_boundary_is_classified(self):
        labels = self.sky_labels([0., 0.], [0., 45.], [.02, .02], [10., 9.8])
        np.testing.assert_array_equal(labels, [0, 0])
        self.assert_saved_labels('sky_regression.hdf5', labels)

    def test_empty_and_single_object_catalogues(self):
        for size in (0, 1):
            with self.subTest(size=size):
                labels = self.sky_labels(np.full(size, 60.), np.full(size, 30.),
                                         np.full(size, .02), np.full(size, 10.))
                np.testing.assert_array_equal(labels, np.zeros(size, dtype=np.int64))
                self.assert_saved_labels('sky_regression.hdf5', labels)

    def test_nonpositive_selected_distance_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'distances must be finite and positive'):
            self.sky_labels([60.], [30.], [0.], [10.])

    def test_dummy_search_radius_is_converted_before_binning(self):
        pos = np.array([[23.5, 117., 117.], [26.5, 117., 117.]])
        for numbins in (5, 10, 20, 25, 50):
            with self.subTest(numbins=numbins):
                labels = censat.CenSat_Observations_xyz(
                    pos, np.zeros(2), np.full(2, 100.), 500.,
                    np.array([.8, .6]), np.full(2, 100.), np.array([2., 1.]),
                    np.ones(2), numbins, 500., MaxScale=1., MaxVel=1.,
                    filename='dummy_radius')
                np.testing.assert_array_equal(labels, [0, 1])

    def test_grid_matches_brute_force_with_edges_and_multiple_cells(self):
        rng = np.random.default_rng(19)
        pos = rng.uniform(0., 500., (60, 3))
        pos[:8] = [[23.5, 117., 117.], [26.5, 117., 117.],
                   [0., 0., 0.], [3., 0., 0.],
                   [500., 500., 500.], [497., 500., 500.],
                   [250., 249.9, 25.01], [250., 250.1, 24.99]]
        velocity = rng.uniform(-50., 50., len(pos))
        distance = rng.uniform(90., 140., len(pos))
        radius = rng.uniform(.8, 2., len(pos))
        velcut = np.full(len(pos), 100.)
        mass = np.arange(len(pos), 0, -1, dtype=float)
        expected = brute_force_projected(pos, velocity, distance, 500., radius, velcut, mass)
        self.assertGreaterEqual(np.sum(expected == 1), 4)
        for numbins in (5, 10, 20, 25, 50):
            with self.subTest(numbins=numbins):
                labels = censat.CenSat_Observations_xyz(
                    pos, velocity, distance, 500., radius, velcut, mass,
                    np.ones(len(pos)), numbins, 500., MaxScale=1., MaxVel=1.,
                    filename='brute_force')
                np.testing.assert_array_equal(labels, expected)

    def test_sky_distances_use_proper_mpc_per_h(self):
        mass = np.array([10.001, 10., 8., 8.])
        redshift = np.full(4, .04)
        radius = 1.5 * censat._stellar_to_halo_mock(mass, 'Msun_log10')[1][0]
        distance = Planck15.angular_diameter_distance(.04).to_value('Mpc') * censat._H0
        offset = ra_offset(.99 * radius, distance)
        # Low-mass anchors keep the sky grid extent independent of pair spacing.
        labels = self.sky_labels([60., 60. + offset, 0., 180.],
                                 [30., 30., 0., 0.], redshift, mass)
        np.testing.assert_array_equal(labels, [0, 1])
        # Either dropping h or using comoving distances would put the pair outside.
        self.assertGreater(.99 * radius / censat._H0, radius)
        self.assertGreater(.99 * radius * (1 + redshift[0]), radius)

    def test_sky_radians_and_degrees_agree(self):
        ra = np.array([60., 60.05, 0., 180.])
        dec = np.array([30., 30., 0., 0.])
        redshift = np.full(4, .03)
        mass = np.array([10.001, 10., 8., 8.])
        degrees = self.sky_labels(ra, dec, redshift, mass)
        radians = self.sky_labels(np.deg2rad(ra), np.deg2rad(dec), redshift, mass,
                                  ra_dec_units='radian')
        np.testing.assert_array_equal(degrees, [0, 1])
        np.testing.assert_array_equal(radians, degrees)

    def test_smaller_nearer_central_cannot_overwrite_earlier_central(self):
        mass = np.array([10.001, 10., 8., 8.])
        radius = 1.5 * censat._stellar_to_halo_mock(mass, 'Msun_log10')[1]
        velcut = 2. * censat._stellar_to_halo_mock(mass, 'Msun_log10')[2]
        redshift = np.array([redshift_at_distance(100.), redshift_at_distance(99.), .02, .02])
        separation_from_a = 1.003 * radius[0]
        separation_from_b = separation_from_a * 99. / 100.
        self.assertGreater(radius[0], radius[1])
        self.assertGreater(separation_from_a, radius[0])
        self.assertLess(separation_from_b, radius[1])
        self.assertLess(3e5 * abs(redshift[0] - redshift[1]), min(velcut[:2]))
        offset = ra_offset(separation_from_a, 100.)
        labels = self.sky_labels([60., 60. + offset, 0., 180.],
                                 [30., 30., 0., 0.], redshift, mass)
        np.testing.assert_array_equal(labels, [0, 0])

    def test_simulation_periodic_position_and_hubble_velocity(self):
        pos = np.array([[.05, .05, .05], [9.95, 9.95, 9.95]])
        velocity = np.zeros((2, 3))
        velocity[1, 2] = 5.
        mass = np.array([2., 1.])
        for numbins in (1, 2, 10):
            with self.subTest(numbins=numbins):
                labels = censat.CenSat_Simulations(
                    pos, velocity, mass, 99, mass, numbins, 10.,
                    MaxScale=.2, nRvirOrMpc='Mpc', MaxVel=20., nVvirORkm='km',
                    filename='periodic')
                np.testing.assert_array_equal(labels, [0, 1])

    def test_simulation_upper_boundary_matches_periodic_origin(self):
        mass = np.array([2., 1.])
        for numbins in (1, 2, 10):
            for central in ([0., 0., 0.], [10., 0., 0.],
                            [0., 10., 0.], [10., 10., 0.]):
                with self.subTest(numbins=numbins, central=central):
                    labels = censat.CenSat_Simulations(
                        np.array([central, [.1, .1, 0.]]),
                        np.zeros((2, 3)), mass, 99, mass, numbins, 10.,
                        MaxScale=.2, nRvirOrMpc='Mpc',
                        MaxVel=20., nVvirORkm='km', filename='upper_boundary')
                    np.testing.assert_array_equal(labels, [0, 1])
                    self.assert_saved_labels('upper_boundary.hdf5', labels)

    def test_output_retains_selected_input_order(self):
        pos = np.array([[1.1, 1., 1.], [5., 5., 5.], [1., 1., 1.], [8., 8., 8.]])
        mass = np.array([1., .1, 3., .5])
        labels = censat.CenSat_Simulations(
            pos, np.zeros((4, 3)), mass, 99, mass, 10, 10.,
            MaxScale=.5, nRvirOrMpc='Mpc', MaxVel=20., nVvirORkm='km',
            LimPar_min=.2, filename='row_order')
        np.testing.assert_array_equal(labels, [1, 0, 0])
        self.assert_saved_labels('row_order.hdf5', labels)

    def native_fixture(self):
        chunks = {}
        for chunk_number, (mass, flag, first_subhalo) in enumerate([
                ([1., .5, .25], [True, False, True], [0, -1]),
                ([2., 1.5], [True, True], [3])]):
            path = Path('/native/groups_099/fof_subhalo_tab_099.%d.hdf5' % chunk_number)
            buffer = io.BytesIO()
            with HDF5_FILE(buffer, 'w') as f:
                header = f.create_group('Header').attrs
                header.update(BoxSize=75000., HubbleParam=.7, Nsubgroups_Total=5,
                              NumFiles=2, Redshift=0.)
                masses = np.zeros((len(mass), 6))
                masses[:, 4] = mass
                f['Subhalo/SubhaloMassType'] = masses
                f['Subhalo/SubhaloPos'] = np.full((len(mass), 3), 1000. * (chunk_number + 1))
                f['Subhalo/SubhaloVel'] = np.zeros((len(mass), 3))
                f['Subhalo/SubhaloFlag'] = flag
                f['Group/GroupFirstSub'] = first_subhalo
            chunks[path] = buffer
        return chunks

    def test_native_reader_uses_global_truth_indices_header_and_flags(self):
        chunks = self.native_fixture()

        def open_chunk(path, mode):
            self.assertEqual(mode, 'r')
            return HDF5_FILE(chunks[Path(path)], 'r')

        with mock.patch.object(Path, 'is_dir', return_value=False), \
                mock.patch.object(Path, 'glob', return_value=list(reversed(chunks))), \
                mock.patch.object(examples.h5py, 'File', side_effect=open_chunk):
            catalogue = examples._native_tng100_catalogue('/native/groups_099')
            all_subhalos = examples._native_tng100_catalogue(
                '/native/groups_099', include_subhalo_clumps=True)
        self.assertEqual(catalogue['boxsize'], 75.)
        self.assertEqual(catalogue['h'], .7)
        np.testing.assert_array_equal(catalogue['subhalo_id'], [0, 2, 3, 4])
        np.testing.assert_array_equal(catalogue['true_type'], [0, 1, 0, 1])
        np.testing.assert_array_equal(catalogue['pos'][:, 0], [1., 1., 2., 2.])
        np.testing.assert_array_equal(all_subhalos['subhalo_id'], [0, 1, 2, 3, 4])
        np.testing.assert_array_equal(all_subhalos['true_type'], [0, 1, 1, 0, 1])


if __name__ == '__main__':
    unittest.main()
