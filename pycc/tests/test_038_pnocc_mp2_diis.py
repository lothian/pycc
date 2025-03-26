"""
Test basic PNO-CCSD energy code with DIIS on first-order amplitude iterations
"""

# Import package, test suite, and other packages as needed
import psi4
import pycc
import pytest
from ..data.molecules import *

def test_pno_ccsd():
    """H2O PNO-CCSD Test"""
    # Psi4 Setup
    psi4.set_memory('2 GB')
    psi4.core.set_output_file('output.dat', False)
    psi4.set_options({'basis': 'cc-pVDZ',
                      'scf_type': 'pk',
                      'mp2_type': 'conv',
                      'freeze_core': 'false',
                      'e_convergence': 1e-13,
                      'd_convergence': 1e-13,
                      'r_convergence': 1e-13,
                      'diis': 1})
    mol = psi4.geometry(moldict["H2O_Teach"])
    rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)

    maxiter = 75
    e_conv = 1e-12
    r_conv = 1e-12

    ccsd = pycc.ccwfn(rhf_wfn, local='PNO', local_mos='BOYS', local_cutoff=1e-7, max_diis=8, start_diis=1)
    eccsd_diis_on = ccsd.solve_cc(e_conv, r_conv, maxiter)

    ccsd = pycc.ccwfn(rhf_wfn, local='PNO', local_mos='BOYS', local_cutoff=1e-7, max_diis=0, start_diis=1)
    eccsd_diis_off = ccsd.solve_cc(e_conv, r_conv, maxiter)

    #epycc = -0.214347601414963

    #assert (abs(epycc - eccsd_diis_on) < 1e-8)
    #assert (abs(epycc - eccsd_diis_off) < 1e-8)
