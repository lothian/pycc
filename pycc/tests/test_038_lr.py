"""
Test CCSD equation solution using various molecule test cases.
"""

# Import package, test suite, and other packages as needed
import psi4
import pycc
import pytest
import numpy as np
from ..data.molecules import *

np.set_printoptions(precision=15, linewidth=200, threshold=200, suppress=True)

def test_ccsd_h2o():
    # Psi4 Setup
    psi4.set_memory('2 GB')
    psi4.core.set_output_file('output.dat', False)
    psi4.set_options({'basis': 'cc-pVDZ',
                      'scf_type': 'pk',
                      'mp2_type': 'conv',
                      'freeze_core': 'false',
                      'e_convergence': 1e-12,
                      'd_convergence': 1e-12,
                      'r_convergence': 1e-12,
                      'diis': 1})
    mol = psi4.geometry("""
O          -0.947809457408    -0.132934425181     0.000000000000
H          -1.513924046286     1.610489987673     0.000000000000
F           0.878279174340     0.026485523618     0.000000000000
unit bohr
noreorient
symmetry c1
""")

    rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)

    maxiter = 75
    e_conv = 1e-12
    r_conv = 1e-12
    
    cc = pycc.ccwfn(rhf_wfn)
    eccsd = cc.solve_cc(e_conv,r_conv,maxiter)
    hbar = pycc.cchbar(cc)
    cclambda = pycc.cclambda(cc, hbar)
    lccsd = cclambda.solve_lambda(e_conv, r_conv)
    density = pycc.ccdensity(cc, cclambda)
    resp = pycc.ccresponse(density)
    resp.linresp("MU", "MU", 0.05)
