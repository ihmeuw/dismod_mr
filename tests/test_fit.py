""" Test fit functions
"""

import dismod_mr

def test_fit_asr():
    dm = dismod_mr.data.ModelData()
    dm.vars = dismod_mr.model.process.age_specific_rate(dm, 'p')
    
    dismod_mr.fit.asr(dm, 'p', iter=10, burn=5, thin=1)

def test_fit_consistent():
    dm = dismod_mr.data.ModelData()
    dm.vars = dismod_mr.model.process.consistent(dm)
    
    dismod_mr.fit.consistent(dm, iter=10, burn=5, thin=1)

def test_check_convergence():
    dm = dismod_mr.data.ModelData()
    dm.vars = dismod_mr.model.process.age_specific_rate(dm, 'p')
    
    dismod_mr.fit.asr(dm, 'p', iter=110, burn=5, thin=1)
    dismod_mr.data.check_convergence(dm.vars)  # TODO: refactor this check into the model class, and move this test, too

if __name__ == '__main__':
    import nose
    nose.runmodule()
    
