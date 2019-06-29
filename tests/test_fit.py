""" Test fit functions
"""


def test_fit_asr():
    dm = src.dismod_mr.data.ModelData()
    dm.vars = src.dismod_mr.model.process.age_specific_rate(dm, 'p')

    src.dismod_mr.fit.asr(dm, 'p', iter=10, burn=5, thin=1)

def test_fit_consistent():
    dm = src.dismod_mr.data.ModelData()
    dm.vars = src.dismod_mr.model.process.consistent(dm)

    src.dismod_mr.fit.consistent(dm, iter=10, burn=5, thin=1)

def test_check_convergence():
    dm = src.dismod_mr.data.ModelData()
    dm.vars = src.dismod_mr.model.process.age_specific_rate(dm, 'p')

    src.dismod_mr.fit.asr(dm, 'p', iter=110, burn=5, thin=1)
    src.dismod_mr.data.check_convergence(dm.vars)  # TODO: refactor this check into the model class, and move this test, too

if __name__ == '__main__':
    import nose
    nose.runmodule()

