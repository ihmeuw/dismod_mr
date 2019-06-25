import glob, os, subprocess, sys

dname = os.path.dirname(os.path.abspath(__file__))
fname_pattern = os.path.join(dname, '*.ipynb')
fnames = glob.glob(fname_pattern)

for fname in sorted(fnames):
    log = f'/ihme/scratch/users/abie/projects/2019/dismod_mr_examples.txt'

    fname_short = os.path.basename(fname)
    print(fname_short)
    name_str = f'dismod_mr_example_{fname_short}'

    call_str = 'qsub -l fthread=1 -l m_mem_free=10G -P ihme_general -q all.q -l archive=TRUE -cwd -o {0} -e {0} '.format(log) \
        + '-N %s ' % name_str \
        + f'{dname}/run_nb_on_cluster.sh {fname}'
    print(call_str)
    subprocess.call(call_str, shell=True)
    print()

print('To monitor progress on cluster, type:')
print(f'qstat |grep {name_str[:6]}')
print(f'tail -f {log}')
