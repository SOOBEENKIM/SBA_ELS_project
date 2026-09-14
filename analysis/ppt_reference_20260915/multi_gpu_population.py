"""Resume the unchanged population engine using one worker per visible GPU."""
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from multiprocessing import current_process
import torch
import run_population as population

def bind_gpu():
    # Spawned pool worker identities are 1, 2, ... in this standalone process.
    count=torch.cuda.device_count()
    if not count:raise RuntimeError('No visible CUDA devices')
    index=(current_process()._identity[-1]-1)%count
    torch.cuda.set_device(index)
    print('POPULATION worker device',index,torch.cuda.get_device_name(index),flush=True)

if __name__=='__main__':
    population.ProcessPoolExecutor=partial(ProcessPoolExecutor,initializer=bind_gpu)
    population.main()
