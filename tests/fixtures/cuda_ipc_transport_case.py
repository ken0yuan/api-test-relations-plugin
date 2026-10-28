"""Synthetic transport checks, separate from production test identities."""
import torch
import torch.multiprocessing as mp


def consume(queue,result,release):
    try:
        tensor=queue.get(timeout=15)
        before=(str(tensor.device),tensor.requires_grad,tensor.storage_offset(),tuple(tensor.stride()),float(tensor.detach().sum()))
        with torch.no_grad():tensor.add_(1)
        torch.cuda.synchronize()
        result.put(before)
    finally:
        if 'tensor' in locals():del tensor
        release.set()


def produce(queue,release):
    tensor=torch.arange(10.,device='cuda')[2:6].requires_grad_(True)
    queue.put(tensor)
    if not release.wait(20):raise RuntimeError('Synthetic receiver did not release producer')


def reap(*children):
    for child in children:
        child.join(5)
        if child.is_alive():child.terminate();child.join(3)
        assert child.exitcode==0


def expected(result):
    assert result==('cuda:0',True,2,(1,),14.)


def test_parent_to_child_preserves_view_grad_and_shared_mutation():
    ctx=mp.get_context('spawn');queue=ctx.Queue();result=ctx.Queue();release=ctx.Event()
    base=torch.arange(10.,device='cuda');view=base[2:6].requires_grad_(True)
    child=ctx.Process(target=consume,args=(queue,result,release));child.start();queue.put(view)
    try:
        expected(result.get(timeout=20));assert release.wait(5)
        assert torch.equal(base[2:6],torch.tensor([3.,4.,5.,6.],device='cuda'))
    finally:release.set();reap(child)


def test_child_to_parent_preserves_view_and_storage():
    ctx=mp.get_context('spawn');queue=ctx.Queue();release=ctx.Event()
    child=ctx.Process(target=produce,args=(queue,release));child.start()
    try:
        tensor=queue.get(timeout=20)
        expected((str(tensor.device),tensor.requires_grad,tensor.storage_offset(),tuple(tensor.stride()),float(tensor.detach().sum())))
        del tensor
    finally:release.set();reap(child)


def test_sibling_to_sibling_preserves_view_and_grad():
    ctx=mp.get_context('spawn');queue=ctx.Queue();result=ctx.Queue();release=ctx.Event()
    sender=ctx.Process(target=produce,args=(queue,release));receiver=ctx.Process(target=consume,args=(queue,result,release))
    sender.start();receiver.start()
    try:expected(result.get(timeout=25));assert release.wait(5)
    finally:release.set();reap(sender,receiver)
