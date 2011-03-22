__all__ = ['run', 'configure', 'checkpoint']

try:
    import stackless
except ImportError:
    stackless = None

rundoc = r"""
    Runs function `func` with the given arguments on jobman
    channel `chan`.

    This function will ignore the passed function if a resume file
    is present in the current directory and resume from that
    instead.  (This functionality is not available when stackless
    is not installed.)

    Only one function may be run at the same time. (Also known as
    'run() is not reentrant')
    """

configdoc = r"""
    Sets up the environnement for the checkpointing code.
    """

checkdoc = r"""
    Performs a checkpoint by querying jobman about run status and
    saving a resume file if needed.

    This function must only be called through function that was
    passed to run() or it will not work (actual error may vary).

    The resume file functionality requires stackless to work.
    """
if stackless is None:
    import warnings
    channel = None

    def run(chan, func, *args, **kwargs):
        global channel
        channel = chan
        func(*args, **kwargs)
        channel.save()

    def configure(fsave=None, fload=None):
        warnings.warn("Not running on stackless, resume disabled",
                      RuntimeWarning)

    def checkpoint():
        mess = channel.switch()
        if mess is not None:
            channel.save()
            if mess == 'stop' or mess == 'finish-up':
                pass # Should stop here ?

else:
    import time, pickle, os
    
    save = None
    load = None

    def run(channel, func, *args, **kwargs):
        r"""
        Runs function `func` with the given arguments on jobman
        channel `chan`.
        """
        try:
            with open('state.ckpt', 'rb') as f:
                t = load(f)
            assert t.restorable
            t.insert()
        except IOError:
            t = stackless.tasklet(func)(*args, **kwargs)

        while stackless.getruncount() > 1:
            stackless.schedule()
            mess = channel.switch()
            if mess is not None:
                with open('state.tmp', 'wb') as f:
                    save(t, f)
                os.rename('state.tmp', 'state.cpkt')
                channel.save()
                if mess == 'stop' or mess == 'finish-up':
                    t.kill()
                    return
        channel.save()

    def default_fsave(o, f):
        pickle.dump(o, f, -1)

    def configure(fsave=default_fsave, fload=pickle.load):
        global save, load

        save = fsave
        load = fload
    
    def checkpoint():
        stackless.schedule()

run.__doc__ = rundoc
configure.__doc__ = configdoc    
checkpoint.__doc__ = checkdoc
