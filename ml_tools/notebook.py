from IPython import display
from tqdm.notebook import tqdm
from matplotlib import pyplot as plt

'''
File containing functionality related to jupyter notebook displays.
'''
class npi:
    '''
    Notebook plot iterator wraps around another iterator and calls
    plot_fn every `freq` timesteps.
    '''
    def __init__(self, iterator, plot_fn, freq = 50):
        self.iterator = iterator
        self.__it = None
        
        self.plot_fn = plot_fn
        self.freq = freq
        
        self.__i = 0
        self.display_handle = None
    
    def __iter__(self):
        self.__i = 0
        self.__it = self.iterator.__iter__()
        return self

    def __next__(self):
        if self.__i % self.freq == 0:
            self.plot_fn()
            if self.display_handle == None:
                self.display_handle = display.display(plt.gcf(), display_id = True)
            else:
                self.display_handle.update(plt.gcf())
            plt.close()
        self.__i += 1
        return self.__it.__next__()


class npi_tqdm(npi):
    '''
    Same as above but automatically also wraps a tqdm progress bar.
    '''
    def __init__(self, iterator, plot_fn, freq = 50):
        super().__init__(tqdm(iterator), plot_fn, freq)