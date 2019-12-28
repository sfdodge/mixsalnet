import progressbar

def my_pbar(name = "Progress:", maxval=None):
    widgets = [name, progressbar.Percentage(), ' ', progressbar.Bar(),
                                ' ', progressbar.ETA()]
    pbar1 = progressbar.ProgressBar(widgets=widgets,term_width=80,maxval=maxval)

    return pbar1
