import progressbar

def setupBar(bar: progressbar.ProgressBar, totalframes):
    bar = progressbar.ProgressBar()
    bar.max_value = totalframes
    bar.update(1)
    return bar

def updateBar(bar: progressbar.ProgressBar, val):
    bar.update(val)

