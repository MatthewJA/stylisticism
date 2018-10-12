"""Colourblind-friendly colour schemes for plotting.

Matthew Alger <matthew.alger@anu.edu.au>
ANU/Data61
"""

import colorspacious
import matplotlib.colors
import numpy
import scipy.optimize
import scipy.special

deuteranomaly = {
    'name': 'sRGB1+CVD',
    'cvd_type': 'deuteranomaly',
    'severity': 100}

def to_matplotlib(colourmap, discrete):
    """Convert a colourmap into a matplotlib colourmap."""
    if discrete:
        return matplotlib.colors.ListedColormap(
            colourmap)
    raise NotImplementedError()


def colourblind_colourmap(colourmap):
    """Convert a colourmap into a colourblind version."""
    return numpy.clip(colorspacious.cspace_convert(
        colourmap, deuteranomaly, 'sRGB1'), 0, 1)


def greyscale_colourmap(colourmap):
    """Convert a colourmap into a greyscale version."""
    jch = colorspacious.cspace_convert(colourmap, 'sRGB1', 'JCh')
    # Removing all chroma makes this greyscale.
    jch[:, 1] = 0
    return numpy.clip(colorspacious.cspace_convert(jch, 'JCh', 'sRGB1'), 0, 1)


def score_discrete_colourmap(colourmap):
    """How good does a discrete colourmap look?

    A lower score indicates a better colourmap.
    """
    # Add up all the distances.
    regular_vision_score = 0
    colourmap_ = colourmap
    colourmap = numpy.clip(colourmap, 0, 1)
    for colour_a in colourmap:
        for colour_b in colourmap:
            dE = colorspacious.deltaE(
                colour_a, colour_b, input_space='sRGB1')
            if dE == 0:
                continue
            regular_vision_score -= numpy.log(dE)

    deuteranomaly_score = 0
    deuteranomalous_colourmap = colourblind_colourmap(colourmap)
    for colour_a in deuteranomalous_colourmap:
        for colour_b in deuteranomalous_colourmap:
            dE = colorspacious.deltaE(
                colour_a, colour_b, input_space='sRGB1')
            if dE == 0:
                continue
            deuteranomaly_score -= numpy.log(dE)

    greyscale_score = 0
    grey_colourmap = greyscale_colourmap(colourmap)
    for colour_a in grey_colourmap:
        for colour_b in grey_colourmap:
            dE = colorspacious.deltaE(
                colour_a, colour_b, input_space='sRGB1')
            if dE == 0:
                continue
            greyscale_score -= numpy.log(dE)

    # Is anything white?
    white_score = 0
    for colour in grey_colourmap:
        dE = colorspacious.deltaE(
            colour, (1, 1, 1), input_space='sRGB1')
        white_score += numpy.exp(15 - dE)

    # # Is anything out of range?
    # range_score = 0
    # for colour in colourmap_:
    #     for i in colour:
    #         range_score += scipy.special.expit((i - 1) / 100)
    #         range_score += 1 - scipy.special.expit(i / 100)

    return {
        'grey': greyscale_score,
        'deuteranomaly': deuteranomaly_score,
        'regular': regular_vision_score,
        'white': white_score,
        # 'range': range_score,
    }


def _discrete_colourmap(n, seed=None):
    """Generate a colourmap for n classes.

    This version of the function is not adjusted for colourblindness.
    """

    # First generate a brightness gradient.
    # 1e-100 for numerical stability in colorspacious.
    rng = numpy.random.RandomState(seed=seed)
    maximum_brightness = 0.8  # Pretty arbitrary, but we don't want to get too close to 1.
    brightnesses = numpy.arange(1, n + 1) / n * maximum_brightness * 100 + 1e-100

    # Even distribution in hue space.
    hues = (numpy.arange(n) / n * 255 + rng.randint(255)) % 255
    colours_JCh = [[j, 100, h] for j, h in zip(brightnesses, hues)]
    return numpy.clip(colorspacious.cspace_convert(
        colours_JCh, 'JCh', 'sRGB1'), 0, 1)


def discrete_colourmap(n, seed=None):
    """Generate a colourmap for n classes."""
    # Initialise colourmap.
    cmap = _discrete_colourmap(n, seed=seed)
    # Then optimise the cost.
    def cost(colourmap):
        colourmap = colourmap.reshape((-1, 3))
        score = score_discrete_colourmap(colourmap)
        return score['grey'] * 10 + score['deuteranomaly'] + score['regular'] + score['white']
    best = scipy.optimize.fmin_bfgs(
        cost, cmap.ravel(), gtol=0.01)
    return numpy.clip(best.reshape((-1, 3)), 0, 1)


def plot_sample(cmap):
    """Plot a sample plot for a colourmap."""
    import matplotlib.pyplot
    lines = numpy.random.randint(100, size=(len(cmap), 3))

    fig = matplotlib.pyplot.figure()

    cmap_init = cmap
    for i, mod in enumerate([
         None,
         colourblind_colourmap,
         greyscale_colourmap]):
        cmap = cmap_init
        if mod:
            cmap = mod(cmap)
        ax = fig.add_subplot(2, 2, i + 1)
        for i in range(len(cmap)):
            ax.plot(range(3), lines[i],
                c=cmap[i], label=str(i))

    matplotlib.pyplot.show()



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    gradient = numpy.linspace(0, 1, 256)

    hues = numpy.array([(60, 45, g * 360) for g in gradient])
    hues_rgb = numpy.clip(colorspacious.cspace_convert(
        hues, 'JCh', 'sRGB1'), 0, 1)
    hues_deut = colorspacious.cspace_convert(
        hues_rgb, deuteranomaly, 'JCh')

    plt.plot(hues[:, 2], hues_deut[:, 2], label='h')
    plt.plot(hues[:, 1] + hues[:, 2], hues_deut[:, 1] + hues[:, 2], label='h + C')
    plt.show()
    # gradient = numpy.vstack((gradient,) * 100)
    # plt.imshow(gradient, cmap=to_matplotlib(cmap, True))
    # plt.show()
    # print(score_discrete_colourmap(cmap))
    # print(cmap)
    # plot_sample(cmap)
