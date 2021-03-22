# AGG backend is for writing to file, not for rendering in a window.
# For more info, check this: https://matplotlib.org/2.0.2/faq/usage_faq.html
import matplotlib; matplotlib.use('Agg')

from inclearn import parser, train
