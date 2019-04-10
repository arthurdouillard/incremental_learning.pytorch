from inclearn import parser
from inclearn.train import train

args = parser.get_parser().parse_args()
args = vars(args)  # Converting argparse Namespace to a dict.

train(args)
