import argparse
import numpy as np
from matplotlib.pylab import plt

parser = argparse.ArgumentParser()
parser.add_argument('--files', nargs='+', help='list of files from which to build graphs', required=True)
parser.add_argument("--size", type=int, default=200, help="num of data points in line")
parser.add_argument("--type", type=str, default='epoch', help="episodes or epochs")
args = parser.parse_args()

LABELS = ['one hot 1 bin', 'one hot 2 bin', 'one hot 4 bin', 'one hot 10 bin']

allY = []
for file_name in args.files:
   Y = []
   with open(file_name) as fp:
      for idx, line in enumerate(fp):
         if len(Y) > args.size:
            break
         if args.type == 'epoch':
            if 'Epoch' not in line:
               continue
         else:
            if 'Episode' not in line:
               continue
         
         Y.append(float(line.split(' ')[-1][:-1]))
   allY.append(Y)

fig = plt.figure()
ax = plt.subplot(111)

plt.xlabel('epochs' if args.type == 'epoch' else 'episodes')
plt.ylabel('reward per epoch' if args.type == 'epoch' else 'reward per episode')
for idx, Y in enumerate(allY):
   ax.plot(Y, label=LABELS[idx])

plt.title('Instruction following learning performance one hot')
ax.legend(loc='upper center', shadow=True, ncol=2)
# plt.legend()
plt.show()
   
         