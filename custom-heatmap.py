import sys
from matplotlib import pyplot as pp
from matplotlib import patches

def box(ax, n, x, y):
    opacity = max(min(1000.0, n) / 1000.0, 0.05) if n > 0 else 0
    ax.add_patch(patches.Rectangle((x, y), 1, 1, alpha=opacity, edgecolor="none"))

def customHeatmap(ax, matrix):
    for y, line in enumerate(matrix):
        for x, num in enumerate(line):
            box(ax, num, x, y)

if __name__ == "__main__":
	print "Reading " + sys.argv[1]
	with open(sys.argv[1]) as f:
		data = [l.strip().split(',') for l in f.readlines()]
		del data[0]
		for l in data:
			 del l[0]
		data = [[float(x) for x in l] for l in data]

	print "Plotting..."
	fig, ax = pp.subplots()
	# Plot custom heatmap
	customHeatmap(ax, data)
	pp.xlim(0, 40)
	pp.ylim(0, 40)
	ax.invert_yaxis()
	ax.set_xlabel("Predicted")
	ax.set_ylabel("Actual")
	# Plot class separation lines
	ax.plot([0,40], [23, 23], color='r', alpha=0.5)
	ax.plot([23,23], [0, 40], color='r', alpha=0.5)
	# Plot grid
	for i in range(1, 40):
		# Horizontal
		ax.plot([0, 40], [i, i], color='k', alpha=0.15)
		# Vertical
		ax.plot([i, i], [0, 40], color='k', alpha=0.15)
	fig.set_size_inches(8,8)
	pp.savefig('heatmap.png', format='png', dpi=400, bbox_inches="tight")
	print "Saved to heatmap.png"