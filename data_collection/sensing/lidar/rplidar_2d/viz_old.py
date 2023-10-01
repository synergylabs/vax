from run import start2DLidar,stop2DLidar,readLidarData
import traceback
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

def update_line(num, scanIterator, line):
    try:
        scan = next(scanIterator)
        offsets = np.array([(np.radians(meas[1]), meas[2]) for meas in scan])
        line.set_offsets(offsets)
        intens = np.array([meas[0] for meas in scan])
        line.set_array(intens)
    except:
        ...
    return line,

lidar= start2DLidar()
DMAX = 4000
IMIN = 0
IMAX = 50
fig = plt.figure()
ax = plt.subplot(111, projection='polar')
line = ax.scatter([0, 0], [0, 0], s=5, c=[IMIN, IMAX],
                  cmap=plt.cm.Greys_r, lw=0)
ax.set_rmax(DMAX)
ax.grid(True)

try:
    scanIterator = lidar.iter_scans()
    ani = animation.FuncAnimation(fig, update_line,
                                  fargs=(scanIterator, line), interval=50)
    plt.show()
except:
    print("Closing 2D Lidar")
    traceback.print_exc()
    stop2DLidar(lidar)
