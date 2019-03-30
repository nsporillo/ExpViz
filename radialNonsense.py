import scipy.spatial.kdtree as kdtree
import numpy as np

def radialJoin(img, stats, centroids):
    stats = np.append(stats,centroids,axis = 1)
    combine = np.array(sorted(stats, key=lambda x: (x[0], x[1])))
    centroids = list(combine[:,-2:])
    stats = list(combine[:,:-2])
    centroids = centroids[1:]
    stats = stats[1:]
    centroidTree = kdtree.KDTree(centroids,10)
    nearests = centroidTree.query(centroids,5)[1][:,1:]
    covered = {}
    results = []
    for x in range(len(nearests)):
        over = False
        sleft, stop, swidth, sheight, sarea = stats[x][0], stats[x][1], stats[x][2], stats[x][3], stats[x][4]
        if sarea<100:
            continue
        for y in range(4):
            if over:
                continue
            if x in covered or nearests[x][y] in covered:
                continue
            n = centroids[nearests[x][y]]
            c = centroids[x]

            nsleft, nstop, nswidth, nsheight, nsarea = stats[nearests[x][y]][0], stats[nearests[x][y]][1], \
                                                       stats[nearests[x][y]][2], stats[nearests[x][y]][3], stats[nearests[x][y]][4]
            if nsarea<100 or abs(nswidth-swidth)>100:
                continue
            if sleft-10 <= nsleft <= sleft + swidth+10 or nsleft-10 <= sleft <= nsleft + nswidth+10:
                bound = (min(sleft, nsleft), min(stop, nstop), max(sleft + swidth,nsleft + nswidth)-min(sleft,nsleft),
                         max(stop + sheight,nstop + nsheight)-min(stop,nstop), sarea+nsarea)
                # cv.rectangle(bounds, (bound[0], bound[1]), (bound[0]+bound[2], bound[1]+bound[3]), (0, 0, 255), thickness=2)
                results.append(np.array(bound,dtype=np.uint32))
                covered[nearests[x][y]] = ""
                print("connecting: " + str(n) + "," + str(c))
                over = True
        if not over and not x in covered:
            results.append(np.array(stats[x],dtype=np.uint32))
        covered[x] = ""
    return results