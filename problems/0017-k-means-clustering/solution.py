def calc_dist(point_1: tuple[float, float], point_2: tuple[float, float]) -> float:
    dist = ((point_1[0]-point_2[0])**2 + (point_1[1]-point_2[1])**2)**(0.5)
    return dist

def calc_centroid(points: list[tuple[float, float]]) -> tuple[float, float]:
    # all points have the same number of coordinates
    num_cordinates = len(points[0])

    centroid = []
    for i in range(num_cordinates):
        coordinate = sum([point[i] for point in points])/len(points)
        centroid.append(coordinate)

    return tuple(centroid)

def k_means_clustering(points: list[tuple[float, float]], k: int, initial_centroids: list[tuple[float, float]], max_iterations: int) -> list[tuple[float, float]]:

    centroids = initial_centroids

    clusters = [[] for i in range(len(centroids))]

    for i in range(max_iterations):
        final_centroids = []

        for point in points:
            min_dist = 1e6
            for i in range(len(centroids)):
                dist = c