import math
import numpy as np

positions = [[8.656796752300236, 49.87058086153881], [9.142540251381442, 49.97175601304856], [8.93133822115928, 50.11801159779943], [8.578497608269988, 50.08328838740962], [8.646369651208078,50.12235240252904],
             [8.081106373198622, 52.264883415484306], [9.94879869561957, 51.53536790399698], [11.116255398951209, 49.420273781762134], [11.61508215616684, 48.10910385981459], [11.47385227102177, 47.27862835730057],
             [ 13.233791117931064, 48.24492651087162], [9.149507269908895, 48.77038341089722], [7.813324053295927, 47.986096990689965], [8.508605420005281, 47.37153191674584], [9.19223402637872, 47.66794977267046],
             [11.770997392536286, 47.69202737706609], [10.022639173285942, 53.52530744793504], [14.571158712242685, 53.44891868140367], [4.443465406623602, 51.21149195329332], [4.4108028586426755, 50.83758776569975],
             [13.791204788464166, 51.048106701629315], [13.717905386458572, 51.03547857990612], [13.72966418982481, 51.05091248345532], [13.761335711300294, 51.07017066586084], [9.42881931835869, 54.804311216581986]]

def compute_distance(lon1, lat1, lon2, lat2):
    radius = 6370000
    pi = 3.14159
    lat = (lat2 - lat1) * (pi/180)
    lon = (lon2 - lon1) * (pi/180)
    a = math.sin(lat / 2) * math.sin(lat / 2) + math.cos(lat1 * (pi/180)) * math.cos(lat2 * (pi/180)) * math.sin(lon / 2) * math.sin(lon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c

    return abs(d)

def generate_cost_matrix(positions_array, positions_count, random_seed=None, verbosity=1):
    if random_seed is not None:
        np.random.seed(random_seed)
    np.random.shuffle(positions_array)
    if verbosity >= 2:
        print(positions_array)

    cost_matrix = []
    for position1 in positions_array[:positions_count]:
        current_pos_costs_array = []
        for position2 in positions_array[:positions_count]:
            distance = compute_distance(position1[0], position1[1], position2[0], position2[1])
            current_pos_costs_array.append(math.floor(distance))
        cost_matrix.append(current_pos_costs_array)
    
    if verbosity >= 2:
        print(cost_matrix)
    return cost_matrix


# Example: generate quadratic cost matrix 25x25 with random seed 42. Create reproducible other 25x25 cost matrix by changing random seed
generate_cost_matrix(positions, 25, 42)
