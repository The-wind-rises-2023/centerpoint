from glob import glob
from os.path import join
from concurrent import futures as futures
import os.path as osp
import mmcv
import numpy as np
import multiprocessing

import pypcd.pypcd as pypcd

ROOTDIR = None


def process_info(info_path):
    # json example
    #[{"psr": {"position": {"x": 71.70016479492188, "y": 19.75863265991211, "z": 1.6592884063720703}, "scale": {"x": 4.199951171875, "y": 1.8000991344451904, "z": 1.6000524759292603}, "rotation": {"x": 0, "y": 0, "z": -3.1186392307281494}}, "obj_type": "Car", "obj_id": "1", "score": 0.6055406332015991}, {"psr": {"position": {"x": 98.10417175292969, "y": -7.810024261474609, "z": 1.4482817649841309}, "scale": {"x": 0.47279322147369385, "y": 0.4559081196784973, "z": 0.8469966650009155}, "rotation": {"x": 0, "y": 0, "z": -0.062082722783088684}}, "obj_type": "Static", "obj_id": "1", "score": 0.524918794631958}, {"psr": {"position": {"x": 63.80940246582031, "y": -17.4814395904541, "z": 0.86873859167099}, "scale": {"x": 4.199342250823975, "y": 1.7998732328414917, "z": 1.599823236465454}, "rotation": {"x": 0, "y": 0, "z": 3.1323139667510986}}, "obj_type": "Car", "obj_id": "1", "score": 0.42969828844070435}, {"psr": {"position": {"x": 105.7983627319336, "y": -4.664283752441406, "z": 1.077820062637329}, "scale": {"x": 0.5848323106765747, "y": 0.45018571615219116, "z": 0.8338186144828796}, "rotation": {"x": 0, "y": 0, "z": 0.006477335933595896}}, "obj_type": "Static", "obj_id": "1", "score": 0.4012830853462219}, {"psr": {"position": {"x": 67.41585540771484, "y": -4.633457183837891, "z": 1.2088322639465332}, "scale": {"x": 0.5703117251396179, "y": 0.5045447945594788, "z": 1.1644877195358276}, "rotation": {"x": 0, "y": 0, "z": 0.019120585173368454}}, "obj_type": "Static", "obj_id": "1", "score": 0.2956109344959259}, {"psr": {"position": {"x": 65.33258819580078, "y": 3.039295196533203, "z": 1.2684826850891113}, "scale": {"x": 0.534128725528717, "y": 0.4223622977733612, "z": 0.8899688720703125}, "rotation": {"x": 0, "y": 0, "z": 0.004217769484966993}}, "obj_type": "Static", "obj_id": "1", "score": 0.2474672794342041}, {"psr": {"position": {"x": 62.596763610839844, "y": -12.5797119140625, "z": 0.9038380980491638}, "scale": {"x": 0.6705108284950256, "y": 0.7062638401985168, "z": 1.3088946342468262}, "rotation": {"x": 0, "y": 0, "z": 0.008213140070438385}}, "obj_type": "Static", "obj_id": "1", "score": 0.24214953184127808}, {"psr": {"position": {"x": 46.92658615112305, "y": 2.7369918823242188, "z": 0.6010438203811646}, "scale": {"x": 0.47479575872421265, "y": 0.359687477350235, "z": 0.8456016778945923}, "rotation": {"x": 0, "y": 0, "z": -0.014253559522330761}}, "obj_type": "Static", "obj_id": "1", "score": 0.23259122669696808}, {"psr": {"position": {"x": 50.19202423095703, "y": 2.8515090942382812, "z": 0.8010841608047485}, "scale": {"x": 0.5368579030036926, "y": 0.39928704500198364, "z": 0.9185896515846252}, "rotation": {"x": 0, "y": 0, "z": -0.0069815535098314285}}, "obj_type": "Static", "obj_id": "1", "score": 0.22934569418430328}, {"psr": {"position": {"x": 68.52100372314453, "y": 3.066009521484375, "z": 0.6821929812431335}, "scale": {"x": 0.41159576177597046, "y": 0.33152151107788086, "z": 0.7518446445465088}, "rotation": {"x": 0, "y": 0, "z": -0.018619338050484657}}, "obj_type": "Static", "obj_id": "1", "score": 0.22329597175121307}, {"psr": {"position": {"x": 77.80522918701172, "y": 3.28936767578125, "z": 0.8458728194236755}, "scale": {"x": 0.5810744166374207, "y": 0.380737841129303, "z": 0.8113184571266174}, "rotation": {"x": 0, "y": 0, "z": -0.004680976737290621}}, "obj_type": "Static", "obj_id": "1", "score": 0.21736499667167664}, {"psr": {"position": {"x": 51.73360061645508, "y": -4.902744293212891, "z": 1.292011022567749}, "scale": {"x": 0.46955427527427673, "y": 0.46450039744377136, "z": 1.0803797245025635}, "rotation": {"x": 0, "y": 0, "z": 1.075319766998291}}, "obj_type": "Static", "obj_id": "1", "score": 0.21556049585342407}, {"psr": {"position": {"x": 56.22526931762695, "y": 2.9086456298828125, "z": 0.9152705669403076}, "scale": {"x": 0.5472469925880432, "y": 0.3815898299217224, "z": 0.8734356760978699}, "rotation": {"x": 0, "y": 0, "z": 0.0032639577984809875}}, "obj_type": "Static", "obj_id": "1", "score": 0.21226933598518372}, {"psr": {"position": {"x": 94.70257568359375, "y": -0.299530029296875, "z": 1.5631461143493652}, "scale": {"x": 0.42103126645088196, "y": 0.4051464796066284, "z": 0.8375328779220581}, "rotation": {"x": 0, "y": 0, "z": -0.018962709233164787}}, "obj_type": "Static", "obj_id": "1", "score": 0.21109190583229065}, {"psr": {"position": {"x": 64.55624389648438, "y": -4.700847625732422, "z": 1.290168285369873}, "scale": {"x": 0.5960432291030884, "y": 0.5515086054801941, "z": 1.445867657661438}, "rotation": {"x": 0, "y": 0, "z": -3.0326693058013916}}, "obj_type": "Static", "obj_id": "1", "score": 0.20595280826091766}, {"psr": {"position": {"x": 61.88581085205078, "y": -17.38518714904785, "z": 0.7702216506004333}, "scale": {"x": 0.3752295970916748, "y": 0.38465267419815063, "z": 0.7665359377861023}, "rotation": {"x": 0, "y": 0, "z": -0.006212360691279173}}, "obj_type": "Static", "obj_id": "1", "score": 0.20266996324062347}, {"psr": {"position": {"x": 71.79183197021484, "y": 3.0645179748535156, "z": 0.8135709762573242}, "scale": {"x": 0.7364543676376343, "y": 0.3853878080844879, "z": 0.8290591239929199}, "rotation": {"x": 0, "y": 0, "z": -0.012801030650734901}}, "obj_type": "Static", "obj_id": "1", "score": 0.2011706829071045}]

    filename = osp.splitext(osp.basename(info_path))[0]
    info = mmcv.load(info_path)
    # process info
    gt_bboxes_3d = []
    gt_labels = []

    for idx, obj in enumerate(info):
        gt_labels.append(obj['obj_type'])
        obj = obj['psr']
        one_box = []
        for key in ('position', 'scale'):
            one_box.extend([float(obj[key][v]) for v in ('x', 'y', 'z')])
        #yaw
        one_box.append(obj['rotation']['z'])

        gt_bboxes_3d.append(one_box)

    gt_bboxes_3d = np.array(gt_bboxes_3d, dtype=np.float32)
    result = {
        'sample_idx': filename,
        'annos': {'box_type_3d': 'lidar',
                  'gt_bboxes_3d': gt_bboxes_3d, 'gt_names': gt_labels},
        'calib': {},
        'images': {}
    }
    return result


def process_pcd(pcd_path):
    file_name = osp.splitext(osp.basename(pcd_path))[0]
    pc = pypcd.PointCloud.from_path(pcd_path)
    pts = np.stack([pc.pc_data['x'],
                    pc.pc_data['y'],
                    pc.pc_data['z']],
                   axis=-1)
    pts = np.concatenate([pts.astype(np.float32), np.expand_dims(
        pc.pc_data["intensity"].astype(np.float32), -1)], axis=1)
    global ROOTDIR
    save_path = osp.join(ROOTDIR, f"bin/{file_name}.bin")
    pts.tofile(save_path)
    return save_path


def process_single_data(info_path, pcd_path):
    # process anno info
    result = process_info(info_path)

    # process pcd
    save_bin_path = process_pcd(pcd_path)

    result.update(
        {'lidar_points': {'lidar_path': save_bin_path,
                          }})
    return result


def generate_pickle(infos_path, pcds_path, filename, num_workers=8, speed_up=True):
    # for io speed
    if speed_up:
        with futures.ProcessPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_data, infos_path, pcds_path)
    else:
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_data, infos_path, pcds_path)
    mmcv.dump(list(infos), filename)
