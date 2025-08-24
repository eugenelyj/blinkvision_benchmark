import os


def is_ignore_scene(scene_pattern_a, ignore_list):
    for ignore_scene in ignore_list:
        if ignore_scene in scene_pattern_a:
            return True
    return False

def get_seq_list(mapping_file, input_path):
    seq_list = []
    ignore_list = ['indoor_tour/AI48_003_people']
    with open(mapping_file, 'r') as f:
        lines = f.readlines()
        data = [line.strip('\n').split(', ') for line in lines]
        for (scene_pattern_a, scene_pattern_b) in data:
            if is_ignore_scene(scene_pattern_a, ignore_list):
                continue

            category, seq = scene_pattern_b.rsplit('_', 1)
            if os.path.exists(os.path.join(input_path, scene_pattern_a)):
                seq_list.append(scene_pattern_a)
            else:
                seq_list.append(f'{category}/{seq}')
    
    return seq_list

