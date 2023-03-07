import h5py

def decode_hdf5_list(data):
    out_list = []
    for v in data:
        out_list.append(v.decode())
    return out_list

# with h5py.File('test_instructions.hdf5', "r") as inst_file:
#      data = inst_file['instructions'][()]
#      instructions = decode_hdf5_list(data)
#      print(f"instructions:{instructions}")

with h5py.File('test_data.hdf5', "r") as f:
    # source_skill
    source_skill = {}
    src_skill_hdf5 = f['source_skill']
    for key in src_skill_hdf5:
        temp = decode_hdf5_list(src_skill_hdf5[key])
        source_skill[key] = temp

    # target_skill
    tgt_skill_hdf5 = f['target_skill']
    target_skill = decode_hdf5_list(tgt_skill_hdf5)
    
    
    obs = f['obs'][()]
    action = f['action'][()]
    skill_idx = f['skill_idx'][()]
    
    print(f"source_skill: {source_skill}")
    print(f"target_skill: {target_skill}")
    print(f"observation shape: {obs.shape}")
    print(f"action shape: {action.shape}")
    print(f"skill index: {skill_idx}")
    

